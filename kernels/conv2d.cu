#include "conv2d.h"

// --------------------------------------------------------------------------
// 1. CONVOLUTION KERNEL
// --------------------------------------------------------------------------
__global__ void conv2d_forward_kernel(
	const float *__restrict__ input,
	const float *__restrict__ weights,
	const float *__restrict__ bias,
	float *__restrict__ output,
	int batch, int in_channels, int out_channels,
	int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding)
{
	// Mỗi thread tính 1 pixel output tại vị trí (n, oc, oh, ow)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_elements = batch * out_channels * out_h * out_w;

	if (idx < total_elements)
	{
		// Giải mã index phẳng thành tọa độ 4D của Output
		int pw = idx % out_w;
		int tmp = idx / out_w;
		int ph = tmp % out_h;
		tmp /= out_h;
		int pc = tmp % out_channels; // Output channel
		int pn = tmp / out_channels; // Batch index

		// Khởi tạo giá trị tổng bằng Bias tương ứng với channel này
		float sum = bias[pc];

		// Duyệt qua Input Channels và Kernel
		for (int ic = 0; ic < in_channels; ++ic)
		{
			for (int kh = 0; kh < k_size; ++kh)
			{
				for (int kw = 0; kw < k_size; ++kw)
				{

					int h_in = ph * stride - padding + kh;
					int w_in = pw * stride - padding + kw;

					if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w)
					{
						// Tính index phẳng cho Input và Weight
						int in_idx = ((pn * in_channels + ic) * in_h + h_in) * in_w + w_in;
						int w_idx = ((pc * in_channels + ic) * k_size + kh) * k_size + kw;

						sum += input[in_idx] * weights[w_idx];
					}
				}
			}
		}
		output[idx] = sum;
	}
}

// 1. Khai báo bộ nhớ hằng (Giới hạn 64KB, đủ cho các layer nhỏ/trung bình)
// max kernel: 3x3, max in_channels: 256, max out_channels: 128
// Cần copy dữ liệu vào đây từ Host trước khi gọi kernel
// __constant__ float c_weights[3 * 3 * 256 * 128];
__constant__ float c_bias[1024];
// Kích thước Block cố định để tối ưu Shared Memory
#define TILE_WIDTH 16
#define KERNEL_SIZE 3
#define RADIUS 1 // (KERNEL_SIZE - 1) / 2
// Kích thước vùng Shared Memory cần thiết (16 + 2 = 18)
#define SHARED_WIDTH (TILE_WIDTH + 2 * RADIUS)

__global__ void conv2d_optimized_kernel(
	const float *__restrict__ input,
	float *__restrict__ output,
	const float *__restrict__ weights,
	int batch, int in_channels, int out_channels,
	int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding,
	bool use_relu) // Fusion
{
	// 1. Khai báo Shared Memory: [18][18]
	__shared__ float s_input[SHARED_WIDTH][SHARED_WIDTH];

	// Tọa độ của Thread trong Block
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Tọa độ đầu ra (Output) mà thread này phụ trách
	int pw = blockIdx.x * blockDim.x + tx;
	int ph = blockIdx.y * blockDim.y + ty;

	// Batch và Output Channel từ trục Z
	int b_c_idx = blockIdx.z;
	int pc = b_c_idx % out_channels;
	int pn = b_c_idx / out_channels;

	float sum = 0.0f;
	if (pc < out_channels && pn < batch)
		sum = c_bias[pc];

	// ---------------------------------------------------------
	// LOOP QUA CÁC INPUT CHANNELS (Tiling theo chiều sâu)
	// ---------------------------------------------------------
	for (int ic = 0; ic < in_channels; ++ic)
	{

		// --- GIAI ĐOẠN 1: LOAD DỮ LIỆU VÀO SHARED MEMORY ---
		// Vấn đề: Block có 256 threads (16x16) nhưng cần load 324 pixel (18x18)
		// Giải pháp: Mỗi thread load 1 pixel, một số thread load thêm pixel thứ 2.

		// Tọa độ gốc (top-left) của Tile trên Input Input
		// (Lưu ý: Tile Input phải dịch sang trái/lên trên do padding)
		int in_tile_start_w = blockIdx.x * blockDim.x - padding;
		int in_tile_start_h = blockIdx.y * blockDim.y - padding;

		// Flatten index của thread trong block (0 -> 255)
		int tid = ty * TILE_WIDTH + tx;

		// Tổng số pixel cần load
		int num_pixels_to_load = SHARED_WIDTH * SHARED_WIDTH; // 18*18 = 324

		// Vòng lặp để load (mỗi thread có thể load > 1 pixel)
		for (int i = tid; i < num_pixels_to_load; i += (TILE_WIDTH * TILE_WIDTH))
		{
			// Map index i (0..323) sang tọa độ Shared Memory (sy, sx)
			int sy = i / SHARED_WIDTH;
			int sx = i % SHARED_WIDTH;

			// Map sang tọa độ Global Input
			int cur_h = in_tile_start_h + sy;
			int cur_w = in_tile_start_w + sx;

			// Kiểm tra biên và Load
			if (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w)
			{
				int in_idx = ((pn * in_channels + ic) * in_h + cur_h) * in_w + cur_w;
				s_input[sy][sx] = input[in_idx];
			}
			else
			{
				s_input[sy][sx] = 0.0f; // Padding bằng 0
			}
		}

		// Đợi tất cả thread load xong Tile của channel hiện tại
		__syncthreads();

		// --- GIAI ĐOẠN 2: TÍNH TOÁN (COMPUTE) ---
		// Chỉ tính nếu thread nằm trong vùng output hợp lệ
		if (pw < out_w && ph < out_h && pn < batch)
		{

// Unroll loops cho kernel 3x3
#pragma unroll
			for (int kh = 0; kh < KERNEL_SIZE; ++kh)
			{
#pragma unroll
				for (int kw = 0; kw < KERNEL_SIZE; ++kw)
				{
					// Đọc từ Shared Memory (Cực nhanh)
					// Thread (ty, tx) cần dữ liệu lân cận -> (ty + kh, tx + kw)
					// Do Shared Memory đã bao gồm padding, ta truy cập trực tiếp
					float val = s_input[ty + kh][tx + kw];

					// Trọng số vẫn đọc từ Global (đã tối ưu bằng __restrict__ và L1 cache)
					// Hoặc Constant Memory nếu bạn đã cài đặt
					int w_idx = ((pc * in_channels + ic) * KERNEL_SIZE + kh) * KERNEL_SIZE + kw;
					sum += val * weights[w_idx];
				}
			}
		}

		// Đợi tất cả tính xong trước khi load channel tiếp theo vào Shared Mem
		__syncthreads();
	}

	// Ghi kết quả (Fusion ReLU luôn ở đây nếu muốn)
	if (pw < out_w && ph < out_h && pn < batch)
	{
		int out_idx = ((pn * out_channels + pc) * out_h + ph) * out_w + pw;
		if (use_relu)
		{
			// Áp dụng ReLU
			sum = fmaxf(0.0f, sum);
		}
		output[out_idx] = sum;
	}
}

// ======================================================================
// 3. CONVOLUTION BACKWARD KERNELS
// ======================================================================

// 3a. Tính Gradient theo Input (dX) - Truyền lỗi về lớp trước
// Thực chất là tích chập (Full Convolution) giữa grad_output và weights
__global__ void conv2d_backward_input_kernel(
	const float *grad_output, const float *weights, float *grad_input,
	int batch, int in_c, int out_c, int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_in = batch * in_c * in_h * in_w;

	if (idx < total_in)
	{
		// Giải mã index input
		int pw = idx % in_w;
		int tmp = idx / in_w;
		int ph = tmp % in_h;
		tmp /= in_h;
		int pc = tmp % in_c;
		int pn = tmp / in_c;

		float d_val = 0.0f;

		// Duyệt qua tất cả các output pixels có thể ảnh hưởng tới input pixel này
		// (Logic đảo ngược của Forward)
		for (int oc = 0; oc < out_c; ++oc)
		{
			for (int kh = 0; kh < k_size; ++kh)
			{
				for (int kw = 0; kw < k_size; ++kw)
				{
					// Tìm vị trí trên output map (oh, ow) sao cho khi convolution nó chạm vào (ph, pw)
					// ph = oh * stride - padding + kh  =>  oh * stride = ph + padding - kh

					int h_val = ph + padding - kh;
					int w_val = pw + padding - kw;

					if (h_val % stride == 0 && w_val % stride == 0)
					{
						int oh = h_val / stride;
						int ow = w_val / stride;

						if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w)
						{
							int out_idx = ((pn * out_c + oc) * out_h + oh) * out_w + ow;
							int w_idx = ((oc * in_c + pc) * k_size + kh) * k_size + kw;

							d_val += grad_output[out_idx] * weights[w_idx];
						}
					}
				}
			}
		}
		grad_input[idx] = d_val;
	}
}

// 3b. Tính Gradient theo Weights (dW)
// Mỗi thread tính 1 phần tử của Weight Gradient
__global__ void conv2d_backward_weight_kernel(
	const float *input, const float *grad_output, float *grad_weights,
	int batch, int in_c, int out_c, int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_weights = out_c * in_c * k_size * k_size;

	if (idx < total_weights)
	{
		// Giải mã index weight: [out_c, in_c, kh, kw]
		int kw = idx % k_size;
		int tmp = idx / k_size;
		int kh = tmp % k_size;
		tmp /= k_size;
		int ic = tmp % in_c;
		int oc = tmp / in_c;

		float sum_dw = 0.0f;

		// Duyệt qua toàn bộ batch và toàn bộ output spatial dimensions
		// (Đây là phần nặng nhất, Phase 3 sẽ tối ưu phần này bằng Im2Col hoặc GEMM)
		for (int b = 0; b < batch; ++b)
		{
			for (int oh = 0; oh < out_h; ++oh)
			{
				for (int ow = 0; ow < out_w; ++ow)
				{

					int ih = oh * stride - padding + kh;
					int iw = ow * stride - padding + kw;

					if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
					{
						int in_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
						int out_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;

						sum_dw += input[in_idx] * grad_output[out_idx];
					}
				}
			}
		}
		grad_weights[idx] = sum_dw;
	}
}

// 3c. Tính Gradient theo Bias (db)
// Mỗi thread tính 1 phần tử bias (1 output channel)
__global__ void conv2d_backward_bias_kernel(
	const float *grad_output, float *grad_bias,
	int batch, int out_c, int out_h, int out_w)
{
	int oc = blockIdx.x * blockDim.x + threadIdx.x;
	if (oc < out_c)
	{
		float sum_db = 0.0f;
		for (int b = 0; b < batch; ++b)
		{
			for (int h = 0; h < out_h; ++h)
			{
				for (int w = 0; w < out_w; ++w)
				{
					int idx = ((b * out_c + oc) * out_h + h) * out_w + w;
					sum_db += grad_output[idx];
				}
			}
		}
		grad_bias[oc] = sum_db;
	}
}

// =================================================================================
// 1. BACKWARD INPUT (dX) - Tối ưu hóa: Grid 3D + Restrict + Unroll
// =================================================================================
__global__ void conv2d_backward_input_shared_kernel(
	const float *__restrict__ grad_output,
	const float *__restrict__ weights,
	float *__restrict__ grad_input,
	int batch, int in_c, int out_c,
	int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding)
{
	// 1. Khai báo Shared Memory để chứa Grad Output
	__shared__ float s_grad_out[SHARED_WIDTH][SHARED_WIDTH];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Tọa độ của pixel Input mà thread này phụ trách tính toán
	int pw = blockIdx.x * blockDim.x + tx;
	int ph = blockIdx.y * blockDim.y + ty;
	int b_c_idx = blockIdx.z;

	int pc = b_c_idx % in_c; // Input Channel (của grad_input)
	int pn = b_c_idx / in_c; // Batch Index

	float sum = 0.0f;

	// ---------------------------------------------------------
	// LOOP QUA CÁC OUTPUT CHANNELS
	// ---------------------------------------------------------
	for (int oc = 0; oc < out_c; ++oc)
	{
		// --- LOAD GRAD_OUTPUT VÀO SHARED MEMORY ---
		// Vùng Grad Output cần thiết bao trùm vùng Input hiện tại
		// Tọa độ gốc của Tile trên Grad Output
		// (Lưu ý: Logic padding đảo ngược trong backward)
		int tile_start_w = blockIdx.x * blockDim.x - padding;
		int tile_start_h = blockIdx.y * blockDim.y - padding;

		// Flatten thread index để load cộng tác (Cooperative loading)
		int tid = ty * TILE_WIDTH + tx;
		int num_pixels = SHARED_WIDTH * SHARED_WIDTH;

		for (int i = tid; i < num_pixels; i += (TILE_WIDTH * TILE_WIDTH))
		{
			int sy = i / SHARED_WIDTH;
			int sx = i % SHARED_WIDTH;

			int cur_h = tile_start_h + sy;
			int cur_w = tile_start_w + sx;

			// Kiểm tra biên khi load từ Global Grad Output
			if (cur_h >= 0 && cur_h < out_h && cur_w >= 0 && cur_w < out_w)
			{
				int out_idx = ((pn * out_c + oc) * out_h + cur_h) * out_w + cur_w;
				s_grad_out[sy][sx] = grad_output[out_idx];
			}
			else
			{
				s_grad_out[sy][sx] = 0.0f;
			}
		}

		__syncthreads(); // Đợi load xong

		// --- TÍNH TOÁN ---
		if (pw < in_w && ph < in_h && pn < batch)
		{
// Convolution đảo ngược
// Input[h][w] bị ảnh hưởng bởi Output[h-1..h+1][w-1..w+1]
#pragma unroll
			for (int kh = 0; kh < KERNEL_SIZE; ++kh)
			{
#pragma unroll
				for (int kw = 0; kw < KERNEL_SIZE; ++kw)
				{
					// Lấy dữ liệu từ Shared Memory
					// Do ta đã load vùng bao quanh, nên truy cập trực tiếp
					// Lưu ý: Logic padding trong backward hơi đối xứng
					// Ta cần tính toán đúng index trọng số

					// Logic Backward chuẩn:
					// grad_input += grad_output * weight
					// Vị trí grad_output tương ứng với trọng số (kh, kw)
					// Ở đây giả sử padding=1, stride=1 thì ánh xạ 1:1

					// Tính lại trọng số tương ứng (Flip kernel logic)
					// Trong Backward Conv, kernel thực tế bị xoay 180 độ
					// Hoặc ta duyệt ngược index trọng số
					int w_kh = KERNEL_SIZE - 1 - kh;
					int w_kw = KERNEL_SIZE - 1 - kw;

					int w_idx = ((oc * in_c + pc) * KERNEL_SIZE + w_kh) * KERNEL_SIZE + w_kw;

					// Truy cập Shared Memory
					// Do s_grad_out đã căn chỉnh theo (ty, tx) + padding
					sum += s_grad_out[ty + kh][tx + kw] * weights[w_idx];
				}
			}
		}
		__syncthreads(); // Đợi tính xong trước khi load channel mới
	}

	// Ghi kết quả
	if (pw < in_w && ph < in_h && pn < batch)
	{
		int in_idx = ((pn * in_c + pc) * in_h + ph) * in_w + pw;
		grad_input[in_idx] = sum;
	}
}

// =================================================================================
// 2. BACKWARD WEIGHTS (dW) - Tối ưu hóa: Restrict + Unroll
// =================================================================================
__global__ void conv2d_backward_weight_kernel_opt(
	const float *__restrict__ input,
	const float *__restrict__ grad_output,
	float *__restrict__ grad_weights,
	int batch, int in_c, int out_c,
	int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding,
	int total_weights)
{
	// Dùng Grid 1D phẳng cho weights (vì số lượng weights ít)
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= total_weights)
		return;

	// Giải mã index weight: [out_c, in_c, kh, kw]
	int kw = idx % k_size;
	int tmp = idx / k_size;
	int kh = tmp % k_size;
	tmp /= k_size;
	int ic = tmp % in_c;
	int oc = tmp / in_c;

	float sum_dw = 0.0f;

	// Vòng lặp nặng nhất là Batch -> Spatial
	// Ta dùng __restrict__ để input/grad_output load qua Cache L1/Texture
	for (int b = 0; b < batch; ++b)
	{
		for (int oh = 0; oh < out_h; ++oh)
		{
#pragma unroll
			for (int ow = 0; ow < out_w; ++ow)
			{
				int ih = oh * stride - padding + kh;
				int iw = ow * stride - padding + kw;

				if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
				{
					int in_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
					int out_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;

					sum_dw += input[in_idx] * grad_output[out_idx];
				}
			}
		}
	}
	grad_weights[idx] = sum_dw;
}

// =================================================================================
// 3. BACKWARD BIAS (db) - Tối ưu hóa: Parallel Atomic Add
// =================================================================================
// Thay vì 1 thread cộng hàng triệu số, ta dùng nhiều thread cộng song song
__global__ void conv2d_backward_bias_kernel_atomic(
	const float *__restrict__ grad_output,
	float *__restrict__ grad_bias,
	int batch, int out_c, int out_h, int out_w)
{
	// Grid 3D: (out_w, out_h, batch * out_c)
	// Mỗi thread đọc 1 pixel gradient và cộng vào bias channel tương ứng
	int pw = blockIdx.x * blockDim.x + threadIdx.x;
	int ph = blockIdx.y * blockDim.y + threadIdx.y;
	int b_c_idx = blockIdx.z;

	if (pw >= out_w || ph >= out_h || b_c_idx >= batch * out_c)
		return;

	int oc = b_c_idx % out_c;
	int pn = b_c_idx / out_c;

	int idx = ((pn * out_c + oc) * out_h + ph) * out_w + pw;
	float val = grad_output[idx];

	// Dùng atomicAdd để tránh Race Condition
	// (Nhiều thread cùng cộng vào 1 địa chỉ grad_bias[oc])
	atomicAdd(&grad_bias[oc], val);
}