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