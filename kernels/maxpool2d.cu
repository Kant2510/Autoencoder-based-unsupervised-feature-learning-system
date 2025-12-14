#include "maxpool2d.h"

// --------------------------------------------------------------------------
// 3. MAXPOOL KERNEL
// --------------------------------------------------------------------------
__global__ void maxpool_forward_kernel(
	const float *input,
	float *output,
	float *mask, // Lưu index để dùng cho Backward
	int batch, int channels,
	int in_h, int in_w, int out_h, int out_w,
	int pool_size, int stride)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_elements = batch * channels * out_h * out_w;

	if (idx < total_elements)
	{
		// Giải mã tọa độ output
		int pw = idx % out_w;
		int tmp = idx / out_w;
		int ph = tmp % out_h;
		tmp /= out_h;
		int pc = tmp % channels;
		int pn = tmp / channels;

		int h_start = ph * stride;
		int w_start = pw * stride;

		float max_val = -1e30f; // Số rất nhỏ
		int max_idx = -1;

		// Duyệt cửa sổ pooling
		for (int i = 0; i < pool_size; ++i)
		{
			for (int j = 0; j < pool_size; ++j)
			{
				int cur_h = h_start + i;
				int cur_w = w_start + j;

				if (cur_h < in_h && cur_w < in_w)
				{
					int in_idx = ((pn * channels + pc) * in_h + cur_h) * in_w + cur_w;
					float val = input[in_idx];
					if (val > max_val)
					{
						max_val = val;
						max_idx = in_idx;
					}
				}
			}
		}

		output[idx] = max_val;
		// Lưu vị trí max vào mask tại vị trí input tương ứng
		// Sử dụng atomic để tránh race condition (mặc dù với stride=pool_size thì không overlap)
		if (max_idx >= 0)
		{
			atomicExch(&mask[max_idx], 1.0f);
		}
	}
}

// MaxPool Backward: Phân phối gradient về đúng vị trí max (dùng mask)
__global__ void maxpool_backward_kernel(
	const float *grad_output,
	const float *mask,
	float *grad_input,
	int n_out)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n_out)
	{
		// Mask lưu index của vị trí max trong input
		int max_idx = (int)mask[idx];
		if (max_idx != -1)
		{
			// Cộng dồn gradient (để an toàn nếu có overlap, dù maxpool thường không overlap)
			// atomicAdd là cần thiết nếu stride < pool_size. Với stride=pool_size thì gán trực tiếp cũng được.
			// Để tổng quát ta dùng atomicAdd.
			atomicAdd(&grad_input[max_idx], grad_output[idx]);
		}
	}
}