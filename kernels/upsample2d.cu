#include "upsample2d.h"

// --------------------------------------------------------------------------
// 4. UPSAMPLE (NEAREST) KERNEL
// --------------------------------------------------------------------------
__global__ void upsample_forward_kernel(
	const float *input,
	float *output,
	int batch, int channels,
	int in_h, int in_w, int out_h, int out_w,
	int scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total_elements = batch * channels * out_h * out_w;

	if (idx < total_elements)
	{
		int pw = idx % out_w;
		int tmp = idx / out_w;
		int ph = tmp % out_h;
		tmp /= out_h;
		int pc = tmp % channels;
		int pn = tmp / channels;

		// Nearest Neighbor: Map ngược về tọa độ input
		int src_h = ph / scale;
		int src_w = pw / scale;

		int in_idx = ((pn * channels + pc) * in_h + src_h) * in_w + src_w;
		output[idx] = input[in_idx];
	}
}

// Upsample Backward: Tính tổng gradient của vùng 2x2 để trả về 1 pixel input
__global__ void upsample_backward_kernel(const float *grad_output, float *grad_input,
										 int batch, int channels,
										 int in_h, int in_w, int out_h, int out_w, int scale)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; // Index của grad_input (nhỏ)
	int total_in = batch * channels * in_h * in_w;

	if (idx < total_in)
	{
		// Giải mã index input
		int pw = idx % in_w;
		int tmp = idx / in_w;
		int ph = tmp % in_h;
		tmp /= in_h;
		int pc = tmp % channels;
		int pn = tmp / channels;

		float sum_grad = 0.0f;
		// Duyệt qua vùng tương ứng trên grad_output
		for (int kh = 0; kh < scale; ++kh)
		{
			for (int kw = 0; kw < scale; ++kw)
			{
				int oh = ph * scale + kh;
				int ow = pw * scale + kw;

				if (oh < out_h && ow < out_w)
				{
					int out_idx = ((pn * channels + pc) * out_h + oh) * out_w + ow;
					sum_grad += grad_output[out_idx];
				}
			}
		}
		grad_input[idx] = sum_grad;
	}
}