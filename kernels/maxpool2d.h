#pragma once
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// 3. MAXPOOL KERNEL
// --------------------------------------------------------------------------
__global__ void maxpool_forward_kernel(
	const float *input,
	float *output,
	float *mask, // Lưu index để dùng cho Backward
	int batch, int channels,
	int in_h, int in_w, int out_h, int out_w,
	int pool_size, int stride);

// MaxPool Backward: Phân phối gradient về đúng vị trí max (dùng mask)
__global__ void maxpool_backward_kernel(
	const float *grad_output,
	const float *mask,
	float *grad_input,
	int n_out);
