#pragma once
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// 4. UPSAMPLE (NEAREST) KERNEL
// --------------------------------------------------------------------------
__global__ void upsample_forward_kernel(
	const float *input,
	float *output,
	int batch, int channels,
	int in_h, int in_w, int out_h, int out_w,
	int scale);

// Upsample Backward: Tính tổng gradient của vùng 2x2 để trả về 1 pixel input
__global__ void upsample_backward_kernel(const float *grad_output, float *grad_input,
										 int batch, int channels,
										 int in_h, int in_w, int out_h, int out_w, int scale);