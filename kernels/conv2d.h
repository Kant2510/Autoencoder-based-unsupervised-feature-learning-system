#pragma once
#include <cuda_runtime.h>

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
	int k_size, int stride, int padding);
__global__ void conv2d_optimized_kernel(
	const float *__restrict__ input,
	float *__restrict__ output,
	const float *__restrict__ weights,
	int batch, int in_channels, int out_channels,
	int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding,
	bool use_relu = true);
// ======================================================================
// 3. CONVOLUTION BACKWARD KERNELS
// ======================================================================

// 3a. Tính Gradient theo Input (dX) - Truyền lỗi về lớp trước
// Thực chất là tích chập (Full Convolution) giữa grad_output và weights
__global__ void conv2d_backward_input_kernel(
	const float *grad_output, const float *weights, float *grad_input,
	int batch, int in_c, int out_c, int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding);

// 3b. Tính Gradient theo Weights (dW)
// Mỗi thread tính 1 phần tử của Weight Gradient
__global__ void conv2d_backward_weight_kernel(
	const float *input, const float *grad_output, float *grad_weights,
	int batch, int in_c, int out_c, int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding);

// 3c. Tính Gradient theo Bias (db)
// Mỗi thread tính 1 phần tử bias (1 output channel)
__global__ void conv2d_backward_bias_kernel(
	const float *grad_output, float *grad_bias,
	int batch, int out_c, int out_h, int out_w);
// ======================================================================
// 4. OPTIMIZED KERNELS WITH SHARED MEMORY
// ======================================================================
__global__ void conv2d_backward_input_shared_kernel(
	const float *__restrict__ grad_output,
	const float *__restrict__ weights,
	float *__restrict__ grad_input,
	int batch, int in_c, int out_c,
	int in_h, int in_w, int out_h, int out_w,
	int k_size, int stride, int padding);
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
	int total_weights);
// =================================================================================
// 3. BACKWARD BIAS (db) - Tối ưu hóa: Parallel Atomic Add
// =================================================================================
// Thay vì 1 thread cộng hàng triệu số, ta dùng nhiều thread cộng song song
__global__ void conv2d_backward_bias_kernel_atomic(
	const float *__restrict__ grad_output,
	float *__restrict__ grad_bias,
	int batch, int out_c, int out_h, int out_w);