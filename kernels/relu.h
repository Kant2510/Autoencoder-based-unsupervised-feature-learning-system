#pragma once
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// 2. RELU KERNEL
// --------------------------------------------------------------------------
__global__ void relu_forward_kernel(float *input, float *output, int size);

// ReLU Backward: Nếu input > 0 thì gradient = grad_output, ngược lại = 0
__global__ void relu_backward_kernel(const float *input, const float *grad_output, float *grad_input, int n);
// Kernel tính đạo hàm ReLU dựa trên Output của layer trước
// grad_output: Gradient từ lớp phía sau truyền tới (dL/dy)
// output_val: Giá trị Output của layer Conv hiện tại (y)
// grad_input: Gradient cần tính để truyền vào Conv Backward (dL/dx)
__global__ void relu_backward_kernel_2(
	const float *__restrict__ grad_output,
	const float *__restrict__ current_output,
	float *__restrict__ grad_input,
	int n);