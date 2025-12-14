#pragma once
#include <cuda_runtime.h>

// --------------------------------------------------------------------------
// 2. RELU KERNEL
// --------------------------------------------------------------------------
__global__ void relu_forward_kernel(float *input, float *output, int size);

// ReLU Backward: Nếu input > 0 thì gradient = grad_output, ngược lại = 0
__global__ void relu_backward_kernel(const float *input, const float *grad_output, float *grad_input, int n);