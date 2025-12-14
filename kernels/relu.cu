#include "relu.h"

// --------------------------------------------------------------------------
// 2. RELU KERNEL
// --------------------------------------------------------------------------
__global__ void relu_forward_kernel(float *input, float *output, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		output[idx] = fmaxf(0.0f, input[idx]);
	}
}

// ReLU Backward: Nếu input > 0 thì gradient = grad_output, ngược lại = 0
__global__ void relu_backward_kernel(const float *input, const float *grad_output, float *grad_input, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
	}
}