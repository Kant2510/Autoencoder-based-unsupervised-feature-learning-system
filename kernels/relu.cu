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

// Kernel tính đạo hàm ReLU dựa trên Output của layer trước
// grad_output: Gradient từ lớp phía sau truyền tới (dL/dy)
// output_val: Giá trị Output của layer Conv hiện tại (y)
// grad_input: Gradient cần tính để truyền vào Conv Backward (dL/dx)
__global__ void relu_backward_kernel_2(
	const float *__restrict__ grad_output,
	const float *__restrict__ current_output,
	float *__restrict__ grad_input,
	int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		// Logic: Nếu Output > 0 thì Input > 0 -> Đạo hàm = 1
		// Ngược lại đạo hàm = 0
		float y = current_output[idx];
		grad_input[idx] = (y > 0.0f) ? grad_output[idx] : 0.0f;
	}
}