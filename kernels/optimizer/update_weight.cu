#include "update_weight.h"

// ======================================================================
// 4. UPDATE WEIGHTS KERNEL (SGD)
// ======================================================================
__global__ void sgd_update_kernel(float *weights, const float *grads, float lr, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		weights[idx] -= lr * grads[idx];
	}
}