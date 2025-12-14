#include "loss.h"

// ======================================================================
// 1. MSE LOSS KERNELS
// ======================================================================

// Kernel tính tổng bình phương lỗi với shared memory reduction
__global__ void mse_loss_forward_kernel(const float *preds, const float *targets, float *partial_sums, int n)
{
	extern __shared__ float shared_data[];

	int tid = threadIdx.x;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Load data and compute squared difference
	float local_sum = 0.0f;
	if (idx < n)
	{
		float diff = preds[idx] - targets[idx];
		local_sum = diff * diff;
	}
	shared_data[tid] = local_sum;
	__syncthreads();

	// Parallel reduction in shared memory
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			shared_data[tid] += shared_data[tid + stride];
		}
		__syncthreads();
	}

	// Write block result to global memory
	if (tid == 0)
	{
		partial_sums[blockIdx.x] = shared_data[0];
	}
}

// Kernel để tổng hợp các partial sums
__global__ void sum_reduction_kernel(float *partial_sums, float *total_loss, int num_blocks)
{
	extern __shared__ float shared_data[];

	int tid = threadIdx.x;
	int idx = tid;

	// Load partial sums
	float local_sum = 0.0f;
	while (idx < num_blocks)
	{
		local_sum += partial_sums[idx];
		idx += blockDim.x;
	}
	shared_data[tid] = local_sum;
	__syncthreads();

	// Parallel reduction
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
	{
		if (tid < stride)
		{
			shared_data[tid] += shared_data[tid + stride];
		}
		__syncthreads();
	}

	// Write final result
	if (tid == 0)
	{
		*total_loss = shared_data[0];
	}
}

// Kernel tính đạo hàm Loss theo Output (Backward Loss)
// dL/dx = 2/N * (pred - target)
__global__ void mse_loss_backward_kernel(const float *preds, const float *targets, float *grad_input, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		grad_input[idx] = 2.0f * (preds[idx] - targets[idx]) / (float)n;
	}
}