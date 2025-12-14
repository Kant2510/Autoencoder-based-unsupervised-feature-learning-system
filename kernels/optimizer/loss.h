#pragma once
#include <cuda_runtime.h>

// ======================================================================
// 1. MSE LOSS KERNELS
// ======================================================================

// Kernel tính tổng bình phương lỗi với shared memory reduction
__global__ void mse_loss_forward_kernel(const float *preds, const float *targets, float *partial_sums, int n);

// Kernel để tổng hợp các partial sums
__global__ void sum_reduction_kernel(float *partial_sums, float *total_loss, int num_blocks);

// Kernel tính đạo hàm Loss theo Output (Backward Loss)
// dL/dx = 2/N * (pred - target)
__global__ void mse_loss_backward_kernel(const float *preds, const float *targets, float *grad_input, int n);