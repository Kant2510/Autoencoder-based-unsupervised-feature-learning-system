#pragma once
#include <cuda_runtime.h>

// ======================================================================
// 4. UPDATE WEIGHTS KERNEL (SGD)
// ======================================================================
__global__ void sgd_update_kernel(float *weights, const float *grads, float lr, int n);