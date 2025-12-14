#include <algorithm>
#include "loss.h"

float MSELoss::forward(const Tensor &output, const Tensor &target, std::string device)
{
    float sum = 0.0f;
    int n = output.numel();

    if (device == "device")
    {
        // GPU implementation với parallel reduction
        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;

        // Allocate device memory for partial sums and final result
        float *d_partial_sums = nullptr;
        float *d_total_loss = nullptr;

        CHECK_CUDA(cudaMalloc(&d_partial_sums, numBlocks * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_total_loss, sizeof(float)));
        CHECK_CUDA(cudaMemset(d_total_loss, 0, sizeof(float)));

        // Launch kernel with shared memory for reduction
        int sharedMemSize = blockSize * sizeof(float);
        mse_loss_forward_kernel<<<numBlocks, blockSize, sharedMemSize>>>(
            output.d_data, target.d_data, d_partial_sums, n);

        // Sum up partial results
        sum_reduction_kernel<<<1, blockSize, blockSize * sizeof(float)>>>(
            d_partial_sums, d_total_loss, numBlocks);

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(&sum, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost));

        // Free temporary memory
        CHECK_CUDA(cudaFree(d_partial_sums));
        CHECK_CUDA(cudaFree(d_total_loss));

        return sum / n;
    }

    // CPU implementation
    for (size_t i = 0; i < output.h_data.size(); i++)
    {
        float diff = output.h_data[i] - target.h_data[i];
        sum += diff * diff;
    }
    return sum / output.h_data.size();
}

Tensor MSELoss::backward(const Tensor &output, const Tensor &target, std::string device)
{
    Tensor grad(output.batch, output.channels, output.height, output.width);
    int n = output.numel();

    if (device == "device")
    {
        // GPU implementation
        grad.allocate_device();

        const int blockSize = 256;
        const int numBlocks = (n + blockSize - 1) / blockSize;

        mse_loss_backward_kernel<<<numBlocks, blockSize>>>(
            output.d_data, target.d_data, grad.d_data, n);

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaGetLastError());

        // grad.to_host();
        return grad;
    }

    // CPU implementation
    float scale = 2.0f / output.h_data.size();
    for (size_t i = 0; i < output.h_data.size(); i++)
    {
        grad.h_data[i] = scale * (output.h_data[i] - target.h_data[i]);
    }

    return grad;
}