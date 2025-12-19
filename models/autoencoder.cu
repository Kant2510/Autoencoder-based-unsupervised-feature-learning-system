#include <iostream>
#include <fstream>
#include "autoencoder.h"

// Hàm này nhận vào Gradient từ lớp sau và Output của lớp Conv hiện tại
void compute_relu_gradient(const Tensor &grad_output, const Tensor &fused_conv_output)
{
    // 1. Tạo Tensor kết quả (grad_input cho Conv)
    // Tensor grad_input(grad_output.batch, grad_output.channels, grad_output.height, grad_output.width);
    // grad_input.allocate_device();

    // 2. Gọi Kernel
    int total = grad_output.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    relu_backward_kernel_2<<<blocks, threads>>>(
        grad_output.d_data,       // dL/dy
        fused_conv_output.d_data, // y (Dùng làm mask)
        // grad_input.d_data,        // dL/dx (Kết quả trả về)
        grad_output.d_data, // dL/dx (Kết quả trả về)
        total);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    // return grad_output;
}

Autoencoder::Autoencoder()
    : conv1(3, 256, 3, 1, 1),
      conv2(256, 128, 3, 1, 1),
      conv3(128, 128, 3, 1, 1),
      conv4(128, 256, 3, 1, 1),
      conv5(256, 3, 3, 1, 1)
{
    std::cout << "Autoencoder initialized!" << std::endl;
}

void Autoencoder::forward(const Tensor &input, const std::string &device)
{
    conv1.forward(input, device);
    // out_conv1 = conv1.cached_output; // Lưu output của conv1
    // x = relu1.forward(x, device);
    pool1.forward(conv1.cached_output, device);

    conv2.forward(pool1.cached_output, device);
    // out_conv2 = x; // Lưu output của conv2
    // x = relu2.forward(x, device);
    pool2.forward(conv2.cached_output, device);

    conv3.forward(pool2.cached_output, device);
    // out_conv3 = conv3.cached_output; // Lưu output của conv3
    // x = relu3.forward(x, device);
    upsample1.forward(conv3.cached_output, device);

    conv4.forward(upsample1.cached_output, device);
    // out_conv4 = conv4.cached_output; // Lưu output của conv4
    // x = relu4.forward(x, device);
    upsample2.forward(conv4.cached_output, device);

    conv5.forward(upsample2.cached_output, device, false); // Lớp cuối không dùng ReLU

    // return x;
}

void Autoencoder::backward(const Tensor &grad_output, float learning_rate, const std::string &device)
{
    // auto grad = relu3.backward(grad_output, device);
    // grad = conv3.backward(grad, device);

    // grad = upsample.backward(grad, device);

    // grad = relu2.backward(grad, device);
    // grad = conv2.backward(grad, device);

    // grad = pool.backward(grad, device);
    // grad = relu1.backward(grad, device);
    // grad = conv1.backward(grad, device);

    // conv1.updateWeights(learning_rate, device);
    // conv2.updateWeights(learning_rate, device);
    // conv3.updateWeights(learning_rate, device);

    // return grad;
    conv5.backward(grad_output, device);

    upsample2.backward(conv5.cached_grad_input, device);
    // grad = relu4.backward(grad, device);
    // Bước A: Tính gradient xuyên qua ReLU bằng cách dùng output của Conv4
    // out_conv4 ở đây đóng vai trò làm mask
    compute_relu_gradient(upsample2.cached_grad_input, conv4.cached_output);
    // Bước B: Truyền gradient đã lọc qua ReLU vào Conv4 Backward
    conv4.backward(upsample2.cached_grad_input, device);

    upsample1.backward(conv4.cached_grad_input, device);
    // grad = relu3.backward(grad, device);
    compute_relu_gradient(upsample1.cached_grad_input, conv3.cached_output);
    conv3.backward(upsample1.cached_grad_input, device);

    pool2.backward(conv3.cached_grad_input, device);
    // grad = relu2.backward(grad, device);
    compute_relu_gradient(pool2.cached_grad_input, conv2.cached_output);
    conv2.backward(pool2.cached_grad_input, device);

    pool1.backward(conv2.cached_grad_input, device);
    // grad = relu1.backward(grad, device);
    compute_relu_gradient(pool1.cached_grad_input, conv1.cached_output);
    conv1.backward(pool1.cached_grad_input, device);

    conv1.updateWeights(learning_rate, device);
    conv2.updateWeights(learning_rate, device);
    conv3.updateWeights(learning_rate, device);
    conv4.updateWeights(learning_rate, device);
    conv5.updateWeights(learning_rate, device);

    // return grad;
}

void Autoencoder::saveWeights(const std::string &path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file to save weights: " << path << std::endl;
        return;
    }

    auto saveTensor = [&](Tensor &t)
    {
        // file.write(reinterpret_cast<const char *>(t.h_data.data()),
        //            t.h_data.size() * sizeof(float));
        // Sử dụng pinned memory nếu có
        if (t.h_pinned != nullptr)
        {
            // std::cout << "t.h_pinned allocated with size: " << t.capacity_pinned << std::endl;
            t.to_host(); // Đồng bộ dữ liệu từ GPU về CPU pinned
            file.write(reinterpret_cast<const char *>(t.h_pinned),
                       t.numel() * sizeof(float));
            std::cout << "Saving tensor using pinned memory." << std::endl;
            for (int i = 0; i < std::min(10, t.numel()); ++i)
            {
                std::cout << "t.h_pinned[" << i << "] = " << t.h_pinned[i] << std::endl;
            }
        }
        else
        {
            file.write(reinterpret_cast<const char *>(t.h_data.data()),
                       t.h_data.size() * sizeof(float));
        }
    };

    // Save all conv layer weights & biases
    saveTensor(conv1.weights);
    saveTensor(conv1.bias);

    saveTensor(conv2.weights);
    saveTensor(conv2.bias);

    saveTensor(conv3.weights);
    saveTensor(conv3.bias);

    saveTensor(conv4.weights);
    saveTensor(conv4.bias);

    saveTensor(conv5.weights);
    saveTensor(conv5.bias);

    file.close();
    std::cout << "Weights saved to: " << path << std::endl;
}

void Autoencoder::loadWeights(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open weight file: " << path << std::endl;
        return;
    }

    auto loadTensor = [&](Tensor &t)
    {
        // file.read(reinterpret_cast<char *>(t.h_data.data()),
        //           t.h_data.size() * sizeof(float));
        // Sử dụng pinned memory nếu có
        if (t.h_pinned != nullptr)
        {
            // std::cout << "t.h_pinned allocated with size: " << t.capacity_pinned << std::endl;
            // Print h_pinned values for debugging
            // for (int i = 0; i < std::min(10, t.numel()); ++i)
            // {
            //     std::cout << "t.h_pinned[" << i << "] = " << t.h_pinned[i] << std::endl;
            // }
            // std::cout << "Loading tensor using pinned memory." << std::endl;
            file.read(reinterpret_cast<char *>(t.h_pinned),
                      t.numel() * sizeof(float));
            // Print h_pinned values for debugging
            for (int i = 0; i < std::min(10, t.numel()); ++i)
            {
                std::cout << "t.h_pinned[" << i << "] = " << t.h_pinned[i] << std::endl;
            }
            t.to_device(); // Đồng bộ dữ liệu lên GPU
        }
        else
        {
            std::cout << "Loading tensor using regular host memory." << std::endl;
            file.read(reinterpret_cast<char *>(t.h_data.data()),
                      t.h_data.size() * sizeof(float));
        }
    };

    // Load all conv layer weights & biases
    loadTensor(conv1.weights);
    loadTensor(conv1.bias);

    loadTensor(conv2.weights);
    loadTensor(conv2.bias);

    loadTensor(conv3.weights);
    loadTensor(conv3.bias);

    loadTensor(conv4.weights);
    loadTensor(conv4.bias);

    loadTensor(conv5.weights);
    loadTensor(conv5.bias);

    file.close();
    std::cout << "Weights loaded from: " << path << std::endl;
}
