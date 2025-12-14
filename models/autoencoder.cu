#include <iostream>
#include <fstream>
#include "autoencoder.h"

// Hàm này nhận vào Gradient từ lớp sau và Output của lớp Conv hiện tại
Tensor compute_relu_gradient(const Tensor &grad_output, const Tensor &fused_conv_output)
{
    // 1. Tạo Tensor kết quả (grad_input cho Conv)
    Tensor grad_input(grad_output.batch, grad_output.channels, grad_output.height, grad_output.width);
    grad_input.allocate_device();

    // 2. Gọi Kernel
    int total = grad_output.numel();
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    relu_backward_kernel_2<<<blocks, threads>>>(
        grad_output.d_data,       // dL/dy
        fused_conv_output.d_data, // y (Dùng làm mask)
        grad_input.d_data,        // dL/dx (Kết quả trả về)
        total);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaGetLastError());

    return grad_input;
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

Tensor Autoencoder::forward(const Tensor &input, const std::string &device)
{
    auto x = conv1.forward(input, device);
    out_conv1 = x; // Lưu output của conv1
    // x = relu1.forward(x, device);
    x = pool1.forward(x, device);

    x = conv2.forward(x, device);
    out_conv2 = x; // Lưu output của conv2
    // x = relu2.forward(x, device);
    x = pool2.forward(x, device);

    Tensor encoded = x; // Lưu mã hóa trung gian

    x = conv3.forward(encoded, device);
    out_conv3 = x; // Lưu output của conv3
    // x = relu3.forward(x, device);
    x = upsample1.forward(x, device);

    x = conv4.forward(x, device);
    out_conv4 = x; // Lưu output của conv4
    // x = relu4.forward(x, device);
    x = upsample2.forward(x, device);

    x = conv5.forward(x, device, false); // Lớp cuối không dùng ReLU

    return x;
}

Tensor Autoencoder::backward(const Tensor &grad_output, float learning_rate, const std::string &device)
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
    auto grad = conv5.backward(grad_output, device);

    grad = upsample2.backward(grad, device);
    // grad = relu4.backward(grad, device);
    // Bước A: Tính gradient xuyên qua ReLU bằng cách dùng output của Conv4
    // out_conv4 ở đây đóng vai trò làm mask
    grad = compute_relu_gradient(grad, out_conv4);
    // Bước B: Truyền gradient đã lọc qua ReLU vào Conv4 Backward
    grad = conv4.backward(grad, device);

    grad = upsample1.backward(grad, device);
    // grad = relu3.backward(grad, device);
    grad = compute_relu_gradient(grad, out_conv3);
    grad = conv3.backward(grad, device);

    grad = pool2.backward(grad, device);
    // grad = relu2.backward(grad, device);
    grad = compute_relu_gradient(grad, out_conv2);
    grad = conv2.backward(grad, device);

    grad = pool1.backward(grad, device);
    // grad = relu1.backward(grad, device);
    grad = compute_relu_gradient(grad, out_conv1);
    grad = conv1.backward(grad, device);

    conv1.updateWeights(learning_rate, device);
    conv2.updateWeights(learning_rate, device);
    conv3.updateWeights(learning_rate, device);
    conv4.updateWeights(learning_rate, device);
    conv5.updateWeights(learning_rate, device);

    return grad;
}

void Autoencoder::saveWeights(const std::string &path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file to save weights: " << path << std::endl;
        return;
    }

    auto saveTensor = [&](const Tensor &t)
    {
        file.write(reinterpret_cast<const char *>(t.h_data.data()),
                   t.h_data.size() * sizeof(float));
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
        file.read(reinterpret_cast<char *>(t.h_data.data()),
                  t.h_data.size() * sizeof(float));
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
