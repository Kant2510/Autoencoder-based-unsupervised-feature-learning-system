#include <iostream>
#include <fstream>
#include "autoencoder.h"

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
    x = relu1.forward(x, device);
    x = pool1.forward(x, device);

    x = conv2.forward(x, device);
    x = relu2.forward(x, device);
    x = pool2.forward(x, device);

    Tensor encoded = x; // Lưu mã hóa trung gian

    x = conv3.forward(encoded, device);
    x = relu3.forward(x, device);
    x = upsample1.forward(x, device);

    x = conv4.forward(x, device);
    x = relu4.forward(x, device);
    x = upsample2.forward(x, device);

    x = conv5.forward(x, device);

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
    grad = relu4.backward(grad, device);
    grad = conv4.backward(grad, device);

    grad = upsample1.backward(grad, device);
    grad = relu3.backward(grad, device);
    grad = conv3.backward(grad, device);

    grad = pool2.backward(grad, device);
    grad = relu2.backward(grad, device);
    grad = conv2.backward(grad, device);

    grad = pool1.backward(grad, device);
    grad = relu1.backward(grad, device);
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
