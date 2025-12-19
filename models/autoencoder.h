#pragma once

#include "tensor.h"
#include "layers/Conv2D.h"
#include "layers/Relu.h"
#include "layers/MaxPool2D.h"
#include "layers/Upsample2D.h"
#include <string>

class Autoencoder
{
public:
    Conv2D conv1, conv2, conv3, conv4, conv5;
    // Cần thêm biến để lưu output của các lớp Conv (vì class Conv2D mặc định chỉ lưu last_input)
    // Tensor out_conv1, out_conv2, out_conv3, out_conv4, out_conv5;
    // ReLU relu1, relu2, relu3, relu4;
    MaxPool2D pool1, pool2;
    Upsample2D upsample1, upsample2;

public:
    Autoencoder();
    ~Autoencoder()
    {
        // out_conv1.free_device();
        // out_conv2.free_device();
        // out_conv3.free_device();
        // out_conv4.free_device();
        // out_conv5.free_device();
        // free GPU memory
        conv1.weights.free_device();
        conv1.bias.free_device();
        conv1.grad_weights.free_device();
        conv1.grad_bias.free_device();
        conv1.cached_output.free_device();
        conv1.cached_grad_input.free_device();
        // conv1.last_input.free_device();

        conv2.weights.free_device();
        conv2.bias.free_device();
        conv2.grad_weights.free_device();
        conv2.grad_bias.free_device();
        conv2.cached_output.free_device();
        conv2.cached_grad_input.free_device();
        // conv2.last_input.free_device();

        conv3.weights.free_device();
        conv3.bias.free_device();
        conv3.grad_weights.free_device();
        conv3.grad_bias.free_device();
        conv3.cached_output.free_device();
        conv3.cached_grad_input.free_device();
        // conv3.last_input.free_device();

        conv4.weights.free_device();
        conv4.bias.free_device();
        conv4.grad_weights.free_device();
        conv4.grad_bias.free_device();
        conv4.cached_output.free_device();
        conv4.cached_grad_input.free_device();
        // conv4.last_input.free_device();

        conv5.weights.free_device();
        conv5.bias.free_device();
        conv5.grad_weights.free_device();
        conv5.grad_bias.free_device();
        conv5.cached_output.free_device();
        conv5.cached_grad_input.free_device();
        // conv5.last_input.free_device();

        // pool1.last_input.free_device();
        pool1.mask.free_device();

        // pool2.last_input.free_device();
        pool2.mask.free_device();
    }
    void forward(const Tensor &input, const std::string &device = "host");
    void backward(const Tensor &grad_output, float learning_rate, const std::string &device = "host");
    void saveWeights(const std::string &path);
    void loadWeights(const std::string &path);
};
