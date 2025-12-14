#pragma once

#include "tensor.h"
#include "layers/Conv2D.h"
#include "layers/Relu.h"
#include "layers/MaxPool2D.h"
#include "layers/Upsample2D.h"
#include <string>

class Autoencoder
{
private:
    Conv2D conv1, conv2, conv3, conv4, conv5;
    ReLU relu1, relu2, relu3, relu4;
    MaxPool2D pool1, pool2;
    Upsample2D upsample1, upsample2;

public:
    Autoencoder();
    Tensor forward(const Tensor &input, const std::string &device = "host");
    Tensor backward(const Tensor &grad_output, float learning_rate, const std::string &device = "host");
    void saveWeights(const std::string &path);
    void loadWeights(const std::string &path);
};
