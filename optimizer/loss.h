#pragma once
#include "../models/tensor.h"
#include "../kernels/optimizer/loss.h"

class MSELoss
{
public:
    float forward(const Tensor &output, const Tensor &target, std::string device = "host");
    Tensor backward(const Tensor &output, const Tensor &target, std::string device = "host");
};