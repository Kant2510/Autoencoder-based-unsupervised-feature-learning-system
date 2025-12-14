#pragma once
#include "../tensor.h"
#include "../../kernels/relu.h"

class ReLU
{
private:
	Tensor last_input;

public:
	Tensor forward(const Tensor &input, const std::string &device = "host");
	Tensor backward(const Tensor &grad_output, const std::string &device = "host");
};