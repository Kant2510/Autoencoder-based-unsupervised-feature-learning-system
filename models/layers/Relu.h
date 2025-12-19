#pragma once
#include "../tensor.h"
#include "../../kernels/relu.h"

class ReLU
{
private:
	const Tensor *last_input;
	Tensor cached_output;	  // Lưu output của lần forward cuối cùng (dùng cho backward)
	Tensor cached_grad_input; // Lưu grad_input của lần backward cuối cùng (dùng

public:
	void forward(const Tensor &input, const std::string &device = "host");
	void backward(const Tensor &grad_output, const std::string &device = "host");
};