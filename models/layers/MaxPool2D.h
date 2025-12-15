#pragma once
#include <vector>
#include "../tensor.h"
#include "../../kernels/maxpool2d.h"

class MaxPool2D
{
public:
	int pool_size, stride;
	Tensor last_input, mask;  // Dùng cho Backward Pass sau này
	Tensor cached_output;	  // Lưu output của lần forward cuối cùng (dùng cho backward)
	Tensor cached_grad_input; // Lưu grad_input của lần backward cuối cùng (dùng cho backward)

	MaxPool2D(int p_size = 2, int s = 2) : pool_size(p_size), stride(s) {}
	Tensor forward(const Tensor &input, const std::string &device = "host");
	Tensor backward(const Tensor &grad_output, const std::string &device = "host");
	void forward_loop_host(const Tensor &input, Tensor &output, int channels, int batch_size, int input_h, int input_w, int output_h, int output_w);
	void forward_loop_device(const Tensor &input, Tensor &output, int channels, int batch_size, int input_h, int input_w, int output_h, int output_w);
	void backward_loop_host(const Tensor &grad_output, Tensor &grad_input, int output_h, int output_w);
	void backward_loop_device(const Tensor &grad_output, Tensor &grad_input, int channels, int batch_size, int input_h, int input_w, int output_h, int output_w);
};