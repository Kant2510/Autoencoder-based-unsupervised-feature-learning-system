#pragma once
#include <vector>
#include <cmath>
#include <random>
#include "../tensor.h"
#include "../../kernels/conv2d.h"
#include "../../kernels/optimizer/update_weight.h"

class Conv2D
{
public:
	int in_channels, out_channels, kernel_size, stride, padding;

	// Trọng số và Bias
	Tensor weights, bias, grad_weights, grad_bias;
	Tensor last_input;

	Conv2D(int in_c, int out_c, int k_size = 3, int s = 1, int p = 1);

	// Hàm Forward
	// input: mảng phẳng chứa batch_size ảnh
	Tensor forward(const Tensor &input, const std::string &device = "host", const bool use_relu = true);
	// Hàm Backward (tính gradient)
	Tensor backward(const Tensor &grad_output, const std::string &device = "host");
	// Cập nhật trọng số
	void updateWeights(float learning_rate, const std::string &device = "host");
	// Forward loop on host
	void forward_loop_host(const Tensor &input, Tensor &output, int batch_size, int input_h, int input_w, int output_h, int output_w, const bool use_relu);
	// Forward loop on device
	void forward_loop_device(const Tensor &input, Tensor &output, int batch_size, int input_h, int input_w, int output_h, int output_w, const bool use_relu);
	// Backward loop on host
	void backward_loop_host(const Tensor &grad_output, Tensor &grad_input, int batch_size, int input_h, int input_w, int output_h, int output_w);
	// Backward loop on device
	void backward_loop_device(const Tensor &grad_output, Tensor &grad_input, int batch_size, int input_h, int input_w, int output_h, int output_w);
};