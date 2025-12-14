#include "Relu.h"

Tensor ReLU::forward(const Tensor &input, const std::string &device)
{
	this->last_input = input;

	Tensor output(input.batch, input.channels, input.height, input.width);

	if (device == "host")
	{
		for (size_t i = 0; i < output.h_data.size(); i++)
		{
			output.h_data[i] = std::max(0.0f, input.h_data[i]);
		}
	}
	else if (device == "device")
	{
		output.allocate_device();
		// GPU implementation
		int total = output.numel();
		int threads = 256;
		int blocks = (total + threads - 1) / threads;

		relu_forward_kernel<<<blocks, threads>>>(
			input.d_data,
			output.d_data,
			total);
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaGetLastError());

		// output.to_host(); // Copy result back to host
	}
	else
	{
		throw std::invalid_argument("Unknown device: " + device);
	}

	return output;
}

Tensor ReLU::backward(const Tensor &grad_output, const std::string &device)
{
	Tensor grad_input(grad_output.batch, grad_output.channels, grad_output.height, grad_output.width);

	if (device == "host")
	{
		for (size_t i = 0; i < grad_input.h_data.size(); i++)
		{
			grad_input.h_data[i] = (last_input.h_data[i] > 0.0f) ? grad_output.h_data[i] : 0.0f;
		}
	}
	else if (device == "device")
	{
		grad_input.allocate_device();
		// GPU implementation
		int total = grad_output.numel();
		int threads = 256;
		int blocks = (total + threads - 1) / threads;

		relu_backward_kernel<<<blocks, threads>>>(
			last_input.d_data,
			grad_output.d_data,
			grad_input.d_data,
			total);
		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaGetLastError());

		// grad_input.to_host(); // Copy result back to host
	}
	else
	{
		throw std::invalid_argument("Unknown device: " + device);
	}

	return grad_input;
}
