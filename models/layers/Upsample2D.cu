#include "Upsample2D.h"

Tensor Upsample2D::forward(const Tensor &input, const std::string &device)
{
	int channels = input.channels;
	int batch_size = input.batch;

	int out_h = input.height * scale_factor;
	int out_w = input.width * scale_factor;

	Tensor output(batch_size, channels, out_h, out_w);

	if (device == "host")
	{
		forward_loop_host(input, output, channels, batch_size, out_h, out_w);
	}
	else if (device == "device")
	{
		output.allocate_device();
		forward_loop_device(input, output, channels, batch_size,
							input.height, input.width,
							out_h, out_w);
	}
	else
	{
		throw std::invalid_argument("Invalid device specified. Use 'host' or 'device'.");
	}

	return output;
}

void Upsample2D::forward_loop_host(const Tensor &input, Tensor &output,
								   int channels, int batch_size,
								   int out_h, int out_w)
{
	// Implementation for host (CPU) forward pass if needed
	for (int b = 0; b < batch_size; b++)
	{
		for (int c = 0; c < channels; c++)
		{
			for (int oh = 0; oh < out_h; oh++)
			{
				for (int ow = 0; ow < out_w; ow++)
				{
					int ih = oh / scale_factor;
					int iw = ow / scale_factor;
					output.at(b, c, oh, ow) = input.at(b, c, ih, iw);
				}
			}
		}
	}
}

void Upsample2D::forward_loop_device(const Tensor &input, Tensor &output,
									 int channels, int batch_size,
									 int input_h, int input_w,
									 int output_h, int output_w)
{
	// Implementation for device (GPU) forward pass if needed
	// This would typically involve launching a CUDA kernel
	int totalElements = output.numel();
	int threadsPerBlock = 256;
	int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
	// Launch your CUDA kernel here
	upsample_forward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
		input.d_data, output.d_data,
		batch_size, channels,
		input_h, input_w,
		output_h, output_w,
		scale_factor);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());

	// output.to_host(); // Copy result back to host if needed
}

Tensor Upsample2D::backward(const Tensor &grad_output, const std::string &device)
{
	int in_h = grad_output.height / scale_factor;
	int in_w = grad_output.width / scale_factor;
	int out_h = grad_output.height;
	int out_w = grad_output.width;

	int channels = grad_output.channels;
	int batch_size = grad_output.batch;

	Tensor grad_input(batch_size, channels, in_h, in_w);

	if (device == "host")
	{
		grad_input.zeros();
		backward_loop_host(grad_output, grad_input, channels, batch_size, in_h, in_w);
	}
	else if (device == "device")
	{
		grad_input.allocate_device();
		grad_input.zeros("device");
		backward_loop_device(grad_output, grad_input, channels, batch_size,
							 in_h, in_w,
							 out_h, out_w);
	}
	else
	{
		throw std::invalid_argument("Invalid device specified. Use 'host' or 'device'.");
	}

	return grad_input;
}

void Upsample2D::backward_loop_host(const Tensor &grad_output, Tensor &grad_input,
									int channels, int batch_size,
									int out_h, int out_w)
{
	// Implementation for host (CPU) backward pass if needed
	for (int b = 0; b < batch_size; b++)
	{
		for (int c = 0; c < channels; c++)
		{
			for (int oh = 0; oh < out_h; oh++)
			{
				for (int ow = 0; ow < out_w; ow++)
				{
					int ih = oh / scale_factor;
					int iw = ow / scale_factor;
					grad_input.at(b, c, ih, iw) +=
						grad_output.at(b, c, oh, ow);
				}
			}
		}
	}
}

void Upsample2D::backward_loop_device(const Tensor &grad_output, Tensor &grad_input,
									  int channels, int batch_size,
									  int input_h, int input_w,
									  int output_h, int output_w)
{
	// Implementation for device (GPU) backward pass if needed
	// This would typically involve launching a CUDA kernel
	int totalElements = grad_output.numel();
	int threadsPerBlock = 256;
	int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;
	// Launch your CUDA kernel here
	upsample_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
		grad_output.d_data, grad_input.d_data,
		batch_size, channels,
		input_h, input_w,
		output_h, output_w,
		scale_factor);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());

	// grad_input.to_host(); // Copy result back to host if needed
}