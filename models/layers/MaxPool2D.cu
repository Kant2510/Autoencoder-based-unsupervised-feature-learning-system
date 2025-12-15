#include <vector>
#include "MaxPool2D.h"

Tensor MaxPool2D::forward(const Tensor &input, const std::string &device)
{
	this->last_input = input;

	int batch_size = input.batch;

	int input_h = input.height;
	int input_w = input.width;

	int channels = input.channels;

	int output_h = (input_h - pool_size) / stride + 1;
	int output_w = (input_w - pool_size) / stride + 1;

	// Tensor output(batch_size, channels, output_h, output_w);
	this->cached_output.reshape_if_needed(batch_size, channels, output_h, output_w);

	// this->mask = Tensor(batch_size, channels, input_h, input_w);
	this->mask.reshape_if_needed(batch_size, channels, input_h, input_w);

	if (device == "host")
	{
		this->mask.zeros("host");
		forward_loop_host(input, this->cached_output, channels, batch_size, input_h, input_w, output_h, output_w);
	}
	else if (device == "device")
	{
		// this->cached_output.allocate_device();
		// this->mask.allocate_device();
		// Initialize mask to zero directly on GPU
		this->mask.zeros("device");
		forward_loop_device(input, this->cached_output, channels, batch_size, input_h, input_w, output_h, output_w);
	}
	else
	{
		throw std::invalid_argument("Device must be 'host' or 'device'");
	}

	return this->cached_output;
}

void MaxPool2D::forward_loop_host(const Tensor &input, Tensor &output, int channels, int batch_size, int input_h, int input_w, int output_h, int output_w)
{
	// Chuyển tiếp sử dụng CPU

	for (int b = 0; b < batch_size; ++b)
	{
		for (int c = 0; c < channels; ++c)
		{
			for (int oh = 0; oh < output_h; ++oh)
			{
				for (int ow = 0; ow < output_w; ++ow)
				{

					// Tìm max trong vùng 2x2
					int h_start = oh * stride;
					int w_start = ow * stride;

					float max_val = -1e9; // Âm vô cùng
					int max_h = 0, max_w = 0;

					for (int i = 0; i < pool_size; ++i)
					{
						for (int j = 0; j < pool_size; ++j)
						{
							int cur_h = h_start + i;
							int cur_w = w_start + j;

							if (cur_h < input_h && cur_w < input_w)
							{
								float val = input.at(b, c, cur_h, cur_w);
								if (val > max_val)
								{
									max_val = val;
									max_h = cur_h;
									max_w = cur_w;
								}
							}
						}
					}

					output.at(b, c, oh, ow) = max_val;
					mask.at(b, c, max_h, max_w) = 1.0f;
				}
			}
		}
	}
}

void MaxPool2D::forward_loop_device(const Tensor &input, Tensor &output, int channels, int batch_size, int input_h, int input_w, int output_h, int output_w)
{
	// Chuyển tiếp sử dụng GPU

	int total = output.numel();
	dim3 blockSize(256);
	dim3 gridSize((total + blockSize.x - 1) / blockSize.x);

	maxpool_forward_kernel<<<gridSize, blockSize>>>(
		input.d_data,
		output.d_data,
		mask.d_data,
		batch_size,
		channels,
		input_h, input_w,
		output_h, output_w,
		pool_size, stride);

	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());

	// output.to_host();
}

Tensor MaxPool2D::backward(const Tensor &grad_output, const std::string &device)
{
	// Tensor grad_input(last_input.batch, last_input.channels,
	// 				  last_input.height, last_input.width);
	this->cached_grad_input.reshape_if_needed(
		last_input.batch, last_input.channels,
		last_input.height, last_input.width);

	int out_h = grad_output.height;
	int out_w = grad_output.width;

	if (device == "host")
	{
		// Initialize on CPU for host computation
		this->cached_grad_input.zeros();
		backward_loop_host(grad_output, this->cached_grad_input, out_h, out_w);
	}
	else if (device == "device")
	{
		// grad_input.allocate_device();
		// Initialize to zero directly on GPU
		this->cached_grad_input.zeros("device");
		backward_loop_device(grad_output, this->cached_grad_input,
							 last_input.channels, last_input.batch,
							 last_input.height, last_input.width,
							 out_h, out_w);
	}
	else
	{
		throw std::invalid_argument("Device must be 'host' or 'device'");
	}

	return this->cached_grad_input;
}

void MaxPool2D::backward_loop_host(const Tensor &grad_output, Tensor &grad_input, int output_h, int output_w)
{
	// Backward sử dụng CPU
	for (int b = 0; b < grad_output.batch; b++)
	{
		for (int c = 0; c < grad_output.channels; c++)
		{
			for (int oh = 0; oh < output_h; oh++)
			{
				for (int ow = 0; ow < output_w; ow++)
				{
					float grad = grad_output.at(b, c, oh, ow);

					for (int ph = 0; ph < pool_size; ph++)
					{
						for (int pw = 0; pw < pool_size; pw++)
						{
							int ih = oh * stride + ph;
							int iw = ow * stride + pw;
							grad_input.at(b, c, ih, iw) +=
								grad * mask.at(b, c, ih, iw);
						}
					}
				}
			}
		}
	}
}

void MaxPool2D::backward_loop_device(const Tensor &grad_output, Tensor &grad_input, int channels, int batch_size, int input_h, int input_w, int output_h, int output_w)
{
	// Backward sử dụng GPU
	int n_out = grad_output.numel();

	dim3 blockSize(256);
	dim3 gridSize((n_out + blockSize.x - 1) / blockSize.x);

	maxpool_backward_kernel<<<gridSize, blockSize>>>(
		grad_output.d_data, mask.d_data, grad_input.d_data, n_out);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());

	// grad_input.to_host();
}