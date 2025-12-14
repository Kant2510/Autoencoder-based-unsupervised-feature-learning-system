#include <vector>
#include <cmath>
#include <random>
#include "Conv2D.h"

Conv2D::Conv2D(int in_c, int out_c, int k_size, int s, int p)
	: in_channels(in_c), out_channels(out_c), kernel_size(k_size), stride(s), padding(p)
{

	// Khởi tạo trọng số (He Initialization hoặc Xavier)
	// Đây là bước quan trọng để mạng hội tụ
	int fan_in = in_channels * kernel_size * kernel_size;
	float std_dev = sqrt(2.0f / fan_in); // He Init cho ReLU

	std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, std_dev);

	weights = Tensor(out_channels, in_channels, kernel_size, kernel_size);
	bias = Tensor(1, out_channels, 1, 1);
	grad_weights = Tensor(out_channels, in_channels, kernel_size, kernel_size);
	grad_bias = Tensor(1, out_channels, 1, 1);

	for (auto &w : weights.h_data)
		w = distribution(generator);
	bias.zeros();
	grad_weights.zeros();
	grad_bias.zeros();

	weights.allocate_device();
	bias.allocate_device();
	grad_weights.allocate_device();
	grad_bias.allocate_device();

	// Copy initialized weights and bias to device
	weights.to_device();
	bias.to_device();
	// Initialize gradients to zero directly on GPU (no need to copy CPU zeros)
	grad_weights.zeros("device");
	grad_bias.zeros("device");
}

Tensor Conv2D::forward(const Tensor &input, const std::string &device, const bool use_relu)
{
	this->last_input = input;

	int batch_size = input.batch;

	int input_h = input.height;
	int input_w = input.width;

	// Tính kích thước output: (H - K + 2P) / S + 1
	int output_h = (input_h - kernel_size + 2 * padding) / stride + 1;
	int output_w = (input_w - kernel_size + 2 * padding) / stride + 1;

	// Cấp phát bộ nhớ cho output
	Tensor output(batch_size, out_channels, output_h, output_w);

	if (device == "host")
	{
		forward_loop_host(input, output, batch_size, input_h, input_w, output_h, output_w, use_relu);
	}
	else if (device == "device")
	{
		output.allocate_device();
		forward_loop_device(input, output, batch_size, input_h, input_w, output_h, output_w, use_relu);
	}
	else
	{
		throw std::invalid_argument("Invalid device specified for Conv2D forward");
	}

	return output;
}

void Conv2D::forward_loop_host(const Tensor &input, Tensor &output, int batch_size, int input_h, int input_w, int output_h, int output_w, const bool use_relu)
{
	// VÒNG LẶP CHÍNH (7 Loop lồng nhau - Rất chậm trên CPU nhưng dễ hiểu)
	for (int b = 0; b < batch_size; ++b)
	{ // 1. Từng ảnh trong batch
		for (int oc = 0; oc < out_channels; ++oc)
		{ // 2. Từng kênh output (Feature map)
			for (int oh = 0; oh < output_h; ++oh)
			{ // 3. Chiều cao output
				for (int ow = 0; ow < output_w; ++ow)
				{ // 4. Chiều rộng output

					float sum = 0.0f;

					// Tích chập: duyệt qua các kênh input và kích thước kernel
					for (int ic = 0; ic < in_channels; ++ic)
					{ // 5. Kênh input
						for (int kh = 0; kh < kernel_size; ++kh)
						{ // 6. Kernel row
							for (int kw = 0; kw < kernel_size; ++kw)
							{ // 7. Kernel col

								// Tính vị trí tương ứng trên input (có tính padding)
								int ih = oh * stride - padding + kh;
								int iw = ow * stride - padding + kw;

								// Kiểm tra biên (Implicit Padding)
								// Nếu nằm trong vùng ảnh thật thì nhân, ngược lại (padding) coi như 0
								if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w)
								{
									sum += input.at(b, ic, ih, iw) * weights.at(oc, ic, kh, kw);
								}
							}
						}
					}
					if (use_relu)
					{
						// Áp dụng ReLU
						sum = std::max(0.0f, sum + bias.at(0, oc, 0, 0));
					}
					else
					{
						sum += bias.at(0, oc, 0, 0);
					}
					// Gán vào output (cộng bias)
					output.at(b, oc, oh, ow) = sum;
				}
			}
		}
	}
}

void Conv2D::forward_loop_device(const Tensor &input, Tensor &output, int batch_size, int input_h, int input_w, int output_h, int output_w, const bool use_relu)
{
	if (input.d_data == nullptr || weights.d_data == nullptr || bias.d_data == nullptr || output.d_data == nullptr)
	{
		std::cerr << "Error: One of the tensors is not allocated on device." << std::endl;
		throw std::runtime_error("Input, weights, bias, or output tensor not allocated on device");
	}

	// Cấu hình kernel CUDA
	dim3 blockSize(16, 16);
	dim3 gridSize(
		(output_w + blockSize.x - 1) / blockSize.x, // Trục X: Bao phủ chiều rộng Output
		(output_h + blockSize.y - 1) / blockSize.y, // Trục Y: Bao phủ chiều cao Output
		batch_size * out_channels					// Trục Z: Bao phủ (Batch * Output Channels)
	);
	// int total = output.numel();

	// dim3 blockSize(256);
	// dim3 gridSize((total + blockSize.x - 1) / blockSize.x);

	// Gọi kernel CUDA
	conv2d_optimized_kernel<<<gridSize, blockSize>>>(
		input.d_data,
		output.d_data,
		weights.d_data,
		batch_size,
		in_channels, out_channels,
		input_h, input_w,
		output_h, output_w,
		kernel_size, stride, padding,
		use_relu);

	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());

	// Chuyển kết quả về CPU (nếu cần)
	// output.to_host();
}

Tensor Conv2D::backward(const Tensor &grad_output, const std::string &device)
{
	Tensor grad_input(last_input.batch, in_channels,
					  last_input.height, last_input.width);

	int batch_size = grad_output.batch;

	int input_h = last_input.height;
	int input_w = last_input.width;

	int output_h = grad_output.height;
	int output_w = grad_output.width;

	if (device == "host")
	{
		grad_input.zeros();
		backward_loop_host(grad_output, grad_input, batch_size, input_h, input_w, output_h, output_w);
	}
	else if (device == "device")
	{
		grad_input.allocate_device();
		grad_input.zeros("device");
		backward_loop_device(grad_output, grad_input, batch_size, input_h, input_w, output_h, output_w);
	}
	else
	{
		throw std::invalid_argument("Invalid device specified for Conv2D backward");
	}

	return grad_input;
}

void Conv2D::backward_loop_host(const Tensor &grad_output, Tensor &grad_input, int batch_size, int input_h, int input_w, int output_h, int output_w)
{
	for (int b = 0; b < batch_size; b++)
	{
		for (int oc = 0; oc < out_channels; oc++)
		{
			for (int oh = 0; oh < output_h; oh++)
			{
				for (int ow = 0; ow < output_w; ow++)
				{
					float grad = grad_output.at(b, oc, oh, ow);
					grad_bias.at(0, oc, 0, 0) += grad;

					for (int ic = 0; ic < in_channels; ic++)
					{
						for (int kh = 0; kh < kernel_size; kh++)
						{
							for (int kw = 0; kw < kernel_size; kw++)
							{
								int ih = oh * stride - padding + kh;
								int iw = ow * stride - padding + kw;

								if (ih >= 0 && ih < last_input.height &&
									iw >= 0 && iw < last_input.width)
								{
									grad_weights.at(oc, ic, kh, kw) +=
										grad * last_input.at(b, ic, ih, iw);
									grad_input.at(b, ic, ih, iw) +=
										grad * weights.at(oc, ic, kh, kw);
								}
							}
						}
					}
				}
			}
		}
	}
}

void Conv2D::backward_loop_device(const Tensor &grad_output, Tensor &grad_input, int batch_size, int input_h, int input_w, int output_h, int output_w)
{
	// -----------------------------------------------------------------------
	// 1. Tính dX (Gradient Input) - Dùng Grid 3D
	// -----------------------------------------------------------------------
	// Cần reset grad_input về 0 nếu kernel dùng cộng dồn,
	// nhưng kernel dX của ta GHI ĐÈ (assign) nên không cần cudaMemset.

	dim3 blockSize(16, 16);
	dim3 gridSizeInput(
		(input_w + blockSize.x - 1) / blockSize.x,
		(input_h + blockSize.y - 1) / blockSize.y,
		batch_size * in_channels // Trục Z bao phủ Batch * Input Channels
	);

	conv2d_backward_input_shared_kernel<<<gridSizeInput, blockSize>>>(
		grad_output.d_data,
		weights.d_data,
		grad_input.d_data,
		batch_size, in_channels, out_channels,
		input_h, input_w, output_h, output_w,
		kernel_size, stride, padding);
	CHECK_CUDA(cudaGetLastError());

	// -----------------------------------------------------------------------
	// 2. Tính dW (Gradient Weights)
	// -----------------------------------------------------------------------
	int total_weights = out_channels * in_channels * kernel_size * kernel_size;
	int threadsW = 256;
	int blocksW = (total_weights + threadsW - 1) / threadsW;

	conv2d_backward_weight_kernel_opt<<<blocksW, threadsW>>>(
		last_input.d_data,
		grad_output.d_data,
		grad_weights.d_data,
		batch_size, in_channels, out_channels,
		input_h, input_w, output_h, output_w,
		kernel_size, stride, padding,
		total_weights);
	CHECK_CUDA(cudaGetLastError());

	// -----------------------------------------------------------------------
	// 3. Tính db (Gradient Bias) - Dùng Atomic
	// -----------------------------------------------------------------------
	// Reset grad_bias về 0 trước vì ta dùng atomicAdd
	CHECK_CUDA(cudaMemset(grad_bias.d_data, 0, grad_bias.numel() * sizeof(float)));

	// Grid 3D phủ toàn bộ output map
	dim3 gridSizeBias(
		(output_w + blockSize.x - 1) / blockSize.x,
		(output_h + blockSize.y - 1) / blockSize.y,
		batch_size * out_channels);

	conv2d_backward_bias_kernel_atomic<<<gridSizeBias, blockSize>>>(
		grad_output.d_data,
		grad_bias.d_data,
		batch_size, out_channels, output_h, output_w);

	// Đồng bộ hóa cuối cùng
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaGetLastError());

	// Chuyển kết quả về CPU (nếu cần)
	// grad_input.to_host();
	// grad_weights.to_host();
	// grad_bias.to_host();
}

void Conv2D::updateWeights(float learning_rate, const std::string &device)
{
	if (device == "device")
	{
		// Cấu hình kernel CUDA
		int total_weights = weights.numel();
		int total_bias = bias.numel();

		dim3 blockSize(256);
		dim3 gridSizeWeights((total_weights + blockSize.x - 1) / blockSize.x);
		dim3 gridSizeBias((total_bias + blockSize.x - 1) / blockSize.x);

		// Gọi kernel cập nhật trọng số trên GPU
		sgd_update_kernel<<<gridSizeWeights, blockSize>>>(
			weights.d_data, grad_weights.d_data, learning_rate, total_weights);

		sgd_update_kernel<<<gridSizeBias, blockSize>>>(
			bias.d_data, grad_bias.d_data, learning_rate, total_bias);

		CHECK_CUDA(cudaDeviceSynchronize());
		CHECK_CUDA(cudaGetLastError());

		// Chuyển weights và bias đã cập nhật về CPU (nếu cần)
		// weights.to_host();
		// bias.to_host();
	}
	else if (device == "host")
	{
		for (size_t i = 0; i < weights.h_data.size(); i++)
		{
			weights.h_data[i] -= learning_rate * grad_weights.h_data[i];
		}
		for (size_t i = 0; i < bias.h_data.size(); i++)
		{
			bias.h_data[i] -= learning_rate * grad_bias.h_data[i];
		}
	}
}