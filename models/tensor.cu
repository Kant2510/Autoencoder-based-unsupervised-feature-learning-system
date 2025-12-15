#include <vector>
#include <iostream>
#include <numeric>
#include <random>
#include <cassert>
#include <iomanip>
#include "tensor.h"

Tensor::Tensor(int b, int c, int h, int w) : batch(b), channels(c), height(h), width(w)
{
	int size = batch * channels * height * width;
	h_data.resize(size, 0.0f); // Mặc định toàn số 0
	d_data = nullptr;
}

int Tensor::numel() const
{
	return batch * channels * height * width;
}

inline int Tensor::index(int n, int c, int h, int w) const
{
	// Giả sử shape chuẩn là 4 chiều: [N, C, H, W]
	// assert(shape.size() == 4);
	return n * (channels * height * width) +
		   c * (height * width) +
		   h * (width) +
		   w;
}

float &Tensor::at(int n, int c, int h, int w)
{
	return h_data[index(n, c, h, w)];
}

float Tensor::at(int n, int c, int h, int w) const
{
	return h_data[index(n, c, h, w)];
}

void Tensor::fill(float value)
{
	std::fill(h_data.begin(), h_data.end(), value);
}

void Tensor::zeros(const std::string &device)
{
	if (device == "host")
	{
		fill(0.0f);
	}
	else if (device == "device")
	{
		// if (d_data == nullptr)
		// 	allocate_device();
		CHECK_CUDA(cudaMemset(d_data, 0, numel() * sizeof(float)));
	}
}

void Tensor::randn(float mean, float stddev)
{
	std::default_random_engine generator;
	std::normal_distribution<float> distribution(mean, stddev);
	for (auto &x : h_data)
		x = distribution(generator);
}

void Tensor::print_shape() const
{
	std::cout << "Shape: (";
	std::cout << batch << ", " << channels << ", " << height << ", " << width;
	std::cout << ")" << std::endl;
}

Tensor::Tensor(const Tensor &other)
	: batch(other.batch),
	  channels(other.channels),
	  height(other.height),
	  width(other.width),
	  h_data(other.h_data),
	  d_data(nullptr)
{
	if (other.d_data)
	{
		allocate_device();
		CHECK_CUDA(cudaMemcpy(
			d_data,
			other.d_data,
			numel() * sizeof(float),
			cudaMemcpyDeviceToDevice));
	}
}
Tensor &Tensor::operator=(const Tensor &other)
{
	if (this == &other)
		return *this;

	free_device();

	batch = other.batch;
	channels = other.channels;
	height = other.height;
	width = other.width;
	h_data = other.h_data;

	if (other.d_data)
	{
		allocate_device();
		CHECK_CUDA(cudaMemcpy(
			d_data,
			other.d_data,
			numel() * sizeof(float),
			cudaMemcpyDeviceToDevice));
	}

	return *this;
}
Tensor::Tensor(Tensor &&other) noexcept
	: h_data(std::move(other.h_data)),
	  d_data(other.d_data),
	  batch(other.batch),
	  channels(other.channels),
	  height(other.height),
	  width(other.width)
{
	other.d_data = nullptr;
	other.batch = other.channels = other.height = other.width = 0;
}
Tensor &Tensor::operator=(Tensor &&other) noexcept
{
	if (this == &other)
		return *this;

	free_device();

	h_data = std::move(other.h_data);
	d_data = other.d_data;

	batch = other.batch;
	channels = other.channels;
	height = other.height;
	width = other.width;

	other.d_data = nullptr;
	other.batch = other.channels = other.height = other.width = 0;

	return *this;
}
