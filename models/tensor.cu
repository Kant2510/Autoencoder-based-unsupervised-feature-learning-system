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
	// std::fill(h_data.begin(), h_data.end(), value);
	if (h_pinned != nullptr)
	{
		std::fill(h_pinned, h_pinned + numel(), value);
	}
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
