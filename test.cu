#include <iostream>
#include <vector>
#include <iomanip>
#include "models/layers/Conv2D.h"
#include "models/layers/MaxPool2D.h"
#include "models/layers/Relu.h"
#include "models/layers/Upsample2D.h"
#include "models/autoencoder.h"
#include "optimizer/loss.h"
#include "utils/logs.h"
#include "dataset/dataset.h"

void print_vector(const std::vector<float> &vec, int size_limit = 10)
{
	std::cout << std::fixed << std::setprecision(4);
	for (int i = 0; i < std::min((int)vec.size(), size_limit); ++i)
	{
		std::cout << vec[i] << " ";
	}
	if (vec.size() > size_limit)
		std::cout << "...";
	std::cout << std::endl;
}

void print_float_array(const float *arr, int size, int size_limit = 10)
{
	std::cout << std::fixed << std::setprecision(4);
	for (int i = 0; i < std::min(size, size_limit); ++i)
	{
		std::cout << arr[i] << " ";
	}
	if (size > size_limit)
		std::cout << "...";
	std::cout << std::endl;
}

void compare(const Tensor &h_output, const Tensor &d_output)
{
	bool match = true;
	float total_diff = 0.0f;
	for (size_t i = 0; i < h_output.h_data.size(); ++i)
	{
		total_diff += std::abs(h_output.h_data[i] - d_output.h_data[i]);
		if (std::abs(h_output.h_data[i] - d_output.h_data[i]) > 1e-3)
		{
			match = false;
			std::cout << "Mismatch at index " << i << ": Host=" << h_output.h_data[i]
					  << ", Device=" << d_output.h_data[i] << std::endl;
			// break;
		}
	}
	std::cout << "Total output size: " << h_output.h_data.size() << std::endl;
	std::cout << "Total output size: " << d_output.h_data.size() << std::endl;
	std::cout << "Total difference between Host and Device outputs: " << total_diff << std::endl;
	std::cout << "Verifying Host vs Device output..." << std::endl;
	if (match)
		std::cout << "Host and Device outputs match!" << std::endl;
	else
		std::cout << "Host and Device outputs do NOT match!" << std::endl;
}

Tensor test_conv2d(const Tensor &input)
{
	std::cout << "\n=== Testing Conv2D Layer ===" << std::endl;

	// Create Conv2D layer: 3 input channel, 6 output channels, 3x3 kernel, stride=1, padding=1
	Conv2D conv(3, 6, 3, 1, 1);

	// Perform forward pass
	Tensor h_output = conv.forward(input, "host");
	Tensor d_output = conv.forward(input, "device");

	std::cout << "Output shape:" << std::endl;
	h_output.print_shape();
	d_output.print_shape();

	std::cout << "Output (first 20 values):" << std::endl;
	print_vector(h_output.h_data, 20);
	// print_vector(d_output.h_data, 20);
	print_float_array(d_output.h_data.data(), h_output.numel(), 20);

	// Verify that host and device outputs match
	compare(h_output, d_output);

	return d_output;
}

Tensor test_maxpool(const Tensor &input)
{
	std::cout << "\n=== Testing MaxPool2D Layer ===" << std::endl;

	// Create MaxPool2D layer with 2x2 pooling and stride 2
	MaxPool2D maxpool;
	Tensor h_output = maxpool.forward(input, "host");
	Tensor d_output = maxpool.forward(input, "device");

	std::cout << "Output shape:" << std::endl;
	h_output.print_shape();
	d_output.print_shape();

	std::cout << "Output (first 20 values):" << std::endl;
	print_vector(h_output.h_data, 20);
	print_vector(d_output.h_data, 20);

	// Verify that host and device outputs match
	compare(h_output, d_output);

	return d_output;
}

Tensor test_relu(const Tensor &input)
{
	std::cout << "\n=== Testing ReLU Layer ===" << std::endl;

	ReLU relu;
	Tensor h_output = relu.forward(input, "host");
	Tensor d_output = relu.forward(input, "device");

	std::cout << "Output shape:" << std::endl;
	h_output.print_shape();
	d_output.print_shape();

	std::cout << "Output (first 20 values):" << std::endl;
	print_vector(h_output.h_data, 20);
	print_vector(d_output.h_data, 20);

	// Verify that host and device outputs match
	compare(h_output, d_output);

	return d_output;
}

Tensor test_upsample(const Tensor &input)
{
	std::cout << "\n=== Testing Upsample Layer ===" << std::endl;

	// Create Upsample layer with scale factor 2
	Upsample2D upsample(2);
	Tensor h_output = upsample.forward(input, "host");
	Tensor d_output = upsample.forward(input, "device");

	std::cout << "Output shape:" << std::endl;
	h_output.print_shape();
	d_output.print_shape();

	std::cout << "Output (first 20 values):" << std::endl;
	print_vector(h_output.h_data, 20);
	print_vector(d_output.h_data, 20);

	// Verify that host and device outputs match
	compare(h_output, d_output);

	return d_output;
}

void train(int BATCH_SIZE, int EPOCHS, float LEARNING_RATE, int LIMIT, const std::string &LOAD_PATH)
{

	// tạo thư mục trainX
	std::string trainFolder = createNewTrainFolder();
	std::string weightOutPath = trainFolder + "/weights.bin";
	std::string reportOutPath = trainFolder + "/baseline.txt";

	std::cout << "Saving weights to: " << weightOutPath << std::endl;
	std::cout << "Saving baseline report to: " << reportOutPath << std::endl;

	// load dataset
	CIFAR10Dataset dataset;
	if (!dataset.loadData("./cifar-10-batches-bin"))
	{
		std::cerr << "Failed to load dataset!" << std::endl;
		return;
	}

	Autoencoder model;
	MSELoss loss_fn;

	// Load pretrained weights
	if (!LOAD_PATH.empty())
	{
		std::cout << "\nLoading pretrained weights from: " << LOAD_PATH << std::endl;
		model.loadWeights(LOAD_PATH);
	}

	int num_batches;
	int num_data = dataset.getSizeTrain();

	if (LIMIT > 0)
	{
		num_data = LIMIT;
		num_batches = LIMIT / BATCH_SIZE;
		if (num_batches < 1)
			num_batches = 1;
	}
	else
	{
		num_batches = dataset.getNumTrainBatches(BATCH_SIZE);
	}

	std::cout << "\n========== Training Started with " << num_data << " samples ==========" << std::endl;
	std::cout << "Batch size: " << BATCH_SIZE << std::endl;
	std::cout << "Num batches: " << num_batches << std::endl;
	std::cout << "Epochs: " << EPOCHS << std::endl;
	std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
	std::cout << "======================================\n"
			  << std::endl;

	for (int epoch = 0; epoch < EPOCHS; epoch++)
	{

		auto epoch_start = std::chrono::high_resolution_clock::now();

		dataset.shuffle();
		float total_loss = 0.0f;
		double total_batch_time_ms = 0.0;

		std::cout << "Epoch [" << epoch + 1 << "/" << EPOCHS << "]\n";

		for (int batch_idx = 0; batch_idx < num_batches; batch_idx++)
		{

			int start_index = batch_idx * BATCH_SIZE;
			if (LIMIT > 0 && start_index >= LIMIT)
				break;

			auto b0 = std::chrono::high_resolution_clock::now();

			Tensor batch = dataset.getBatch(start_index, BATCH_SIZE, true);
			batch.allocate_device();
			batch.to_device();
			Tensor output = model.forward(batch, "device");

			float batch_loss = loss_fn.forward(output, batch, "device");
			total_loss += batch_loss;

			Tensor grad = loss_fn.backward(output, batch, "device");
			model.backward(grad, LEARNING_RATE, "device");

			auto b1 = std::chrono::high_resolution_clock::now();
			total_batch_time_ms += std::chrono::duration<double, std::milli>(b1 - b0).count();

			if ((batch_idx + 1) % 100 == 0)
			{
				std::cout << "  Batch [" << batch_idx + 1 << "/" << num_batches
						  << "] Loss: " << batch_loss << std::endl;
			}
		}

		auto epoch_end = std::chrono::high_resolution_clock::now();
		double epoch_time_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();

		float avg_loss = total_loss / num_batches;
		double avg_batch_time_ms = total_batch_time_ms / num_batches;

		double throughput = (double)(BATCH_SIZE * num_batches) / epoch_time_sec;

		std::cout << "Epoch [" << epoch + 1 << "/" << EPOCHS << "] "
				  << "Loss: " << std::fixed << std::setprecision(6) << avg_loss
				  << " Time: " << epoch_time_sec << "s" << std::endl;

		// ghi baseline
		logBaseline(
			reportOutPath,
			epoch + 1,
			EPOCHS,
			avg_loss,
			epoch_time_sec,
			avg_batch_time_ms,
			throughput,
			BATCH_SIZE,
			num_batches);
	}

	// lưu trọng số vào trainX/weights.bin
	model.saveWeights(weightOutPath);

	std::cout << "\n========== Training Completed ==========" << std::endl;
	std::cout << "Weights saved to: " << weightOutPath << std::endl;
}

Tensor test_forward(const Tensor &input)
{
	std::cout << "\n=== Testing Autoencoder Forward ===" << std::endl;
	Autoencoder model;
	MSELoss loss_fn;
	Tensor truth_output = input; // Autoencoder cố gắng tái tạo lại đầu vào

	Tensor h_output = model.forward(input, "host");
	float h_loss = loss_fn.forward(h_output, input, "host");
	std::cout << "Host Loss: " << h_loss << std::endl;
	Tensor d_output = model.forward(input, "device");
	float d_loss = loss_fn.forward(d_output, input, "device");
	std::cout << "Device Loss: " << d_loss << std::endl;

	Tensor h_grad = loss_fn.backward(h_output, input, "host");
	h_grad = model.backward(h_grad, 0.001, "host");
	Tensor d_grad = loss_fn.backward(d_output, input, "device");
	d_grad = model.backward(d_grad, 0.001, "device");

	compare(h_grad, d_grad);

	std::cout << "Output (first 20 values):" << std::endl;
	print_vector(h_grad.h_data, 20);
	print_vector(d_grad.h_data, 20);

	// Verify that host and device outputs match
	compare(h_output, d_output);
	std::cout << "Diff between Host and Device Loss: " << std::abs(h_loss - d_loss) << std::endl;
	if (std::abs(h_loss - d_loss) < 1e-3)
	{
		std::cout << "Host and Device losses match!" << std::endl;
	}
	else
	{
		std::cout << "Host and Device losses do NOT match!" << std::endl;
	}
	return d_output;
}
int main(int argc, char *argv[])
{
	// Create a simple 1x1x4x4 input Tensor (batch=1, channels=1, height=4, width=4)
	// Tensor input(1, 1, 8, 8);
	// for (int n = 0; n < 1; ++n)
	// {
	// 	for (int c = 0; c < 1; ++c)
	// 	{
	// 		for (int h = 0; h < 8; ++h)
	// 		{
	// 			for (int w = 0; w < 8; ++w)
	// 			{
	// 				input.at(n, c, h, w) = static_cast<float>(n * 100 + c * 10 + h * 8 + w + 1);
	// 			}
	// 		}
	// 	}
	// }
	// input.allocate_device();
	// input.to_device();
	// std::cout << "Input shape:" << std::endl;
	// input.print_shape();

	// // Test Autoencoder forward
	// Tensor output = test_forward(input);
	int batch_size = 64;
	int epochs = 5;
	float lr = 0.1f;
	// 2% of data
	int data_limit = -1;
	std::string load_path = "";

	std::cout << "CIFAR-10 Autoencoder - CPU Implementation" << std::endl;
	std::cout << "=========================================\n"
			  << std::endl;

	train(batch_size, epochs, lr, data_limit, load_path);

	return 0;
}