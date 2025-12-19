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

	// 1. Cấp phát vùng đệm Input TRƯỚC vòng lặp
	Tensor gpu_batch_buffer;
	Tensor batch(BATCH_SIZE, 3, 32, 32);
	batch.allocate_pinned();
	// batch.ensure_device_memory(BATCH_SIZE * 3 * 32 * 32);
	// Cấp sẵn dung lượng cho max batch size (ví dụ 64)
	gpu_batch_buffer.ensure_device_memory(BATCH_SIZE * 3 * 32 * 32);

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

			dataset.getBatch(batch, start_index, BATCH_SIZE, true);

			// Update shape cho buffer GPU (phòng trường hợp batch cuối lẻ)
			gpu_batch_buffer.batch = batch.batch;
			gpu_batch_buffer.channels = batch.channels;
			gpu_batch_buffer.height = batch.height;
			gpu_batch_buffer.width = batch.width;

			// Copy H2D vào buffer cố định (KHÔNG allocate mới)
			CHECK_CUDA(cudaMemcpy(
				gpu_batch_buffer.d_data,
				batch.h_pinned, // Hoặc batch.h_pinned
				gpu_batch_buffer.numel() * sizeof(float),
				cudaMemcpyHostToDevice));
			// batch.allocate_device();
			// batch.to_device();
			// gpu_batch_buffer.to_device();
			model.forward(gpu_batch_buffer, "device");

			float batch_loss = loss_fn.forward(model.conv5.cached_output, gpu_batch_buffer, "device");
			total_loss += batch_loss;

			loss_fn.backward(model.conv5.cached_output, gpu_batch_buffer, "device");
			model.backward(loss_fn.cached_grad, LEARNING_RATE, "device");
			auto b1 = std::chrono::high_resolution_clock::now();
			double step_time = std::chrono::duration<double, std::milli>(b1 - b0).count();
			total_batch_time_ms += step_time;

			// if ((batch_idx + 1) % 100 == 0)
			// {
			// 	std::cout << "  Batch [" << batch_idx + 1 << "/" << num_batches
			// 			  << "] Loss: " << batch_loss << std::endl;
			// }
			// --- CẬP NHẬT THANH TIẾN TRÌNH ---
			// Gọi hàm vẽ bar mỗi batch (hoặc mỗi n batch nếu muốn giảm lag console)
			drawProgressBar(batch_idx + 1, num_batches, batch_loss, step_time);

			// Giải phóng bộ nhớ tạm trong vòng lặp (Rất quan trọng để tránh đầy VRAM)
			// batch.free_device();
			// output.free_device();
			// grad.free_device();
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

	gpu_batch_buffer.free_device();
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
	int epochs = 7;
	float lr = 0.004f;
	// 2% of data
	int data_limit = 1024;
	std::string load_path = "";

	// Parse command-line arguments
	for (int i = 1; i < argc; i++)
	{
		std::string arg = argv[i];

		if (arg == "--batch" && i + 1 < argc)
		{
			batch_size = std::atoi(argv[++i]);
		}
		else if (arg == "--epochs" && i + 1 < argc)
		{
			epochs = std::atoi(argv[++i]);
		}
		else if (arg == "--lr" && i + 1 < argc)
		{
			lr = std::atof(argv[++i]);
		}
		else if (arg == "--limit" && i + 1 < argc)
		{
			data_limit = std::atoi(argv[++i]);
		}
		else if (arg == "--load" && i + 1 < argc)
		{
			load_path = argv[++i];
		}
	}

	std::cout << "CIFAR-10 Autoencoder - CPU Implementation" << std::endl;
	std::cout << "=========================================\n"
			  << std::endl;

	train(batch_size, epochs, lr, data_limit, load_path);

	return 0;
}