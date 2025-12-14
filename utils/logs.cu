#include "logs.h"
#include <string>

std::string createNewTrainFolder()
{
	fs::create_directories("./Phrase1/models");

	int idx = 1;
	while (true)
	{
		std::string folder = "./Phrase1/models/train" + std::to_string(idx);
		if (!fs::exists(folder))
		{
			fs::create_directories(folder);
			return folder;
		}
		idx++;
	}
}

void logBaseline(
	const std::string &outpath,
	int epoch,
	int total_epochs,
	float avg_loss,
	double epoch_time_sec,
	double avg_batch_time_ms,
	double throughput_img_per_sec,
	int batch_size,
	int num_batches)
{
	std::ofstream out(outpath, std::ios::app);

	if (!out.is_open())
	{
		std::cerr << "Cannot open: " << outpath << std::endl;
		return;
	}

	out << "========== Epoch " << epoch << "/" << total_epochs << " ==========\n";
	out << "Average Loss: " << avg_loss << "\n";
	out << "Epoch Time:  " << epoch_time_sec << " seconds\n";
	out << "Average Batch Time: " << avg_batch_time_ms << " ms\n";
	out << "Throughput: " << throughput_img_per_sec << " images/second\n";
	out << "Batch Size: " << batch_size << "\n";
	out << "Num Batches: " << num_batches << "\n";
	out << "==========================================\n\n";

	out.close();
}

// style: [=========>......] 50% Loss: 0.1234 Time: 15ms
void drawProgressBar(int current, int total, float current_loss, double batch_time_ms, int barWidth)
{
	float progress = (float)current / total;
	int pos = barWidth * progress;

	std::cout << "\r"; // Quay về đầu dòng
	std::cout << "[";

	for (int i = 0; i < barWidth; ++i)
	{
		if (i < pos)
			std::cout << "=";
		else if (i == pos)
			std::cout << ">";
		else
			std::cout << ".";
	}

	std::cout << "] " << int(progress * 100.0) << "% ";
	std::cout << current << "/" << total << " ";
	std::cout << "Loss: " << std::fixed << std::setprecision(4) << current_loss << " ";

	// Hiển thị thời gian/batch
	if (batch_time_ms > 1000.0)
	{
		std::cout << "| " << std::setprecision(2) << batch_time_ms / 1000.0 << "s/step";
	}
	else
	{
		std::cout << "| " << std::setprecision(0) << batch_time_ms << "ms/step";
	}

	std::cout << std::flush; // Bắt buộc phải flush để in ngay lập tức
}