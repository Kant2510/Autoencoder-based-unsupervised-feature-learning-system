#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

// Hàm tìm thư mục trainX tiếp theo
std::string createNewTrainFolder();

// Hàm ghi baseline performance
void logBaseline(
	const std::string &outpath,
	int epoch,
	int total_epochs,
	float avg_loss,
	double epoch_time_sec,
	double avg_batch_time_ms,
	double throughput_img_per_sec,
	int batch_size,
	int num_batches);
