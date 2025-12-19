#pragma once

#include <random>
#include <string>
#include <vector>
#include "../models/tensor.h"

class CIFAR10Dataset
{
private:
  // Dùng Pinned Memory để chứa toàn bộ dataset
  float *train_images_pinned = nullptr;
  float *test_images_pinned = nullptr;
  std::vector<int> train_labels;
  std::vector<int> test_labels;

  int num_train = 50000;
  int num_test = 10000;
  int image_size = 3 * 32 * 32;

  std::vector<int> shuffle_indices;
  std::mt19937 rng;

  bool loadBatchToBuffer(const std::string &filename, float *buffer_ptr, std::vector<int> &labels_vec, int offset_idx);

public:
  CIFAR10Dataset();
  ~CIFAR10Dataset(); // Cần destructor để free pinned memory
  bool loadData(const std::string &data_path);
  void shuffle();
  // Trả về Tensor đã có h_pinned chứa data
  void getBatch(Tensor &batch, int start_idx, int batch_size, bool is_train = true);
  int getNumTrainBatches(int batch_size) const;
  int getNumTestBatches(int batch_size) const;
  int getSizeTrain() const;
  int getSizeTest() const;
};