#include <iostream>
#include <fstream>
#include <algorithm>
#include "dataset.h"

CIFAR10Dataset::CIFAR10Dataset() : rng(std::random_device{}()) {}

CIFAR10Dataset::~CIFAR10Dataset()
{
    // Giải phóng bộ nhớ Pinned khi hủy Dataset
    if (train_images_pinned)
        cudaFreeHost(train_images_pinned);
    if (test_images_pinned)
        cudaFreeHost(test_images_pinned);
}

bool CIFAR10Dataset::loadData(const std::string &data_path)
{
    std::cout << "Loading CIFAR-10 dataset..." << std::endl;

    // train_images.reserve(num_train * image_size);
    // 1. Cấp phát Pinned Memory cho toàn bộ Dataset
    // Train: 50,000 * 3072 * 4 bytes ~ 600 MB
    size_t train_size_bytes = (size_t)num_train * image_size * sizeof(float);
    CHECK_CUDA(cudaMallocHost((void **)&train_images_pinned, train_size_bytes));

    size_t test_size_bytes = (size_t)num_test * image_size * sizeof(float);
    CHECK_CUDA(cudaMallocHost((void **)&test_images_pinned, test_size_bytes));

    train_labels.reserve(num_train);

    int samples_per_file = 10000;
    for (int batch = 1; batch <= 5; batch++)
    {
        std::string filename = data_path + "/data_batch_" + std::to_string(batch) + ".bin";
        // if (!loadBatch(filename, train_images, train_labels))
        // {
        //     std::cerr << "Failed to load " << filename << std::endl;
        //     return false;
        // }
        // Tính offset: file 1 ghi từ 0, file 2 ghi từ 10000...
        int offset = (batch - 1) * samples_per_file;

        if (!loadBatchToBuffer(filename, train_images_pinned, train_labels, offset))
        {
            std::cerr << "Failed to load " << filename << std::endl;
            return false;
        }
    }

    std::string test_file = data_path + "/test_batch.bin";
    // if (!loadBatch(test_file, test_images, test_labels))
    // {
    //     std::cerr << "Failed to load " << test_file << std::endl;
    //     return false;
    // }
    if (!loadBatchToBuffer(test_file, test_images_pinned, test_labels, 0))
    {
        std::cerr << "Failed to load " << test_file << std::endl;
        return false;
    }

    std::cout << "Dataset loaded successfully!" << std::endl;
    std::cout << "Training samples: " << train_labels.size() << std::endl;
    std::cout << "Test samples: " << test_labels.size() << std::endl;

    shuffle_indices.resize(train_labels.size());
    for (size_t i = 0; i < shuffle_indices.size(); i++)
    {
        shuffle_indices[i] = i;
    }

    return true;
}

// bool CIFAR10Dataset::loadBatch(const std::string &filename,
//                                std::vector<float> &images,
//                                std::vector<int> &labels)
// {
//     std::ifstream file(filename, std::ios::binary);
//     if (!file.is_open())
//     {
//         return false;
//     }

//     const int record_size = 1 + 3072;
//     std::vector<unsigned char> buffer(record_size);

//     while (file.read(reinterpret_cast<char *>(buffer.data()), record_size))
//     {
//         labels.push_back(buffer[0]);

//         for (int i = 1; i < record_size; i++)
//         {
//             images.push_back(buffer[i] / 255.0f);
//         }
//     }

//     file.close();
//     return true;
// }
bool CIFAR10Dataset::loadBatchToBuffer(const std::string &filename, float *buffer_ptr, std::vector<int> &labels_vec, int offset_idx)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    // Cấu trúc file: [1 byte label] [3072 bytes image]
    const int entry_bytes = 1 + 3072;
    std::vector<unsigned char> temp_buffer(entry_bytes); // Buffer tạm để đọc 1 dòng

    // Đọc từng ảnh
    // Lưu ý: Có thể tối ưu hơn bằng cách đọc cả file vào RAM rồi parse,
    // nhưng cách này an toàn và đủ nhanh cho việc load 1 lần.
    int current_idx = offset_idx;

    while (file.read(reinterpret_cast<char *>(temp_buffer.data()), entry_bytes))
    {
        // 1. Label
        labels_vec.push_back(temp_buffer[0]);

        // 2. Image: Convert uint8 -> float [0, 1] và ghi vào Pinned Memory
        float *img_dest = buffer_ptr + (size_t)current_idx * image_size;

        for (int i = 0; i < 3072; i++)
        {
            // temp_buffer[1 + i] là pixel
            img_dest[i] = static_cast<float>(temp_buffer[1 + i]) / 255.0f;
        }
        current_idx++;
    }

    file.close();
    return true;
}

void CIFAR10Dataset::shuffle()
{
    std::shuffle(shuffle_indices.begin(), shuffle_indices.end(), rng);
}

void CIFAR10Dataset::getBatch(Tensor &batch, int start_idx, int batch_size, bool is_train)
{
    // const auto &images = is_train ? train_images : test_images;
    float *src_data_base = is_train ? train_images_pinned : test_images_pinned;
    int num_samples = is_train ? train_labels.size() : test_labels.size();

    int actual_batch = std::min(batch_size, num_samples - start_idx);
    batch.batch = actual_batch;
    // std::cout << "Check debugging" << std::endl;
    // Tensor batch(actual_batch, 3, 32, 32);
    // batch.allocate_pinned();

    // for (int b = 0; b < actual_batch; b++)
    // {
    //     int idx = is_train ? shuffle_indices[start_idx + b] : start_idx + b;

    //     for (int c = 0; c < 3; c++)
    //     {
    //         for (int h = 0; h < 32; h++)
    //         {
    //             for (int w = 0; w < 32; w++)
    //             {
    //                 int src_idx = idx * image_size + c * 1024 + h * 32 + w;
    //                 batch.at(b, c, h, w) = images[src_idx];
    //             }
    //         }
    //     }
    // }
    // Copy dữ liệu
    size_t single_img_size_bytes = image_size * sizeof(float); // 3072 * 4

#pragma omp parallel for // Nếu có OpenMP, parallel copy cho nhanh
    // std::cout << "Starting to copy batch data..." << std::endl;
    for (int b = 0; b < actual_batch; b++)
    {
        // Lấy index (đã shuffle)
        int src_idx = is_train ? shuffle_indices[start_idx + b] : start_idx + b;

        // Con trỏ nguồn (trong dataset pinned)
        float *src_ptr = src_data_base + (size_t)src_idx * image_size;

        // Con trỏ đích (trong batch pinned)
        float *dest_ptr = batch.h_pinned + (size_t)b * image_size;

        // Dùng memcpy thay vì 3 vòng for lồng nhau -> Tăng tốc đáng kể
        // Vì layout dataset (CHW) khớp với layout Tensor (CHW)
        // std::cout << "Copying image " << b << " from dataset index " << src_idx << std::endl;
        std::memcpy(dest_ptr, src_ptr, single_img_size_bytes);
        // std::cout << "Copied image " << b << std::endl;
    }
    // std::cout << "Batch data copy completed." << std::endl;
    // return batch;
}

int CIFAR10Dataset::getNumTrainBatches(int batch_size) const
{
    return (train_labels.size() + batch_size - 1) / batch_size;
}

int CIFAR10Dataset::getNumTestBatches(int batch_size) const
{
    return (test_labels.size() + batch_size - 1) / batch_size;
}

int CIFAR10Dataset::getSizeTrain() const
{
    return num_train;
}
int CIFAR10Dataset::getSizeTest() const
{
    return num_test;
}