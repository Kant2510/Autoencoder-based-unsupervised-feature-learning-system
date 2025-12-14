#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

// Macro kiểm tra lỗi CUDA (Cực kỳ quan trọng để debug)
#define CHECK_CUDA(call)                                                  \
    {                                                                     \
        const cudaError_t error = call;                                   \
        if (error != cudaSuccess)                                         \
        {                                                                 \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                      << cudaGetErrorString(error) << std::endl;          \
            exit(1);                                                      \
        }                                                                 \
    }

// Hàm chia lưới (Grid) cho kernel
inline int GET_BLOCKS(int total_threads, int threads_per_block)
{
    return (total_threads + threads_per_block - 1) / threads_per_block;
}

class Tensor
{
public:
    std::vector<float> h_data; // Host data (CPU)
    float *d_data = nullptr;   // Device data (GPU)

    int batch, channels, height, width;

    Tensor() : d_data(nullptr), batch(0), channels(0), height(0), width(0) {}

    Tensor(int b, int c, int h, int w);

    Tensor(const Tensor &other);            // deep copy
    Tensor &operator=(const Tensor &other); // deep copy

    Tensor(Tensor &&other) noexcept; // move
    Tensor &operator=(Tensor &&other) noexcept;

    // Lấy tổng số phần tử
    int numel() const;

    // --- CÁC HÀM TIỆN ÍCH CHO INDEXING (QUAN TRỌNG) ---
    // Ánh xạ index 4D (NCHW) sang 1D
    inline int index(int n, int c, int h, int w) const;
    // Truy cập phần tử (Read/Write)
    float &at(int n, int c, int h, int w);
    // Truy cập phần tử (Read-only)
    float at(int n, int c, int h, int w) const;

    // --- CÁC HÀM KHỞI TẠO DỮ LIỆU ---
    void fill(float value);
    void zeros(const std::string &device = "host");
    // Khởi tạo ngẫu nhiên (dùng cho Weights) - He Initialization
    void randn(float mean, float stddev);
    // In shape ra màn hình để debug
    void print_shape() const;

    // Cấp phát bộ nhớ trên GPU
    void allocate_device()
    {
        if (d_data == nullptr)
        {
            CHECK_CUDA(cudaMalloc(&d_data, numel() * sizeof(float)));
        }
    }

    // Copy CPU -> GPU
    void to_device()
    {
        if (d_data == nullptr)
            allocate_device();
        CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), numel() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Copy GPU -> CPU (Dùng để debug hoặc lấy kết quả cuối)
    void to_host()
    {
        if (d_data != nullptr)
        {
            CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, numel() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    // Giải phóng bộ nhớ GPU
    void free_device()
    {
        if (d_data != nullptr)
        {
            cudaFree(d_data);
            d_data = nullptr;
        }
    }

    // Destructor (Tùy chọn: có thể gọi free_device ở đây nếu muốn quản lý tự động RAII)
    ~Tensor()
    {
        // Lưu ý: Trong thực tế C++, cần cẩn thận với copy constructor để tránh double free.
        // Ở code mẫu này tôi để việc free thủ công hoặc quản lý ở lớp cao hơn.
        // free_device();
    }
};