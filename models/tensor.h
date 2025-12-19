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
    float *h_pinned = nullptr; // Host pinned memory (CPU pinned)
    float *d_data = nullptr;   // Device data (GPU)

    int batch, channels, height, width;

    Tensor() : d_data(nullptr), batch(0), channels(0), height(0), width(0) {}

    Tensor(int b, int c, int h, int w);

    // Tensor(const Tensor &other);            // deep copy
    // Tensor &operator=(const Tensor &other); // deep copy

    // Tensor(Tensor &&other) noexcept; // move
    // Tensor &operator=(Tensor &&other) noexcept;

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    Tensor(Tensor &&) noexcept = default;
    Tensor &operator=(Tensor &&) noexcept = default;

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
        {
            // allocate_device();
            std::cout << "Device memory not allocated. Call ensure_device_memory() first." << std::endl;
            throw std::runtime_error("Device memory not allocated. Call ensure_device_memory() first.");
        }
        // CHECK_CUDA(cudaMemcpy(d_data, h_data.data(), numel() * sizeof(float), cudaMemcpyHostToDevice));
        // Sử dụng pinned memory nếu có
        if (h_pinned != nullptr)
        {
            // std::cout << "Using pinned memory for to_device()" << std::endl;
            CHECK_CUDA(cudaMemcpy(d_data, h_pinned, numel() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // Copy GPU -> CPU (Dùng để debug hoặc lấy kết quả cuối)
    void to_host()
    {
        if (d_data != nullptr)
        {
            // CHECK_CUDA(cudaMemcpy(h_data.data(), d_data, numel() * sizeof(float), cudaMemcpyDeviceToHost));
            // Sử dụng pinned memory nếu có
            if (h_pinned != nullptr)
            {
                // std::cout << "Using pinned memory for to_host()" << std::endl;
                CHECK_CUDA(cudaMemcpy(h_pinned, d_data, numel() * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
    }

    size_t capacity = 0;        // Dung lượng thực tế (số phần tử float) đang có trên GPU
    size_t capacity_pinned = 0; // Dung lượng thực tế (số phần tử float) đang có trên CPU pinned

    void allocate_pinned()
    {
        size_t needed = numel();
        if (needed > capacity_pinned)
        {
            if (h_pinned)
                cudaFreeHost(h_pinned);
            // cudaHostAllocDefault cho phép portable pinned memory
            CHECK_CUDA(cudaHostAlloc((void **)&h_pinned, needed * sizeof(float), cudaHostAllocDefault));
            capacity_pinned = needed;
        }
    }

    // Hàm cấp phát thông minh (Chỉ malloc nếu chưa đủ chỗ)
    void ensure_device_memory(int needed_elements)
    {
        if (needed_elements > capacity)
        {
            // Nếu chưa có hoặc không đủ chỗ -> Cấp phát mới (Lớn hơn hoặc bằng)
            if (d_data != nullptr)
            {
                cudaFree(d_data);
            }

            // Cấp phát dư ra một chút hoặc đúng bằng needed
            // Ở đây ta cấp đúng bằng needed để đơn giản
            CHECK_CUDA(cudaMalloc(&d_data, needed_elements * sizeof(float)));
            capacity = needed_elements;

            // Debug log (để thấy nó chỉ chạy 1 lần)
            std::cout << "Allocated GPU memory: " << capacity * 4 / 1024 << " KB" << std::endl;
        }
        // Nếu capacity >= needed_elements -> KHÔNG LÀM GÌ CẢ (Tái sử dụng)
    }

    // Hàm resize logic (cập nhật shape nhưng giữ nguyên vùng nhớ nếu đủ)
    void reshape_if_needed(int b, int c, int h, int w)
    {
        batch = b;
        channels = c;
        height = h;
        width = w;
        int needed = b * c * h * w;
        ensure_device_memory(needed);
    }

    // Giải phóng bộ nhớ GPU
    void free_device()
    {
        if (d_data != nullptr)
        {
            cudaFree(d_data);
            d_data = nullptr;
            capacity = 0;
        }
        if (h_pinned != nullptr)
        {
            cudaFreeHost(h_pinned);
            h_pinned = nullptr;
            capacity_pinned = 0;
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