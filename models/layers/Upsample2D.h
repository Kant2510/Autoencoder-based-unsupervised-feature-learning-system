#pragma once
#include "../tensor.h"
#include "../../kernels/upsample2d.h"

class Upsample2D
{
private:
    int scale_factor;
    Tensor cached_output;     // Lưu output của lần forward cuối cùng (dùng cho backward)
    Tensor cached_grad_input; // Lưu grad_input của lần backward cuối cùng (dùng

public:
    Upsample2D(int scale = 2) : scale_factor(scale) {};
    Tensor forward(const Tensor &input, const std::string &device = "host");
    Tensor backward(const Tensor &grad_output, const std::string &device = "host");
    void forward_loop_host(const Tensor &input, Tensor &output,
                           int channels, int batch_size,
                           int out_h, int out_w);
    void forward_loop_device(const Tensor &input, Tensor &output,
                             int channels, int batch_size,
                             int input_h, int input_w,
                             int output_h, int output_w);
    void backward_loop_host(const Tensor &grad_output, Tensor &grad_input,
                            int channels, int batch_size,
                            int out_h, int out_w);
    void backward_loop_device(const Tensor &grad_output, Tensor &grad_input,
                              int channels, int batch_size,
                              int input_h, int input_w,
                              int output_h, int output_w);
};