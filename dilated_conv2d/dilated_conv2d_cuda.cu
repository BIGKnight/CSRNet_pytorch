#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include "dilated_conv2d.h"

#define CUDA_KERNEL_LOOP(i ,n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); i+= blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

inline int GET_BLOCKS(const int N){
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


__global__ void add_bias_kernel(
    int n,
    float* data_out,
    const float* bias,
    const int out_channels,
    const int height_out, const int width_out
){
    CUDA_KERNEL_LOOP(index, n){
        const int c_col = (index / width_out / height_out) % out_channels;
        float value = bias[c_col];
        atomicAdd(data_out + index, value);
    }
}

__global__ void calculate_dbias_kernel(
    int n,
    const float* grad_output,
    float* grad_bias,
    const int out_channels,
    const int height_out, const int width_out
){
    CUDA_KERNEL_LOOP(index, n){
        const int c_col = (index / width_out / height_out) % out_channels;
        float value = grad_output[index];
        atomicAdd(grad_bias + c_col, value);
    }
}

__global__ void dilated_conv2d_im2col_kernel(
    int n,
    const float* data_im,
    const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int num_channels,
    const int height_col, const int width_col,
    float* data_col
    ){
    CUDA_KERNEL_LOOP(index, n){
        const int w_col = index % width_col;
        const int h_col = (index / width_col) % height_col;
        const int c_im = (index / width_col / height_col);
        const int c_col = c_im * kernel_h * kernel_w;

        const int h_in = h_col * stride_h;
        const int w_in = w_col * stride_w;

        float* data_col_ptr = data_col + (c_col * height_col + h_col) * width_col + w_col;
        const float* data_im_ptr = data_im + c_im * height * width;
        for (int i = 0; i < kernel_h; ++i) {
            for (int j = 0; j < kernel_w; ++j) {
                const int h_im = h_in + i * dilation_h;
                const int w_im = w_in + j * dilation_w;
                float value = data_im_ptr[h_im * width + w_im];
                *data_col_ptr = value;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}


__global__ void dilated_conv2d_col2im_kernel(
    const int n,
    const float* data_col,
    const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    float* grad_im
){
    CUDA_KERNEL_LOOP(index, n){
        // the relative location in the filter
        const int j = (index / width_col / height_col) % kernel_w;
        const int i = (index / width_col / height_col / kernel_w) % kernel_h;
        const int c = index / width_col / height_col / kernel_w / kernel_h; // which channel
        // 计算当前这个index对应的值被卷积操作的哪个内积点(也就是输出的spatial location)使用了.
        int w_out = index % width_col;
        int h_out = (index / width_col) % height_col;
        const int h_in = h_out * stride_h ;
        const int w_in = w_out * stride_w ;
        const int cur_inv_h_grid = h_in + i * dilation_h;
        const int cur_inv_w_grid = w_in + j * dilation_w;
        const float cur_top_grad = data_col[index];
        int cur_bottom_grad_pos = (c * height + cur_inv_h_grid) * width + cur_inv_w_grid;
        atomicAdd(grad_im + cur_bottom_grad_pos, cur_top_grad);
    }
}


void dilated_conv2d_im2col(cudaStream_t stream,
    const float* data_im,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_out, const int width_out,
    float* data_col){
    int num_kernels = in_channels * height_out * width_out;
    dilated_conv2d_im2col_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
            num_kernels,
            data_im,
            height, width,
            kernel_h, kernel_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            in_channels,
            height_out, width_out,
            data_col
    );
}


void dilated_conv2d_col2im(cudaStream_t stream,
    const float* data_col,
    const int in_channels, const int height, const int width,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_out, const int width_out,
    float* grad_im){
    int  num_kernels = in_channels * kernel_h * kernel_w * height_out * width_out;
    dilated_conv2d_col2im_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_col,
        in_channels, height, width,
        kernel_h, kernel_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        height_out, width_out,
        grad_im
    );
}

void add_bias(cudaStream_t stream,
    float* data_out,
    const float* bias,
    const int out_channels,
    const int height_out, const int width_out
    ){
    int num_kernels = out_channels * height_out * width_out;
    add_bias_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        data_out,
        bias,
        out_channels,
        height_out, width_out
    );
}

void calculate_dbias(cudaStream_t stream,
    const float* grad_output,
    float* grad_bias,
    const int out_channels,
    const int height_out, const int width_out
    ){
    int num_kernels = out_channels * height_out * width_out;
    calculate_dbias_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        num_kernels,
        grad_output,
        grad_bias,
        out_channels,
        height_out, width_out
    );
}
