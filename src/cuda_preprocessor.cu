#include "cuda_preprocessor.h"
#include <device_launch_parameters.h>
#include <cmath>

namespace cuda_preprocess {

// ImageNet Normalization Constants
__device__ __constant__ float d_scale_r = 1.0f / (255.0f * 0.229f);
__device__ __constant__ float d_bias_r  = -0.485f / 0.229f;
__device__ __constant__ float d_scale_g = 1.0f / (255.0f * 0.224f);
__device__ __constant__ float d_bias_g  = -0.456f / 0.224f;
__device__ __constant__ float d_scale_b = 1.0f / (255.0f * 0.225f);
__device__ __constant__ float d_bias_b  = -0.406f / 0.225f;

/**
 * @brief Fused Bilinear Resize + BGR->RGB + Normalization + NCHW Layout
 */
__global__ void preprocess_kernel(
    const uint8_t* __restrict__ src_bgr, int src_w, int src_h,
    float* __restrict__ dst_nchw, int dst_w, int dst_h) {

    // 1. Calculate 2D coordinates for current thread
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // 2. Map target coordinate to source image float coordinate (Bilinear)
    float src_x = (dst_x + 0.5f) * ((float)src_w / (float)dst_w) - 0.5f;
    float src_y = (dst_y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;

    // Clamp coordinates to valid range
    src_x = fmaxf(0.0f, fminf(src_x, (float)(src_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(src_h - 1)));

    // 3. Four surrounding neighbor coordinates
    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    // 4. Bilinear weights
    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    // 5. Memory indices in source continuous BGR image
    int idx00 = (y0 * src_w + x0) * 3;
    int idx01 = (y0 * src_w + x1) * 3;
    int idx10 = (y1 * src_w + x0) * 3;
    int idx11 = (y1 * src_w + x1) * 3;

    // 6. Bilinear sampling (B, G, R)
    float b = w00 * src_bgr[idx00]     + w01 * src_bgr[idx01]     + w10 * src_bgr[idx10]     + w11 * src_bgr[idx11];
    float g = w00 * src_bgr[idx00 + 1] + w01 * src_bgr[idx01 + 1] + w10 * src_bgr[idx10 + 1] + w11 * src_bgr[idx11 + 1];
    float r = w00 * src_bgr[idx00 + 2] + w01 * src_bgr[idx01 + 2] + w10 * src_bgr[idx10 + 2] + w11 * src_bgr[idx11 + 2];

    // 7. Write to NCHW destination (R -> Channel 0, G -> Channel 1, B -> Channel 2)
    int plane_size = dst_w * dst_h;
    int dst_idx = dst_y * dst_w + dst_x;

    dst_nchw[0 * plane_size + dst_idx] = r * d_scale_r + d_bias_r;
    dst_nchw[1 * plane_size + dst_idx] = g * d_scale_g + d_bias_g;
    dst_nchw[2 * plane_size + dst_idx] = b * d_scale_b + d_bias_b;
}

void launch_cuda_preprocess(
    const uint8_t* src_bgr, int src_w, int src_h,
    float* dst_nchw, int dst_w, int dst_h,
    cudaStream_t stream) {

    // 16x16 threads per block = 256 threads
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    preprocess_kernel<<<grid, block, 0, stream>>>(
        src_bgr, src_w, src_h,
        dst_nchw, dst_w, dst_h
    );
}

/**
 * @brief Batched Fused Bilinear Resize + BGR->RGB + Normalization + NCHW Layout
 */
__global__ void preprocess_batch_kernel(
    const uint8_t* __restrict__ src_bgr_batch, int src_w, int src_h,
    float* __restrict__ dst_nchw_batch, int dst_w, int dst_h,
    int batch_size) {

    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (dst_x >= dst_w || dst_y >= dst_h || b >= batch_size) return;

    // Pointer offset for batch index b
    size_t src_batch_offset = static_cast<size_t>(b) * (src_w * src_h * 3);
    const uint8_t* cur_src_bgr = src_bgr_batch + src_batch_offset;

    size_t dst_batch_offset = static_cast<size_t>(b) * (3 * dst_w * dst_h);
    float* cur_dst_nchw = dst_nchw_batch + dst_batch_offset;

    // Map target coordinate to source image float coordinate (Bilinear)
    float src_x = (dst_x + 0.5f) * ((float)src_w / (float)dst_w) - 0.5f;
    float src_y = (dst_y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, (float)(src_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(src_h - 1)));

    int x0 = (int)floorf(src_x);
    int y0 = (int)floorf(src_y);
    int x1 = min(x0 + 1, src_w - 1);
    int y1 = min(y0 + 1, src_h - 1);

    float dx = src_x - (float)x0;
    float dy = src_y - (float)y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    int idx00 = (y0 * src_w + x0) * 3;
    int idx01 = (y0 * src_w + x1) * 3;
    int idx10 = (y1 * src_w + x0) * 3;
    int idx11 = (y1 * src_w + x1) * 3;

    float val_b = w00 * cur_src_bgr[idx00]     + w01 * cur_src_bgr[idx01]     + w10 * cur_src_bgr[idx10]     + w11 * cur_src_bgr[idx11];
    float val_g = w00 * cur_src_bgr[idx00 + 1] + w01 * cur_src_bgr[idx01 + 1] + w10 * cur_src_bgr[idx10 + 1] + w11 * cur_src_bgr[idx11 + 1];
    float val_r = w00 * cur_src_bgr[idx00 + 2] + w01 * cur_src_bgr[idx01 + 2] + w10 * cur_src_bgr[idx10 + 2] + w11 * cur_src_bgr[idx11 + 2];

    int plane_size = dst_w * dst_h;
    int dst_idx = dst_y * dst_w + dst_x;

    cur_dst_nchw[0 * plane_size + dst_idx] = val_r * d_scale_r + d_bias_r;
    cur_dst_nchw[1 * plane_size + dst_idx] = val_g * d_scale_g + d_bias_g;
    cur_dst_nchw[2 * plane_size + dst_idx] = val_b * d_scale_b + d_bias_b;
}

void launch_cuda_preprocess_batch(
    const uint8_t* src_bgr_batch, int src_w, int src_h,
    float* dst_nchw_batch, int dst_w, int dst_h,
    int batch_size,
    cudaStream_t stream) {

    if (batch_size <= 0) return;

    dim3 block(16, 16, 1);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y, batch_size);

    preprocess_batch_kernel<<<grid, block, 0, stream>>>(
        src_bgr_batch, src_w, src_h,
        dst_nchw_batch, dst_w, dst_h,
        batch_size
    );
}

} // namespace cuda_preprocess
