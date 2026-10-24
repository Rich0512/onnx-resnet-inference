#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace cuda_preprocess {

/**
 * @brief Launch fused GPU preprocessing kernel:
 *        Bilinear Resize + BGR-to-RGB + ImageNet Normalization (scale & bias) + NCHW output.
 * 
 * @param src_bgr      Pointer to continuous BGR uint8 image in GPU memory (src_w * src_h * 3 bytes)
 * @param src_w        Original image width
 * @param src_h        Original image height
 * @param dst_nchw     Pointer to destination NCHW float buffer in GPU memory (3 * dst_w * dst_h floats)
 * @param dst_w        Target width (typically 224)
 * @param dst_h        Target height (typically 224)
 * @param stream       CUDA stream to execute on
 */
void launch_cuda_preprocess(
    const uint8_t* src_bgr, int src_w, int src_h,
    float* dst_nchw, int dst_w, int dst_h,
    cudaStream_t stream
);

/**
 * @brief Launch batched GPU preprocessing kernel for N images:
 *        Each image in the batch is processed in parallel on a 3D CUDA grid.
 * 
 * @param src_bgr_batch Pointer to contiguous array of N BGR images in GPU memory (N * src_w * src_h * 3 bytes)
 * @param src_w         Original image width
 * @param src_h         Original image height
 * @param dst_nchw_batch Pointer to contiguous destination 4D tensor (N * 3 * dst_w * dst_h floats)
 * @param dst_w         Target width (typically 224)
 * @param dst_h         Target height (typically 224)
 * @param batch_size    Number of images in the batch (N)
 * @param stream        CUDA stream to execute on
 */
void launch_cuda_preprocess_batch(
    const uint8_t* src_bgr_batch, int src_w, int src_h,
    float* dst_nchw_batch, int dst_w, int dst_h,
    int batch_size,
    cudaStream_t stream
);

} // namespace cuda_preprocess
