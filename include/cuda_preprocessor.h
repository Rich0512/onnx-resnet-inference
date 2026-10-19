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

} // namespace cuda_preprocess
