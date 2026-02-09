#pragma once

/**
 * @file preprocessor.h
 * @brief Image preprocessing for ResNet inference
 */

#include <vector>
#include <string> // Added
#include <opencv2/opencv.hpp>
#include "pinned_allocator.h" // Added

namespace preprocessor {

#ifdef USE_CUDA
    using DataVector = std::vector<float, inference::PinnedAllocator<float>>;
#else
    using DataVector = std::vector<float>;
#endif

/**
 * Preprocess image with buffer reuse
 * @param image Input OpenCV image
 * @param output_buffer Buffer to store result (will be resized if needed)
 * @param input_width Target width (default: 224)
 * @param input_height Target height (default: 224)
 */
void preprocess_image(const cv::Mat& image, DataVector& output_buffer, int input_width = 224, int input_height = 224);

// ImageNet normalization constants
constexpr float MEAN_R = 0.485f;
constexpr float MEAN_G = 0.456f;
constexpr float MEAN_B = 0.406f;
constexpr float STD_R = 0.229f;
constexpr float STD_G = 0.224f;
constexpr float STD_B = 0.225f;

// ResNet input size
constexpr int INPUT_SIZE = 224;

} // namespace preprocessor
