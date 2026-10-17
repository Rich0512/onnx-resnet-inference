/**
 * @file preprocessor.cpp
 * @brief Image preprocessing implementation
 */

#include "preprocessor.h"
#include <stdexcept>
#include <iostream>

namespace preprocessor {

void preprocess_image(const cv::Mat& img, DataVector& input_data, int input_width, int input_height) {
    // Check if image is empty
    if(img.empty()) {
        throw std::runtime_error("Image is empty");
    }

    // Check if image size too small
    if(img.rows < 32 || img.cols < 32) {
        throw std::runtime_error("Image too small: " + std::to_string(img.cols) + "x" + std::to_string(img.rows) + ", minimum 32x32");
    }

    // If there is a problem, throw an exception (throw std::invalid_argument(...))
    if(img.channels() != 3) {
        throw std::invalid_argument("Image must be BGR");
    }


    // 1. Resize
    cv::Mat resized;
    if (img.cols != input_width || img.rows != input_height) {
        cv::resize(img, resized, cv::Size(input_width, input_height));
    } else {
        resized = img;
    }
    
    // 2. Fused Preprocessing (Operator Fusion + Single-Pass)
    // Precompute scale and bias constants: out = pixel * scale + bias
    // Equivalent to: (pixel / 255.0 - mean) / std
    constexpr float scale_r = 1.0f / (255.0f * STD_R);
    constexpr float bias_r  = -MEAN_R / STD_R;
    constexpr float scale_g = 1.0f / (255.0f * STD_G);
    constexpr float bias_g  = -MEAN_G / STD_G;
    constexpr float scale_b = 1.0f / (255.0f * STD_B);
    constexpr float bias_b  = -MEAN_B / STD_B;

    const int total_pixels = input_width * input_height;
    
    // Ensure output buffer has exact size (reusing allocated Pinned memory)
    input_data.resize(3 * total_pixels);

    // Get direct raw pointers for each channel in the destination CHW buffer
    float* r_ptr = input_data.data();
    float* g_ptr = r_ptr + total_pixels;
    float* b_ptr = g_ptr + total_pixels;

    // Single-pass traversal: OpenCV resized image is continuous BGR
    const uint8_t* src_ptr = resized.data;

    // Process all pixels sequentially: cache-friendly and easily auto-vectorized
    for (int i = 0; i < total_pixels; ++i) {
        uint8_t b = *src_ptr++;
        uint8_t g = *src_ptr++;
        uint8_t r = *src_ptr++;

        *r_ptr++ = static_cast<float>(r) * scale_r + bias_r;
        *g_ptr++ = static_cast<float>(g) * scale_g + bias_g;
        *b_ptr++ = static_cast<float>(b) * scale_b + bias_b;
    }
}

} // namespace preprocessor
