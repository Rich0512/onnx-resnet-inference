#pragma once

/**
 * @file preprocessor.h
 * @brief Image preprocessing for ResNet inference
 */

#include <vector>
#include <opencv2/opencv.hpp>

namespace preprocessor {

/**
 * Preprocess image for ResNet inference
 * 
 * Steps:
 * 1. Resize to 224x224
 * 2. Convert BGR to RGB
 * 3. Normalize to [0, 1]
 * 4. Apply ImageNet normalization
 * 5. Convert HWC to CHW format
 * 
 * @param img Input image (BGR format from OpenCV)
 * @return Preprocessed image data in CHW format
 */
std::vector<float> preprocess(const cv::Mat& img);

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
