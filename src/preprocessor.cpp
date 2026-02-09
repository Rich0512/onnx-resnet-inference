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
    
    // 2. BGR -> RGB and convert to float [0, 1]
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);
    
    // 3. ImageNet normalization: (pixel - mean) / std
    cv::Mat channels[3];
    cv::split(rgb, channels);
    channels[0] = (channels[0] - MEAN_R) / STD_R;  // R
    channels[1] = (channels[1] - MEAN_G) / STD_G;  // G
    channels[2] = (channels[2] - MEAN_B) / STD_B;  // B
    
    // 4. Convert HWC to CHW format
    input_data.clear();
    input_data.reserve(3 * input_width * input_height);
    
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < input_height; h++) {
            for (int w = 0; w < input_width; w++) {
                input_data.push_back(channels[c].at<float>(h, w));
            }
        }
    }
}

} // namespace preprocessor
