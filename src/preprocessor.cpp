/**
 * @file preprocessor.cpp
 * @brief Image preprocessing implementation
 */

#include "preprocessor.h"

namespace preprocessor {

std::vector<float> preprocess(const cv::Mat& img) {
    // 1. Resize to 224x224
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_SIZE, INPUT_SIZE));
    
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
    std::vector<float> input_data;
    input_data.reserve(3 * INPUT_SIZE * INPUT_SIZE);
    
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < INPUT_SIZE; h++) {
            for (int w = 0; w < INPUT_SIZE; w++) {
                input_data.push_back(channels[c].at<float>(h, w));
            }
        }
    }
    
    return input_data;
}

} // namespace preprocessor
