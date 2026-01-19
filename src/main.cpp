#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// ImageNet class names (simplified - just showing index)
// In production, you would load the full 1000 class names from a file

/**
 * Preprocess image for ResNet inference
 * 1. Resize to 224x224
 * 2. Convert BGR to RGB
 * 3. Normalize to [0,1]
 * 4. Apply ImageNet normalization
 * 5. Convert to CHW format
 */
std::vector<float> preprocess_image(const cv::Mat& img) {
    // Resize to 224x224
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(224, 224));
    
    // BGR -> RGB and convert to float [0, 1]
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    
    // ImageNet normalization: (pixel - mean) / std
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    cv::Mat channels[3];
    cv::split(rgb, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;  // R
    channels[1] = (channels[1] - 0.456f) / 0.224f;  // G
    channels[2] = (channels[2] - 0.406f) / 0.225f;  // B
    
    // Convert HWC to CHW format (Channel, Height, Width)
    std::vector<float> input_data;
    input_data.reserve(3 * 224 * 224);
    
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                input_data.push_back(channels[c].at<float>(h, w));
            }
        }
    }
    
    return input_data;
}

// std::vector<float> preprocess_image_v2(const cv::Mat& img) {
//     float c1 = (0.485f * 255.0f) / 0.229f;
//     float c2 = (0.456f * 255.0f) / 0.224f;
//     float c3 = (0.406f * 255.0f) / 0.225f;
//     cv::Mat blob = cv::dnn::blobFromImage(
//         img,
//         1.0 / 255.0,
//         cv::Size(224, 224),
//         // cv::Scalar((0.485 * 255)/0.229f, (0.456 * 255)/0.224f, (0.406 * 255)/0.225f),
//         cv::Scalar(c1, c2, c3),
//         true,
//         false
//     ); 
//     return std::vector<float>(blob.begin<float>(), blob.end<float>());
// }


/**
 * Find top-k predictions from output scores
 */
std::vector<std::pair<int, float>> get_top_k(float* scores, int num_classes, int k) {
    std::vector<std::pair<int, float>> results;
    
    for (int i = 0; i < num_classes; i++) {
        results.push_back({i, scores[i]});
    }
    
    // Sort by score descending
    std::sort(results.begin(), results.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    results.resize(k);
    return results;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    std::string image_path = (argc > 1) ? argv[1] : "test_cat.jpg";
    std::string model_path_str = (argc > 2) ? argv[2] : "resnet18.onnx";
    
    std::cout << "========================================" << std::endl;
    std::cout << "   ResNet Inference Engine (C++)" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // ========== 1. Load Image ==========
    std::cout << "\n[1] Loading image: " << image_path << std::endl;
    cv::Mat img = cv::imread(image_path);
    
    if (img.empty()) {
        std::cerr << "Error: Cannot load image from " << image_path << std::endl;
        std::cerr << "Usage: main.exe <image_path> [model_path]" << std::endl;
        return 1;
    }
    std::cout << "    Image size: " << img.cols << "x" << img.rows << std::endl;
    
    // ========== 2. Initialize ONNX Runtime ==========
    std::cout << "\n[2] Initializing ONNX Runtime..." << std::endl;
    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNetInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);  // Use 4 threads
    
    // Convert model path to wide string for Windows
    std::wstring model_path_w(model_path_str.begin(), model_path_str.end());
    
    auto start_load = std::chrono::high_resolution_clock::now();
    Ort::Session session(env, model_path_w.c_str(), session_options);
    auto end_load = std::chrono::high_resolution_clock::now();
    
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count();
    std::cout << "    Model loaded in " << load_time << " ms" << std::endl;
    
    // ========== 3. Preprocess Image ==========
    std::cout << "\n[3] Preprocessing image..." << std::endl;
    
    auto start_preprocess = std::chrono::high_resolution_clock::now();
    std::vector<float> input_data = preprocess_image(img);
    // std::vector<float> input_data = preprocess_image_v2(img);
    auto end_preprocess = std::chrono::high_resolution_clock::now();
    
    auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(end_preprocess - start_preprocess).count();
    std::cout << "    Preprocessing done in " << preprocess_time / 1000.0 << " ms" << std::endl;
    
    // ========== 4. Create Input Tensor ==========
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    const char* input_names[] = {"input_tensor"};
    const char* output_names[] = {"output_tensor"};
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());
    
    // ========== 5. Run Inference ==========
    std::cout << "\n[4] Running inference..." << std::endl;
    
    // Warm-up run
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    
    // Timed inference (average of 10 runs)
    const int num_runs = 10;
    auto start_infer = std::chrono::high_resolution_clock::now();
    
    std::vector<Ort::Value> outputs;
    for (int i = 0; i < num_runs; i++) {
        outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    }
    
    auto end_infer = std::chrono::high_resolution_clock::now();
    auto infer_time = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer).count();
    double avg_infer_time = infer_time / 1000.0 / num_runs;
    
    std::cout << "    Inference time: " << avg_infer_time << " ms (avg of " << num_runs << " runs)" << std::endl;
    std::cout << "    Throughput: " << 1000.0 / avg_infer_time << " FPS" << std::endl;
    
    // ========== 6. Get Results ==========
    std::cout << "\n[5] Results:" << std::endl;
    
    float* output_data = outputs[0].GetTensorMutableData<float>();
    auto top5 = get_top_k(output_data, 1000, 5);
    
    std::cout << "    Top-5 Predictions:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "      " << (i + 1) << ". Class " << top5[i].first 
                  << " (score: " << top5[i].second << ")" << std::endl;
    }
    
    // ========== Summary ==========
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Performance Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Model load time:   " << load_time << " ms" << std::endl;
    std::cout << "  Preprocess time:   " << preprocess_time / 1000.0 << " ms" << std::endl;
    std::cout << "  Inference time:    " << avg_infer_time << " ms" << std::endl;
    std::cout << "  Total pipeline:    " << (load_time + preprocess_time/1000.0 + avg_infer_time) << " ms" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}