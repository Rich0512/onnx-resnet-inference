/**
 * @file inference.cpp
 * @brief ONNX Runtime inference engine implementation
 */

#include "inference.h"
#include <chrono>
#include <algorithm>
#include <stdexcept>

namespace inference {

Engine::Engine(const std::string& model_path, int num_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ResNetInference") {
    
    session_options_.SetIntraOpNumThreads(num_threads);
    
    // Convert path to wide string for Windows
    std::wstring model_path_w(model_path.begin(), model_path.end());
    
    // Load model and measure time
    auto start = std::chrono::high_resolution_clock::now();
    session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options_);
    auto end = std::chrono::high_resolution_clock::now();
    
    load_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::vector<float> Engine::run(const std::vector<float>& input_data) {
    // Create input tensor
    std::vector<int64_t> input_shape = {BATCH_SIZE, CHANNELS, HEIGHT, WIDTH};
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );
    
    // Run inference
    const char* input_names[] = {INPUT_NAME};
    const char* output_names[] = {OUTPUT_NAME};
    
    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );
    
    // Extract output data
    float* output_data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + NUM_CLASSES);
}

std::vector<Result> Engine::get_top_k(const std::vector<float>& scores, int k) {
    std::vector<Result> results;
    results.reserve(scores.size());
    
    for (size_t i = 0; i < scores.size(); i++) {
        results.push_back({static_cast<int>(i), scores[i]});
    }
    
    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) { return a.score > b.score; });
    
    results.resize(std::min(k, static_cast<int>(results.size())));
    return results;
}

} // namespace inference
