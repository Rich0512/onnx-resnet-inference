/**
 * @file inference.cpp
 * @brief ONNX Runtime inference engine implementation with GPU support
 */

#include "inference.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stdexcept>

namespace inference {

Engine::Engine(const std::string& model_path, int num_threads, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ResNetInference")
    , use_gpu_(false) {
    
    session_options_.SetIntraOpNumThreads(num_threads);
    
    // Configure GPU if requested
    #ifdef USE_CUDA
    if (use_gpu) {
        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options_.AppendExecutionProvider_CUDA(cuda_options);
            use_gpu_ = true;
            std::cout << "    [GPU mode: CUDA enabled]\n";
        } catch (const Ort::Exception& e) {
            std::cerr << "    [GPU mode: Failed - " << e.what() << "]\n";
            std::cerr << "    [Falling back to CPU]\n";
            use_gpu_ = false;
        }
    }
    #else
    if (use_gpu) {
        std::cerr << "    [GPU mode: Not compiled with CUDA support]\n";
        std::cerr << "    [Rebuild with: cmake -DUSE_CUDA=ON ..]\n";
    }
    #endif
    
    if (!use_gpu_) {
        std::cout << "    [CPU mode: " << num_threads << " threads]\n";
    }
    
    // Convert path to wide string for Windows
    std::wstring model_path_w(model_path.begin(), model_path.end());
    
    // Load model and measure time
    auto start = std::chrono::high_resolution_clock::now();
    session_ = std::make_unique<Ort::Session>(env_, model_path_w.c_str(), session_options_);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Model path: " << model_path << ", Threads: " << num_threads << ", GPU: " << use_gpu << ", Load time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
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
