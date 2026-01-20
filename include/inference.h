#pragma once

/**
 * @file inference.h
 * @brief ONNX Runtime inference engine wrapper with optional GPU acceleration
 */

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace inference {

/**
 * Inference result containing class ID and confidence score
 */
struct Result {
    int class_id;
    float score;
};

/**
 * ONNX Runtime inference engine
 * Supports both CPU and GPU (CUDA) execution
 */
class Engine {
public:
    /**
     * Constructor
     * @param model_path Path to ONNX model file
     * @param num_threads Number of threads for CPU inference (default: 4)
     * @param use_gpu Enable GPU acceleration if available (default: false)
     */
    explicit Engine(const std::string& model_path, 
                    int num_threads = 4,
                    bool use_gpu = false);
    
    /**
     * Run inference on preprocessed input data
     * @param input_data Preprocessed image data in CHW format
     * @return Raw output scores (1000 classes for ImageNet)
     */
    std::vector<float> run(const std::vector<float>& input_data);
    
    /**
     * Get top-k predictions
     * @param scores Output scores from run()
     * @param k Number of top predictions to return
     * @return Vector of (class_id, score) pairs sorted by score
     */
    static std::vector<Result> get_top_k(const std::vector<float>& scores, int k);
    
    /**
     * Get model load time in milliseconds
     */
    long long get_load_time_ms() const { return load_time_ms_; }
    
    /**
     * Check if GPU acceleration is enabled
     */
    bool is_using_gpu() const { return use_gpu_; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    long long load_time_ms_ = 0;
    bool use_gpu_ = false;
    
    // Model input/output names
    static constexpr const char* INPUT_NAME = "input_tensor";
    static constexpr const char* OUTPUT_NAME = "output_tensor";
    
    // Input shape: [batch, channels, height, width]
    static constexpr int64_t BATCH_SIZE = 1;
    static constexpr int64_t CHANNELS = 3;
    static constexpr int64_t HEIGHT = 224;
    static constexpr int64_t WIDTH = 224;
    static constexpr int NUM_CLASSES = 1000;
};

} // namespace inference
