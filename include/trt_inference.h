#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include "preprocessor.h"

namespace trt_native {

struct Prediction {
    int class_id;
    float confidence;
    std::string label;
};

class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path);
    ~TrtEngine();

    // Disable copy
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Run end-to-end inference on cv::Mat
    std::vector<Prediction> predict(const cv::Mat& image, int top_k = 5);

    // Benchmark mode: runs only the GPU inference portion
    double benchmark_inference(int iterations = 100);

    // Multi-threaded Multi-Stream Benchmark Mode
    double benchmark_multithreaded(int total_iterations = 100, int num_streams = 2);

    // Getters
    double get_load_time_ms() const { return load_time_ms_; }
    double get_last_preprocess_time_ms() const { return last_preprocess_time_ms_; }
    double get_last_inference_time_ms() const { return last_inference_time_ms_; }

private:
    void init_cuda();
    void cleanup_cuda();

    std::string engine_path_;
    double load_time_ms_ = 0.0;
    double last_preprocess_time_ms_ = 0.0;
    double last_inference_time_ms_ = 0.0;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Device memory pointers
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    cudaStream_t stream_ = nullptr;

    // Host pinned/aligned buffers
    preprocessor::DataVector h_input_buffer_;
    std::vector<float> h_output_buffer_;
};

} // namespace trt_native
