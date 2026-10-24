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
    explicit TrtEngine(const std::string& engine_path, int max_batch_size = 32);
    ~TrtEngine();

    // Disable copy
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Run end-to-end inference on cv::Mat (CPU Fused Preprocessor)
    std::vector<Prediction> predict(const cv::Mat& image, int top_k = 5);

    // Run end-to-end inference on cv::Mat (GPU CUDA Kernel Preprocessor)
    std::vector<Prediction> predict_gpu_preprocess(const cv::Mat& image, int top_k = 5);

    // ========================================================================
    // Station 8: Dynamic Batching API
    // ========================================================================
    // Predict a batch of N images using GPU batched preprocessing and dynamic TensorRT
    std::vector<std::vector<Prediction>> predict_batch(const std::vector<cv::Mat>& images, int top_k = 5);

    // Benchmark a specific batch size (e.g. 1, 2, 4, 8, 16, 32)
    double benchmark_batch(int batch_size, const cv::Mat& sample_image, int iterations = 100);

    // Full Saturation Curve Benchmark across multiple batch sizes
    void benchmark_saturation(const cv::Mat& sample_image, const std::vector<int>& batch_sizes = {1, 2, 4, 8, 16, 32}, int iterations = 100);

    // Benchmark mode: runs only the GPU inference portion
    double benchmark_inference(int iterations = 100);

    // Benchmark GPU Preprocessing Kernel alone
    double benchmark_gpu_preprocess(const cv::Mat& raw_image, int iterations = 100);

    // Benchmark full End-to-End GPU Pipeline (CUDA Preprocess Kernel + TRT enqueueV3 + D2H)
    double benchmark_end_to_end_gpu(const cv::Mat& raw_image, int iterations = 100);

    // ========================================================================
    // Station 7: CUDA Graphs API
    // ========================================================================
    // 1. Capture the entire pipeline (Preprocess -> TRT enqueueV3 -> D2H) into a CUDA Graph
    void capture_cuda_graph(const cv::Mat& sample_image);

    // 2. Predict using the captured CUDA Graph replay (Zero CPU launch overhead)
    std::vector<Prediction> predict_cuda_graph(const cv::Mat& image, int top_k = 5);

    // 3. Benchmark CUDA Graph execution alone (measuring hardware replay latency)
    double benchmark_cuda_graph(const cv::Mat& sample_image, int iterations = 100);

    // Multi-threaded Multi-Stream Benchmark Mode
    double benchmark_multithreaded(int total_iterations = 100, int num_streams = 2);

    // Getters
    double get_load_time_ms() const { return load_time_ms_; }
    double get_last_preprocess_time_ms() const { return last_preprocess_time_ms_; }
    double get_last_gpu_preprocess_time_ms() const { return last_gpu_preprocess_time_ms_; }
    double get_last_inference_time_ms() const { return last_inference_time_ms_; }
    double get_last_graph_time_ms() const { return last_graph_time_ms_; }
    bool is_graph_captured() const { return graph_captured_; }

private:
    void init_cuda();
    void cleanup_cuda();

    std::string engine_path_;
    double load_time_ms_ = 0.0;
    double last_preprocess_time_ms_ = 0.0;
    double last_gpu_preprocess_time_ms_ = 0.0;
    double last_inference_time_ms_ = 0.0;
    double last_graph_time_ms_ = 0.0;

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Device memory pointers
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    void* d_raw_bgr_ = nullptr;
    size_t allocated_raw_bgr_size_ = 0;
    
    // Batch memory pointers
    int max_batch_size_ = 32;
    void* d_raw_bgr_batch_ = nullptr;
    size_t allocated_raw_bgr_batch_size_ = 0;

    cudaStream_t stream_ = nullptr;

    // CUDA Graph state
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    bool graph_captured_ = false;

    // Host pinned/aligned buffers
    preprocessor::DataVector h_input_buffer_;
    std::vector<float> h_output_buffer_;
    std::vector<float> h_output_batch_buffer_;
};

} // namespace trt_native
