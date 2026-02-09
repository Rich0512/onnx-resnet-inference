/**
 * @file inference.cpp
 * @brief ONNX Runtime inference engine implementation with GPU support
 */

#include "inference.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <stdexcept>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace inference {

Engine::Engine(const std::string& model_path, int num_threads, bool use_gpu, bool use_tensorrt)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ResNetInference")
    , use_gpu_(false)
    , use_tensorrt_(false) {
    
    session_options_.SetIntraOpNumThreads(num_threads);
    
    // Configure GPU if requested
    #ifdef USE_CUDA
    if (use_gpu || use_tensorrt) {  // TensorRT also requires CUDA implicitly usually, or just GPU flag
        try {
            if (use_tensorrt) {
                // Settings for TensorRT
                // Note: OrtTensorRTProviderOptionsV2 is available in newer versions, 
                // but we use AppendExecutionProvider_TensorRT which takes OrtTensorRTProviderOptions usually
                // or just standard options. 
                // For simplicity we try to append it directly if available in this build.
                // The C++ API helper for TRT might need specific headers or just use the Option struct.
                // Let's assume standard appending works similarly to CUDA.
                OrtTensorRTProviderOptions trt_options{};
                trt_options.device_id = 0;
                trt_options.trt_max_partition_iterations = 1000;
                trt_options.trt_min_subgraph_size = 1;
                trt_options.trt_fp16_enable = 1;
                session_options_.AppendExecutionProvider_TensorRT(trt_options);
                use_tensorrt_ = true;
                std::cout << "    [GPU mode: TensorRT enabled]\n";
                // TensorRT often benefits from CUDA provider as fallback
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                use_gpu_ = true; // Mark GPU as used
            } else {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options_.AppendExecutionProvider_CUDA(cuda_options);
                use_gpu_ = true;
                std::cout << "    [GPU mode: CUDA enabled]\n";
            }
        } catch (const Ort::Exception& e) {
            std::cerr << "    [GPU mode: Failed - " << e.what() << "]\n";
            std::cerr << "    [Falling back to CPU]\n";
            use_gpu_ = false;
            use_tensorrt_ = false;
        }
    }
    #else
    if (use_gpu || use_tensorrt) {
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
    std::cout << "    [INFO] Model path: " << model_path << ", Threads: " << num_threads 
              << ", GPU: " << (use_gpu ? "true" : "false") 
              << ", TensorRT: " << (use_tensorrt ? "true" : "false")
              << ", Load time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    load_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::vector<float> Engine::run(const float* input_data, size_t input_size) {
    // Check input size
    if (input_size != BATCH_SIZE * CHANNELS * HEIGHT * WIDTH) {
        throw std::runtime_error("Input data size mismatch");
    }

    // Create input tensor
    // Crucial: Use Ort::MemoryInfo::CreateCpu even for GPU if input is on CPU/Pinned Memory
    // ONNX Runtime will handle the copy to GPU automatically.
    // However, if we allocate input_data with cudaHostAlloc (pinned), 
    // the copy will be much faster (DMA).
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
        
    // Create tensor from raw pointer
    // Note: We need const_cast because CreateTensor takes non-const pointer, 
    // but we promise not to modify it if we only run inference.
    // Actually, CreateTensor might modify it? No, usually inputs are read-only. 
    // But API requires float*.
    std::vector<int64_t> input_node_dims = {BATCH_SIZE, CHANNELS, HEIGHT, WIDTH}; // Define input_node_dims locally
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(input_data), 
        input_size, 
        input_node_dims.data(), 
        input_node_dims.size());
    
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

#ifdef USE_CUDA
std::vector<float> Engine::run_async(const float* input_data, size_t input_size, cudaStream_t stream) {
    if (input_size != BATCH_SIZE * CHANNELS * HEIGHT * WIDTH) {
        throw std::runtime_error("Input data size mismatch");
    }

    // 1. Create RunOptions and set stream
    Ort::RunOptions run_options;
    // Note: The key "gpu_compute_stream" expects the address of the cudaStream_t cast to void* then to integer string?
    // According to ORT docs/issues, for CUDA provider it's often passed via AddConfigEntry.
    // However, the exact key might be "gpu_compute_stream" or similar.
    // Let's use the standard approach: passing the pointer address as string.
    
    // Actually, newer ORT versions support OrtCUDAProviderOptions options;
    // but per-run stream is tricky.
    // Detailed check: ORT C++ API doesn't have direct "SetStream".
    // It relies on "gpu_compute_stream" config entry.
    // Value should be the pointer address of cudaStream_t.
    std::string stream_ptr_str = std::to_string((unsigned long long)stream);
    run_options.AddConfigEntry("gpu_compute_stream", stream_ptr_str.c_str());

    // 2. Wrap input data
    // Important: input_data MUST be kept alive until Run completes if it's async?
    // No, Run() is blocking unless we use true async API (RunAsync), which C++ API might not fully expose easily.
    // Wait, if Run() is blocking, then "gpu_compute_stream" just simply queues the kernels to that stream.
    // It returns when kernels are queued (maybe?) or when they finish?
    // Standard ORT Run is blocking on CPU until inference finishes.
    // So "run_async" here actually means "Run on a specific stream", but the call itself might still block CPU 
    // waiting for results if we immediately access output tensor.
    
    // BUT! if we use Pinned Memory for output, we might be able to return before GPU finishes?
    // Actually, getting the output tensor data usually triggers synchronization.
    
    // Let's implement it and see. The goal is to allow *other* streams to run concurrently.
    
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_node_dims = {BATCH_SIZE, CHANNELS, HEIGHT, WIDTH};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(input_data), 
        input_size, 
        input_node_dims.data(), 
        input_node_dims.size());

    const char* input_names[] = {INPUT_NAME};
    const char* output_names[] = {OUTPUT_NAME};

    auto outputs = session_->Run(
        run_options,
        input_names, &input_tensor, 1,
        output_names, 1
    );
    
    // Extract output
    // This calls implicit synchronization because we access the data on CPU.
    float* output_data = outputs[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + NUM_CLASSES);
}
#endif


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
