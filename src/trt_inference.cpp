#include "trt_inference.h"
#include "imagenet_labels.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>

namespace trt_native {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress verbose info messages
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "[TRT-ERROR] " << msg << std::endl;
        } else if (severity == Severity::kWARNING) {
            std::cout << "[TRT-WARN] " << msg << std::endl;
        }
    }
};

static TrtLogger gLogger;

TrtEngine::TrtEngine(const std::string& engine_path)
    : engine_path_(engine_path) {
    
    auto t0 = std::chrono::high_resolution_clock::now();

    // 1. Read serialized engine file into memory buffer
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open TensorRT engine file: " + engine_path);
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    if (!file.read(engine_data.data(), size)) {
        throw std::runtime_error("Failed to read TensorRT engine file: " + engine_path);
    }
    file.close();

    // 2. Create Runtime, Engine, ExecutionContext
    runtime_.reset(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT Runtime");
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize CUDA Engine");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create Execution Context");
    }

    // 3. Allocate GPU Memory and stream
    init_cuda();

    auto t1 = std::chrono::high_resolution_clock::now();
    load_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

TrtEngine::~TrtEngine() {
    cleanup_cuda();
}

void TrtEngine::init_cuda() {
    // Input: 1 x 3 x 224 x 224 = 150,528 floats
    // Output: 1 x 1000 = 1,000 floats
    size_t input_size = 1 * 3 * 224 * 224 * sizeof(float);
    size_t output_size = 1000 * sizeof(float);

    h_input_buffer_.resize(1 * 3 * 224 * 224);
    h_output_buffer_.resize(1000);

    cudaError_t err = cudaMalloc(&d_input_, input_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for d_input: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_output_, output_size);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for d_output: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaStreamCreate failed: " + std::string(cudaGetErrorString(err)));
    }

    // Modern TensorRT 10: Bind device pointers directly via setTensorAddress
    context_->setTensorAddress("input_tensor", d_input_);
    context_->setTensorAddress("output_tensor", d_output_);
}

void TrtEngine::cleanup_cuda() {
    if (d_input_) {
        cudaFree(d_input_);
        d_input_ = nullptr;
    }
    if (d_output_) {
        cudaFree(d_output_);
        d_output_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

std::vector<Prediction> TrtEngine::predict(const cv::Mat& image, int top_k) {
    // 1. Station 3 Fused CPU Preprocessor (0.22 ms)
    auto t_pre0 = std::chrono::high_resolution_clock::now();
    preprocessor::preprocess_image(image, h_input_buffer_);
    auto t_pre1 = std::chrono::high_resolution_clock::now();
    last_preprocess_time_ms_ = std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count();

    // 2. Async H2D Transfer
    size_t input_size = 1 * 3 * 224 * 224 * sizeof(float);
    size_t output_size = 1000 * sizeof(float);
    cudaMemcpyAsync(d_input_, h_input_buffer_.data(), input_size, cudaMemcpyHostToDevice, stream_);

    // 3. Enqueue Inference Kernel on Stream
    auto t_inf0 = std::chrono::high_resolution_clock::now();
    context_->enqueueV3(stream_);

    // 4. Async D2H Transfer
    cudaMemcpyAsync(h_output_buffer_.data(), d_output_, output_size, cudaMemcpyDeviceToHost, stream_);

    // 5. Synchronize CUDA Stream
    cudaStreamSynchronize(stream_);
    auto t_inf1 = std::chrono::high_resolution_clock::now();
    last_inference_time_ms_ = std::chrono::duration<double, std::milli>(t_inf1 - t_inf0).count();

    // 6. Top-K postprocessing
    std::vector<int> indices(1000);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
        [this](int a, int b) {
            return h_output_buffer_[a] > h_output_buffer_[b];
        });

    std::vector<Prediction> predictions;
    for (int i = 0; i < top_k; ++i) {
        int idx = indices[i];
        std::string label = "unknown";
        auto it = imagenet::LABELS.find(idx);
        if (it != imagenet::LABELS.end()) {
            label = it->second;
        }
        predictions.push_back({
            idx,
            h_output_buffer_[idx],
            label
        });
    }

    return predictions;
}

double TrtEngine::benchmark_inference(int iterations) {
    // Warmup
    for (int i = 0; i < 10; ++i) {
        context_->enqueueV3(stream_);
    }
    cudaStreamSynchronize(stream_);

    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        context_->enqueueV3(stream_);
        cudaStreamSynchronize(stream_);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double total = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = total / iterations;
    double min_lat = *std::min_element(latencies.begin(), latencies.end());
    double max_lat = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "    [Native TensorRT Benchmark Mode]\n"
              << "    Runs: " << iterations << "\n"
              << "    Min:  " << min_lat << " ms\n"
              << "    Max:  " << max_lat << " ms\n"
              << "    Avg:  " << avg << " ms\n"
              << "    FPS:  " << (1000.0 / avg) << " FPS\n";

    return avg;
}

double TrtEngine::benchmark_multithreaded(int total_iterations, int num_streams) {
    if (num_streams <= 0) num_streams = 2;

    struct WorkerContext {
        std::unique_ptr<nvinfer1::IExecutionContext> context;
        void* d_input = nullptr;
        void* d_output = nullptr;
        cudaStream_t stream = nullptr;
        preprocessor::DataVector h_input;
        std::vector<float> h_output;
    };

    size_t input_size = 1 * 3 * 224 * 224 * sizeof(float);
    size_t output_size = 1000 * sizeof(float);

    std::vector<WorkerContext> workers(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        workers[i].context.reset(engine_->createExecutionContext());
        cudaMalloc(&workers[i].d_input, input_size);
        cudaMalloc(&workers[i].d_output, output_size);
        cudaStreamCreate(&workers[i].stream);
        workers[i].context->setTensorAddress("input_tensor", workers[i].d_input);
        workers[i].context->setTensorAddress("output_tensor", workers[i].d_output);
        workers[i].h_input = h_input_buffer_;
        workers[i].h_output.resize(1000);
    }

    // Warmup all streams
    for (int i = 0; i < num_streams; ++i) {
        workers[i].context->enqueueV3(workers[i].stream);
        cudaStreamSynchronize(workers[i].stream);
    }

    int runs_per_worker = total_iterations / num_streams;

    auto worker_fn = [&](int worker_idx) {
        auto& w = workers[worker_idx];
        for (int r = 0; r < runs_per_worker; ++r) {
            w.context->enqueueV3(w.stream);
        }
        cudaStreamSynchronize(w.stream);
    };

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::future<void>> futures;
    for (int i = 0; i < num_streams; ++i) {
        futures.push_back(std::async(std::launch::async, worker_fn, i));
    }

    for (auto& f : futures) {
        f.get();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double fps = (1000.0 * total_iterations) / total_ms;

    std::cout << "\n    [Native TensorRT Multi-Threaded Async Benchmark (" << num_streams << " Streams)]\n"
              << "    Total time for " << total_iterations << " runs: " << total_ms << " ms\n"
              << "    Throughput: " << fps << " FPS\n";

    // Cleanup worker GPU resources
    for (int i = 0; i < num_streams; ++i) {
        if (workers[i].d_input) cudaFree(workers[i].d_input);
        if (workers[i].d_output) cudaFree(workers[i].d_output);
        if (workers[i].stream) cudaStreamDestroy(workers[i].stream);
    }

    return fps;
}

} // namespace trt_native
