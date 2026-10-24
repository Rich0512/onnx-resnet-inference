#include "trt_inference.h"
#include "imagenet_labels.h"
#include "cuda_preprocessor.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include <iomanip>

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

TrtEngine::TrtEngine(const std::string& engine_path, int max_batch_size)
    : engine_path_(engine_path), max_batch_size_(max_batch_size) {
    
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
    // Check if input tensor has dynamic batch dimension
    auto in_dims = engine_->getTensorShape("input_tensor");
    if (in_dims.d[0] == -1) {
        // Set default batch size = 1 for dynamic profiles
        context_->setInputShape("input_tensor", nvinfer1::Dims4(1, 3, 224, 224));
    }

    // Allocate memory for up to max_batch_size_
    size_t input_size = static_cast<size_t>(max_batch_size_) * 3 * 224 * 224 * sizeof(float);
    size_t output_size = static_cast<size_t>(max_batch_size_) * 1000 * sizeof(float);

    h_input_buffer_.resize(1 * 3 * 224 * 224);
    h_output_buffer_.resize(1000);
    h_output_batch_buffer_.resize(max_batch_size_ * 1000);

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
    if (d_raw_bgr_) {
        cudaFree(d_raw_bgr_);
        d_raw_bgr_ = nullptr;
        allocated_raw_bgr_size_ = 0;
    }
    if (d_raw_bgr_batch_) {
        cudaFree(d_raw_bgr_batch_);
        d_raw_bgr_batch_ = nullptr;
        allocated_raw_bgr_batch_size_ = 0;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    graph_captured_ = false;
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

std::vector<Prediction> TrtEngine::predict_gpu_preprocess(const cv::Mat& image, int top_k) {
    if (image.empty()) {
        throw std::runtime_error("Empty image passed to predict_gpu_preprocess");
    }

    // 1. Ensure GPU buffer for raw BGR image is allocated
    size_t raw_size = static_cast<size_t>(image.cols) * image.rows * 3;
    if (raw_size > allocated_raw_bgr_size_) {
        if (d_raw_bgr_) cudaFree(d_raw_bgr_);
        cudaMalloc(&d_raw_bgr_, raw_size);
        allocated_raw_bgr_size_ = raw_size;
    }

    // 2. Direct copy of raw BGR image from Host to Device
    cudaMemcpyAsync(d_raw_bgr_, image.data, raw_size, cudaMemcpyHostToDevice, stream_);

    // 3. Fused GPU CUDA Preprocessor (Bilinear Resize + BGR->RGB + Normalization + NCHW)
    cudaEvent_t ev_start, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);

    cudaEventRecord(ev_start, stream_);
    cuda_preprocess::launch_cuda_preprocess(
        static_cast<const uint8_t*>(d_raw_bgr_), image.cols, image.rows,
        static_cast<float*>(d_input_), 224, 224,
        stream_
    );
    cudaEventRecord(ev_end, stream_);

    // 4. Enqueue Inference Kernel on Stream
    auto t_inf0 = std::chrono::high_resolution_clock::now();
    context_->enqueueV3(stream_);

    // 5. Async D2H Transfer
    size_t output_size = 1000 * sizeof(float);
    cudaMemcpyAsync(h_output_buffer_.data(), d_output_, output_size, cudaMemcpyDeviceToHost, stream_);

    // 6. Synchronize CUDA Stream
    cudaStreamSynchronize(stream_);
    auto t_inf1 = std::chrono::high_resolution_clock::now();
    last_inference_time_ms_ = std::chrono::duration<double, std::milli>(t_inf1 - t_inf0).count();

    float gpu_pre_ms = 0.0f;
    cudaEventElapsedTime(&gpu_pre_ms, ev_start, ev_end);
    last_gpu_preprocess_time_ms_ = static_cast<double>(gpu_pre_ms);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);

    // 7. Top-K postprocessing
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

double TrtEngine::benchmark_gpu_preprocess(const cv::Mat& raw_image, int iterations) {
    if (raw_image.empty()) return 0.0;

    size_t raw_size = static_cast<size_t>(raw_image.cols) * raw_image.rows * 3;
    if (raw_size > allocated_raw_bgr_size_) {
        if (d_raw_bgr_) cudaFree(d_raw_bgr_);
        cudaMalloc(&d_raw_bgr_, raw_size);
        allocated_raw_bgr_size_ = raw_size;
    }
    cudaMemcpy(d_raw_bgr_, raw_image.data, raw_size, cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        cuda_preprocess::launch_cuda_preprocess(
            static_cast<const uint8_t*>(d_raw_bgr_), raw_image.cols, raw_image.rows,
            static_cast<float*>(d_input_), 224, 224, stream_
        );
    }
    cudaStreamSynchronize(stream_);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream_);
    for (int i = 0; i < iterations; ++i) {
        cuda_preprocess::launch_cuda_preprocess(
            static_cast<const uint8_t*>(d_raw_bgr_), raw_image.cols, raw_image.rows,
            static_cast<float*>(d_input_), 224, 224, stream_
        );
    }
    cudaEventRecord(stop, stream_);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = static_cast<double>(total_ms) / iterations;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return avg_ms;
}

double TrtEngine::benchmark_end_to_end_gpu(const cv::Mat& raw_image, int iterations) {
    if (raw_image.empty()) return 0.0;

    size_t raw_size = static_cast<size_t>(raw_image.cols) * raw_image.rows * 3;
    if (raw_size > allocated_raw_bgr_size_) {
        if (d_raw_bgr_) cudaFree(d_raw_bgr_);
        cudaMalloc(&d_raw_bgr_, raw_size);
        allocated_raw_bgr_size_ = raw_size;
    }
    cudaMemcpy(d_raw_bgr_, raw_image.data, raw_size, cudaMemcpyHostToDevice);

    size_t output_size = 1000 * sizeof(float);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        cuda_preprocess::launch_cuda_preprocess(
            static_cast<const uint8_t*>(d_raw_bgr_), raw_image.cols, raw_image.rows,
            static_cast<float*>(d_input_), 224, 224, stream_
        );
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(h_output_buffer_.data(), d_output_, output_size, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cuda_preprocess::launch_cuda_preprocess(
            static_cast<const uint8_t*>(d_raw_bgr_), raw_image.cols, raw_image.rows,
            static_cast<float*>(d_input_), 224, 224, stream_
        );
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(h_output_buffer_.data(), d_output_, output_size, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double total = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = total / iterations;
    double min_lat = *std::min_element(latencies.begin(), latencies.end());
    double max_lat = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "    [Station 6 Pure GPU Pipeline (CUDA Preprocess + TRT FP16 + D2H)]\n"
              << "    Runs: " << iterations << "\n"
              << "    Min:  " << min_lat << " ms\n"
              << "    Max:  " << max_lat << " ms\n"
              << "    Avg:  " << avg << " ms\n"
              << "    FPS:  " << (1000.0 / avg) << " FPS\n";

    return avg;
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

void TrtEngine::capture_cuda_graph(const cv::Mat& sample_image) {
    if (sample_image.empty()) {
        throw std::runtime_error("Cannot capture CUDA graph with an empty image");
    }

    size_t raw_size = static_cast<size_t>(sample_image.cols) * sample_image.rows * 3;
    if (raw_size > allocated_raw_bgr_size_) {
        if (d_raw_bgr_) cudaFree(d_raw_bgr_);
        cudaMalloc(&d_raw_bgr_, raw_size);
        allocated_raw_bgr_size_ = raw_size;
    }
    cudaMemcpy(d_raw_bgr_, sample_image.data, raw_size, cudaMemcpyHostToDevice);

    size_t output_size = 1000 * sizeof(float);

    // Warmup: TensorRT and CUDA runtimes need one iteration on the stream
    // to complete one-time driver state setups before stream capture.
    cuda_preprocess::launch_cuda_preprocess(
        static_cast<const uint8_t*>(d_raw_bgr_), sample_image.cols, sample_image.rows,
        static_cast<float*>(d_input_), 224, 224, stream_
    );
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(h_output_buffer_.data(), d_output_, output_size, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // If a graph was already captured previously, destroy it before recapturing
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }

    // 1. Begin Stream Capture
    cudaError_t err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaStreamBeginCapture failed: " + std::string(cudaGetErrorString(err)));
    }

    // 2. Enqueue the entire pipeline into the stream (Preprocess -> TRT enqueueV3 -> D2H)
    cuda_preprocess::launch_cuda_preprocess(
        static_cast<const uint8_t*>(d_raw_bgr_), sample_image.cols, sample_image.rows,
        static_cast<float*>(d_input_), 224, 224, stream_
    );
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(h_output_buffer_.data(), d_output_, output_size, cudaMemcpyDeviceToHost, stream_);

    // 3. End Stream Capture
    err = cudaStreamEndCapture(stream_, &graph_);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaStreamEndCapture failed: " + std::string(cudaGetErrorString(err)));
    }

    // 4. Instantiate the Executable Graph
    err = cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaGraphInstantiate failed: " + std::string(cudaGetErrorString(err)));
    }

    graph_captured_ = true;
    std::cout << "    [CUDA Graph] Successfully captured and instantiated execution graph!\n";
}

std::vector<Prediction> TrtEngine::predict_cuda_graph(const cv::Mat& image, int top_k) {
    if (!graph_captured_) {
        capture_cuda_graph(image);
    }

    // 1. Copy fresh raw image to pre-allocated device memory
    size_t raw_size = static_cast<size_t>(image.cols) * image.rows * 3;
    if (raw_size > allocated_raw_bgr_size_) {
        // If image size changed, need to reallocate and recapture
        capture_cuda_graph(image);
    }
    cudaMemcpyAsync(d_raw_bgr_, image.data, raw_size, cudaMemcpyHostToDevice, stream_);

    // 2. Launch the entire captured graph with a SINGLE call (Zero CPU launch overhead!)
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaGraphLaunch(graph_exec_, stream_);
    cudaStreamSynchronize(stream_);
    auto t1 = std::chrono::high_resolution_clock::now();

    last_graph_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 3. Top-k extraction
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

double TrtEngine::benchmark_cuda_graph(const cv::Mat& sample_image, int iterations) {
    if (!graph_captured_) {
        capture_cuda_graph(sample_image);
    }

    // Warmup graph replay
    for (int i = 0; i < 10; ++i) {
        cudaGraphLaunch(graph_exec_, stream_);
    }
    cudaStreamSynchronize(stream_);

    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(graph_exec_, stream_);
        cudaStreamSynchronize(stream_);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double total = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = total / iterations;
    double min_lat = *std::min_element(latencies.begin(), latencies.end());
    double max_lat = *std::max_element(latencies.begin(), latencies.end());

    std::cout << "    [Station 7 CUDA Graph Replay (Zero Launch Overhead: Pre + TRT + D2H)]\n"
              << "    Runs: " << iterations << "\n"
              << "    Min:  " << min_lat << " ms\n"
              << "    Max:  " << max_lat << " ms\n"
              << "    Avg:  " << avg << " ms\n"
              << "    FPS:  " << (1000.0 / avg) << " FPS\n";

    return avg;
}

// ============================================================================
// Station 8: Dynamic Batching Implementation
// ============================================================================

std::vector<std::vector<Prediction>> TrtEngine::predict_batch(const std::vector<cv::Mat>& images, int top_k) {
    if (images.empty()) return {};

    int batch_size = static_cast<int>(images.size());
    if (batch_size > max_batch_size_) {
        throw std::runtime_error("Batch size " + std::to_string(batch_size) + " exceeds max_batch_size " + std::to_string(max_batch_size_));
    }

    // 1. Prepare contiguous host BGR buffer and upload to device
    int src_w = images[0].cols;
    int src_h = images[0].rows;
    size_t single_raw_size = static_cast<size_t>(src_w) * src_h * 3;
    size_t total_raw_size = single_raw_size * batch_size;

    if (total_raw_size > allocated_raw_bgr_batch_size_) {
        if (d_raw_bgr_batch_) cudaFree(d_raw_bgr_batch_);
        cudaMalloc(&d_raw_bgr_batch_, total_raw_size);
        allocated_raw_bgr_batch_size_ = total_raw_size;
    }

    // Copy each image in the batch into contiguous device buffer
    for (int b = 0; b < batch_size; ++b) {
        if (images[b].cols != src_w || images[b].rows != src_h) {
            throw std::runtime_error("All images in a batch must have identical resolution");
        }
        uint8_t* dst_dev_ptr = static_cast<uint8_t*>(d_raw_bgr_batch_) + b * single_raw_size;
        cudaMemcpyAsync(dst_dev_ptr, images[b].data, single_raw_size, cudaMemcpyHostToDevice, stream_);
    }

    // 2. Launch Batched CUDA Preprocessing Kernel
    cuda_preprocess::launch_cuda_preprocess_batch(
        static_cast<const uint8_t*>(d_raw_bgr_batch_), src_w, src_h,
        static_cast<float*>(d_input_), 224, 224, batch_size, stream_
    );

    // 3. Set Dynamic Input Shape for TensorRT (if dynamic)
    auto in_dims = engine_->getTensorShape("input_tensor");
    if (in_dims.d[0] == -1) {
        context_->setInputShape("input_tensor", nvinfer1::Dims4(batch_size, 3, 224, 224));
    }

    // 4. Enqueue Batched TensorRT Inference
    context_->enqueueV3(stream_);

    // 5. Async D2H Transfer
    size_t total_output_size = static_cast<size_t>(batch_size) * 1000 * sizeof(float);
    cudaMemcpyAsync(h_output_batch_buffer_.data(), d_output_, total_output_size, cudaMemcpyDeviceToHost, stream_);

    // 6. Synchronize Stream
    cudaStreamSynchronize(stream_);

    // 7. Extract Top-K predictions for each batch item
    std::vector<std::vector<Prediction>> all_predictions(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        const float* cur_logits = h_output_batch_buffer_.data() + b * 1000;
        std::vector<int> indices(1000);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
            [cur_logits](int i1, int i2) {
                return cur_logits[i1] > cur_logits[i2];
            });

        for (int k = 0; k < top_k; ++k) {
            int idx = indices[k];
            std::string label = "unknown";
            auto it = imagenet::LABELS.find(idx);
            if (it != imagenet::LABELS.end()) {
                label = it->second;
            }
            all_predictions[b].push_back({
                idx,
                cur_logits[idx],
                label
            });
        }
    }

    return all_predictions;
}

double TrtEngine::benchmark_batch(int batch_size, const cv::Mat& sample_image, int iterations) {
    if (batch_size <= 0 || batch_size > max_batch_size_) {
        throw std::runtime_error("Invalid batch size: " + std::to_string(batch_size));
    }

    int src_w = sample_image.cols;
    int src_h = sample_image.rows;
    size_t single_raw_size = static_cast<size_t>(src_w) * src_h * 3;
    size_t total_raw_size = single_raw_size * batch_size;

    if (total_raw_size > allocated_raw_bgr_batch_size_) {
        if (d_raw_bgr_batch_) cudaFree(d_raw_bgr_batch_);
        cudaMalloc(&d_raw_bgr_batch_, total_raw_size);
        allocated_raw_bgr_batch_size_ = total_raw_size;
    }

    // Replicate sample image to device buffer
    for (int b = 0; b < batch_size; ++b) {
        uint8_t* dst_dev_ptr = static_cast<uint8_t*>(d_raw_bgr_batch_) + b * single_raw_size;
        cudaMemcpy(dst_dev_ptr, sample_image.data, single_raw_size, cudaMemcpyHostToDevice);
    }

    auto in_dims = engine_->getTensorShape("input_tensor");
    if (in_dims.d[0] == -1) {
        context_->setInputShape("input_tensor", nvinfer1::Dims4(batch_size, 3, 224, 224));
    }

    size_t total_output_size = static_cast<size_t>(batch_size) * 1000 * sizeof(float);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        cuda_preprocess::launch_cuda_preprocess_batch(
            static_cast<const uint8_t*>(d_raw_bgr_batch_), src_w, src_h,
            static_cast<float*>(d_input_), 224, 224, batch_size, stream_
        );
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(h_output_batch_buffer_.data(), d_output_, total_output_size, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
    }

    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cuda_preprocess::launch_cuda_preprocess_batch(
            static_cast<const uint8_t*>(d_raw_bgr_batch_), src_w, src_h,
            static_cast<float*>(d_input_), 224, 224, batch_size, stream_
        );
        context_->enqueueV3(stream_);
        cudaMemcpyAsync(h_output_batch_buffer_.data(), d_output_, total_output_size, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    double total = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg_batch_ms = total / iterations;
    return avg_batch_ms;
}

void TrtEngine::benchmark_saturation(const cv::Mat& sample_image, const std::vector<int>& batch_sizes, int iterations) {
    std::cout << "\n=========================================================================\n"
              << "   Station 8: Dynamic Batching Saturation Curve Benchmark\n"
              << "=========================================================================\n"
              << "  Batch Size | Batch Latency (ms) | Per-Image (ms) | Throughput (FPS) | Speedup\n"
              << " ------------+--------------------+----------------+------------------+---------\n";

    double baseline_fps = 0.0;
    for (int b : batch_sizes) {
        if (b > max_batch_size_) continue;
        double batch_ms = benchmark_batch(b, sample_image, iterations);
        double per_img_ms = batch_ms / b;
        double fps = (1000.0 * b) / batch_ms;
        if (b == 1) baseline_fps = fps;
        double speedup = (baseline_fps > 0.0) ? (fps / baseline_fps) : 1.0;

        std::cout << "  " << std::setw(10) << b << " | "
                  << std::setw(18) << std::fixed << std::setprecision(3) << batch_ms << " | "
                  << std::setw(14) << std::fixed << std::setprecision(4) << per_img_ms << " | "
                  << std::setw(16) << std::fixed << std::setprecision(1) << fps << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << speedup << "x\n";
    }
    std::cout << "=========================================================================\n\n";
}

} // namespace trt_native
