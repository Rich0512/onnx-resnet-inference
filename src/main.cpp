/**
 * @file main.cpp
 * @brief ResNet Inference Engine - Entry Point
 * 
 * A high-performance image classification system using C++ and ONNX Runtime.
 * Supports both CPU and GPU (CUDA) acceleration.
 */

#include <iostream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include "inference.h"
#include "preprocessor.h"
#include "imagenet_labels.h"
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ============================================================================
// Command Line Interface
// ============================================================================

struct Config {
    std::string image_path = "test_cat.jpg";
    std::string model_path = "resnet18.onnx";
    int top_k = 5;
    bool use_gpu = false;
    bool use_tensorrt = false;
    bool verbose = false;
    bool benchmark = false;
};

void print_help(const std::string& program) {
    std::cout << "\nUsage: " << program << " [OPTIONS] <image_path>\n\n"
              << "Options:\n"
              << "  -h, --help         Show this help message\n"
              << "  -m, --model PATH   Path to ONNX model (default: resnet18.onnx)\n"
              << "  -t, --top N        Show top N predictions (default: 5)\n"
              << "  -g, --gpu          Enable GPU acceleration (CUDA)\n"
              << "  --tensorrt         Enable TensorRT optimization (requires GPU)\n"
              << "  -b, --benchmark    Run benchmark mode (100 iterations)\n"
              << "  -v, --verbose      Show detailed output\n"
              << "\nExamples:\n"
              << "  " << program << " cat.jpg\n"
              << "  " << program << " --gpu cat.jpg\n"
              << "  " << program << " --tensorrt cat.jpg\n"
              << "  " << program << " -m custom.onnx image.jpg\n\n";
}

Config parse_args(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            std::exit(0);
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if ((arg == "-t" || arg == "--top") && i + 1 < argc) {
            config.top_k = std::stoi(argv[++i]);
        } else if (arg == "-g" || arg == "--gpu") {
            config.use_gpu = true;
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "-b" || arg == "--benchmark") {
            config.benchmark = true;
        } else if (arg == "--tensorrt") {
            config.use_tensorrt = true;
            config.use_gpu = true;  // TensorRT requires GPU
        } else if (arg[0] != '-') {
            config.image_path = arg;
        }
    }
    
    return config;
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "\n========================================\n"
              << "   ResNet Inference Engine (C++)\n"
              << "========================================\n\n";
    
    try {
        // Parse arguments
        Config config = parse_args(argc, argv);
        
        // Validate files
        std::cout << "[1] Validating files...\n";
        if (!fs::exists(config.image_path)) {
            throw std::runtime_error("Image not found: " + config.image_path);
        }
        if (!fs::exists(config.model_path)) {
            throw std::runtime_error("Model not found: " + config.model_path);
        }
        std::cout << "    Image: " << config.image_path << "\n"
                  << "    Model: " << config.model_path << "\n";
        
        // Load image
        std::cout << "\n[2] Loading image...\n";
        cv::Mat img = cv::imread(config.image_path);
        if (img.empty()) {
            throw std::runtime_error("Failed to decode: " + config.image_path);
        }
        std::cout << "    Size: " << img.cols << "x" << img.rows << " pixels\n";
        
        // Initialize inference engine
        std::cout << "\n[3] Initializing ONNX Runtime...\n";
        inference::Engine engine(config.model_path, 4, config.use_gpu, config.use_tensorrt);
        std::cout << "    Model loaded in " << engine.get_load_time_ms() << " ms\n";
        
        // Create reusable buffer for Pinned Memory
        preprocessor::DataVector input_buffer;

        // Preprocess image (first time)
        std::cout << "\n[4] Preprocessing image...\n";
        auto start_prep = std::chrono::high_resolution_clock::now();
        preprocessor::preprocess_image(img, input_buffer, 224, 224);
        auto end_prep = std::chrono::high_resolution_clock::now();
        auto prep_time = std::chrono::duration_cast<std::chrono::microseconds>(end_prep - start_prep).count();
        std::cout << "    Done in " << prep_time / 1000.0 << " ms\n";
        
        // Run inference (with warm-up)
        std::cout << "\n[5] Running inference...\n";
        engine.run(input_buffer.data(), input_buffer.size());  // Warm-up
        
        const int NUM_RUNS = config.benchmark ? 100 : 10;
        std::vector<double> times;
        std::vector<float> scores;
        
        for (int i = 0; i < NUM_RUNS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            
            // In a real application, we might preprocess different images here.
            // For benchmark, we reuse the same preprocessed data.
            // If we wanted to benchmark full pipeline:
            // preprocessor::preprocess_image(img, input_buffer, 224, 224);

            scores = engine.run(input_buffer.data(), input_buffer.size());
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            times.push_back(ms);
        }
        
        double min_time = *std::min_element(times.begin(), times.end());
        double max_time = *std::max_element(times.begin(), times.end());
        double avg_time = std::accumulate(times.begin(), times.end(), 0.0) / NUM_RUNS;
        
        if (config.benchmark) {
            std::cout << "    [Benchmark Mode]\n"
                      << "    Runs: " << NUM_RUNS << "\n"
                      << "    Min:  " << min_time << " ms\n"
                      << "    Max:  " << max_time << " ms\n"
                      << "    Avg:  " << avg_time << " ms\n";
        }
        
#ifdef USE_CUDA
        // Multi-threaded Benchmark with CUDA Streams
        if (config.use_gpu && config.benchmark) {
            std::cout << "\n[Benchmark] Running Multi-threaded Async Inference (2 Streams)...\n";
            
            cudaStream_t stream1, stream2;
            cudaStreamCreate(&stream1);
            cudaStreamCreate(&stream2);
            
            int runs_per_thread = NUM_RUNS / 2;
            
            auto worker = [&](cudaStream_t stream) {
                // Each thread needs its own buffer to be safe
                preprocessor::DataVector local_buffer = input_buffer; 
                std::vector<double> local_times;
                
                for(int i=0; i<runs_per_thread; ++i) {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    engine.run_async(local_buffer.data(), local_buffer.size(), stream);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    local_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0);
                }
                return local_times;
            };
            
            auto start_multi = std::chrono::high_resolution_clock::now();
            
            auto f1 = std::async(std::launch::async, worker, stream1);
            auto f2 = std::async(std::launch::async, worker, stream2);
            
            auto t1 = f1.get();
            auto t2 = f2.get();
            
            auto end_multi = std::chrono::high_resolution_clock::now();
            double total_time_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end_multi - start_multi).count());
            
            std::cout << "    Total time for " << NUM_RUNS << " runs (2 threads): " << total_time_ms << " ms\n";
            std::cout << "    Throughput: " << (1000.0 * NUM_RUNS) / total_time_ms << " FPS\n";
            
            // Cleanup matches
            times.clear();
            times.insert(times.end(), t1.begin(), t1.end());
            times.insert(times.end(), t2.begin(), t2.end());
            
            cudaStreamDestroy(stream1);
            cudaStreamDestroy(stream2);
            
            // Recalculate avg for display
            avg_time = total_time_ms / NUM_RUNS; // Effective avg time per image
        } else {
             // Standard sequential benchmark
             std::cout << "    Inference: " << avg_time << " ms (avg of " << NUM_RUNS << " runs)\n"
                       << "    Throughput: " << 1000.0 / avg_time << " FPS\n";
        }
#else
        std::cout << "    Inference: " << avg_time << " ms (avg of " << NUM_RUNS << " runs)\n"
                  << "    Throughput: " << 1000.0 / avg_time << " FPS\n";
#endif
        
        // Get results
        std::cout << "\n[6] Results:\n"
                  << "    Top-" << config.top_k << " Predictions:\n";
        auto results = inference::Engine::get_top_k(scores, config.top_k);
        for (size_t i = 0; i < results.size(); i++) {
            std::cout << "      " << (i + 1) << ". " 
                      << imagenet::get_label(results[i].class_id)
                      << " (score: " << results[i].score << ")\n";
        }
        
        // Summary
        std::cout << "\n========================================\n"
                  << "   Performance Summary\n"
                  << "========================================\n"
                  << "  Mode:        " << (engine.is_using_tensorrt() ? "GPU (TensorRT)" : (engine.is_using_gpu() ? "GPU (CUDA)" : "CPU")) << "\n"
                  << "  Model load:  " << engine.get_load_time_ms() << " ms\n"
                  << "  Preprocess:  " << prep_time / 1000.0 << " ms\n"
                  << "  Inference:   " << avg_time << " ms\n"
                  << "========================================\n\n";
        
        return 0;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "\n[ERROR] ONNX Runtime: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}