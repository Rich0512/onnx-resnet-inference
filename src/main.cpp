/**
 * @file main.cpp
 * @brief ResNet Inference Engine - Entry Point
 * 
 * A high-performance image classification system using C++ and ONNX Runtime.
 */

#include <iostream>
#include <filesystem>
#include <chrono>
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
    bool verbose = false;
};

void print_help(const std::string& program) {
    std::cout << "\nUsage: " << program << " [OPTIONS] <image_path>\n\n"
              << "Options:\n"
              << "  -h, --help         Show this help message\n"
              << "  -m, --model PATH   Path to ONNX model (default: resnet18.onnx)\n"
              << "  -t, --top N        Show top N predictions (default: 5)\n"
              << "  -v, --verbose      Show detailed output\n"
              << "\nExamples:\n"
              << "  " << program << " cat.jpg\n"
              << "  " << program << " --top 3 dog.jpg\n"
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
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
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
        inference::Engine engine(config.model_path);
        std::cout << "    Model loaded in " << engine.get_load_time_ms() << " ms\n";
        
        // Preprocess image
        std::cout << "\n[4] Preprocessing image...\n";
        auto start_prep = std::chrono::high_resolution_clock::now();
        auto input_data = preprocessor::preprocess(img);
        auto end_prep = std::chrono::high_resolution_clock::now();
        auto prep_time = std::chrono::duration_cast<std::chrono::microseconds>(end_prep - start_prep).count();
        std::cout << "    Done in " << prep_time / 1000.0 << " ms\n";
        
        // Run inference (with warm-up)
        std::cout << "\n[5] Running inference...\n";
        engine.run(input_data);  // Warm-up
        
        const int NUM_RUNS = 10;
        auto start_infer = std::chrono::high_resolution_clock::now();
        std::vector<float> scores;
        for (int i = 0; i < NUM_RUNS; i++) {
            scores = engine.run(input_data);
        }
        auto end_infer = std::chrono::high_resolution_clock::now();
        auto infer_time = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer).count();
        double avg_time = infer_time / 1000.0 / NUM_RUNS;
        
        std::cout << "    Inference: " << avg_time << " ms (avg of " << NUM_RUNS << " runs)\n"
                  << "    Throughput: " << 1000.0 / avg_time << " FPS\n";
        
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