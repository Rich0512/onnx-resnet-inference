#include <iostream>
#include <filesystem>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "trt_inference.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string image_path = "test_cat.jpg";
    std::string engine_path = "models/resnet18_native_fp16.engine";
    bool run_benchmark = false;

    // Parse simple args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-b" || arg == "--benchmark") {
            run_benchmark = true;
        } else if (arg == "-e" || arg == "--engine") {
            if (i + 1 < argc) engine_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: main_trt [OPTIONS] <image_path>\n"
                      << "Options:\n"
                      << "  -e, --engine PATH   Path to TensorRT .engine (default: models/resnet18_native_fp16.engine)\n"
                      << "  -b, --benchmark     Run 100 benchmark iterations\n"
                      << "  -h, --help          Show help\n";
            return 0;
        } else if (arg[0] != '-') {
            image_path = arg;
        }
    }

    std::cout << "========================================\n"
              << "   Native TensorRT Engine (Option B - Pure C++)\n"
              << "========================================\n\n";

    // 1. Validation
    if (!fs::exists(image_path)) {
        // Also check Release directory
        if (fs::exists("Release/" + image_path)) {
            image_path = "Release/" + image_path;
        } else if (fs::exists("../Release/" + image_path)) {
            image_path = "../Release/" + image_path;
        }
    }

    // Auto-detect engine if default path not found
    if (!fs::exists(engine_path)) {
        std::vector<std::string> search_dirs = {"trt_cache", "../trt_cache", ".", "Release/trt_cache", "models", "../models"};
        for (const auto& dir : search_dirs) {
            if (fs::exists(dir)) {
                for (const auto& entry : fs::directory_iterator(dir)) {
                    if (entry.path().extension() == ".engine") {
                        engine_path = entry.path().string();
                        break;
                    }
                }
                if (fs::exists(engine_path) && fs::path(engine_path).extension() == ".engine") {
                    break;
                }
            }
        }
    }

    std::cout << "[1] Validating files...\n"
              << "    Image:  " << image_path << "\n"
              << "    Engine: " << engine_path << "\n\n";

    // 2. Load Image
    std::cout << "[2] Loading image...\n";
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "ERROR: Failed to load image: " << image_path << "\n";
        return 1;
    }
    std::cout << "    Size: " << image.cols << "x" << image.rows << " pixels\n\n";

    // 3. Initialize Native TensorRT Engine
    std::cout << "[3] Initializing Native TensorRT Engine...\n";
    std::unique_ptr<trt_native::TrtEngine> engine;
    try {
        engine = std::make_unique<trt_native::TrtEngine>(engine_path);
        std::cout << "    Engine loaded & deserialized in " << engine->get_load_time_ms() << " ms\n\n";
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    // 4. Inference
    std::cout << "[4] Running End-to-End Prediction (Station 3 Fused C++ Preprocessor)...\n";
    auto predictions = engine->predict(image, 5);
    std::cout << "    Preprocessing: " << engine->get_last_preprocess_time_ms() << " ms\n"
              << "    Inference:     " << engine->get_last_inference_time_ms() << " ms\n\n";

    // 5. Benchmark mode
    if (run_benchmark) {
        std::cout << "[5] Running Benchmark Mode (100 Iterations)...\n";
        engine->benchmark_inference(100);
        std::cout << "\n";
    }

    // 6. Results
    std::cout << "[6] Top-5 Predictions:\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "    " << (i + 1) << ". " << predictions[i].label
                  << " (score: " << predictions[i].confidence << ", id: " << predictions[i].class_id << ")\n";
    }

    std::cout << "\n========================================\n"
              << "   Performance Summary (Pure C++ Option B)\n"
              << "========================================\n"
              << "  Mode:        Native TensorRT 10 (Pure C++)\n"
              << "  Engine load: " << engine->get_load_time_ms() << " ms\n"
              << "  Preprocess:  " << engine->get_last_preprocess_time_ms() << " ms (Fused C++)\n"
              << "  Inference:   " << engine->get_last_inference_time_ms() << " ms\n"
              << "========================================\n";

    return 0;
}
