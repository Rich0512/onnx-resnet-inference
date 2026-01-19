/**
 * ONNX ResNet Inference Engine
 * 
 * A high-performance image classification system using C++ and ONNX Runtime.
 * 
 * Features:
 * - ResNet18 inference with ImageNet preprocessing
 * - Error handling with try-catch
 * - Command-line argument parsing
 * - ImageNet class name display
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

// ============================================================================
// ImageNet Class Names (Top categories for demo)
// ============================================================================
const std::vector<std::pair<int, std::string>> IMAGENET_CLASSES = {
    {281, "tabby cat"},
    {282, "tiger cat"},
    {285, "Egyptian cat"},
    {287, "lynx"},
    {288, "leopard"},
    {151, "Chihuahua"},
    {207, "golden retriever"},
    {208, "Labrador retriever"},
    {230, "Shetland sheepdog"},
    {235, "German shepherd"},
    {243, "bull mastiff"},
    {254, "husky"},
    {263, "Pembroke Welsh corgi"},
    {1, "goldfish"},
    {65, "sea snake"},
    {99, "goose"},
    {130, "flamingo"},
    {291, "lion"},
    {292, "tiger"},
    {340, "zebra"},
    {386, "African elephant"},
    {388, "giant panda"},
    {409, "gibbon"},
    {417, "balloon"},
    {508, "computer keyboard"},
    {531, "digital watch"},
    {609, "jeep"},
    {717, "pickup truck"},
    {779, "school bus"},
    {852, "tennis ball"},
    {920, "traffic light"},
    {950, "orange"},
    {954, "banana"}
};

/**
 * Get class name by class ID
 */
std::string get_class_name(int class_id) {
    for (const auto& [id, name] : IMAGENET_CLASSES) {
        if (id == class_id) {
            return name;
        }
    }
    return "unknown (class " + std::to_string(class_id) + ")";
}

/**
 * Print help message
 */
void print_help(const std::string& program_name) {
    std::cout << "\nUsage: " << program_name << " [OPTIONS] <image_path>\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help         Show this help message\n";
    std::cout << "  -m, --model PATH   Path to ONNX model (default: resnet18.onnx)\n";
    std::cout << "  -t, --top N        Show top N predictions (default: 5)\n";
    std::cout << "  -v, --verbose      Show detailed output\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " cat.jpg\n";
    std::cout << "  " << program_name << " --top 3 dog.jpg\n";
    std::cout << "  " << program_name << " -m custom_model.onnx image.jpg\n";
    std::cout << std::endl;
}

/**
 * Check if file exists
 */
bool file_exists(const std::string& path) {
    return fs::exists(path);
}

/**
 * Preprocess image for ResNet inference
 */
std::vector<float> preprocess_image(const cv::Mat& img) {
    // Resize to 224x224
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(224, 224));
    
    // BGR -> RGB and convert to float [0, 1]
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);
    
    // ImageNet normalization: (pixel - mean) / std
    cv::Mat channels[3];
    cv::split(rgb, channels);
    channels[0] = (channels[0] - 0.485f) / 0.229f;  // R
    channels[1] = (channels[1] - 0.456f) / 0.224f;  // G
    channels[2] = (channels[2] - 0.406f) / 0.225f;  // B
    
    // Convert HWC to CHW format
    std::vector<float> input_data;
    input_data.reserve(3 * 224 * 224);
    
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < 224; h++) {
            for (int w = 0; w < 224; w++) {
                input_data.push_back(channels[c].at<float>(h, w));
            }
        }
    }
    
    return input_data;
}

/**
 * Find top-k predictions from output scores
 */
std::vector<std::pair<int, float>> get_top_k(float* scores, int num_classes, int k) {
    std::vector<std::pair<int, float>> results;
    results.reserve(num_classes);
    
    for (int i = 0; i < num_classes; i++) {
        results.push_back({i, scores[i]});
    }
    
    std::sort(results.begin(), results.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    results.resize(std::min(k, num_classes));
    return results;
}

// ============================================================================
// Main Function
// ============================================================================
int main(int argc, char* argv[]) {
    // Default parameters
    std::string image_path = "";
    std::string model_path = "resnet18.onnx";
    int top_k = 5;
    bool verbose = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_help(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                model_path = argv[++i];
            } else {
                std::cerr << "Error: --model requires a path argument\n";
                return 1;
            }
        } else if (arg == "-t" || arg == "--top") {
            if (i + 1 < argc) {
                try {
                    top_k = std::stoi(argv[++i]);
                    if (top_k < 1 || top_k > 1000) {
                        std::cerr << "Error: --top must be between 1 and 1000\n";
                        return 1;
                    }
                } catch (...) {
                    std::cerr << "Error: --top requires a number\n";
                    return 1;
                }
            }
        } else if (arg == "-v" || arg == "--verbose") {
            verbose = true;
        } else if (arg[0] != '-') {
            image_path = arg;
        } else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            print_help(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (image_path.empty()) {
        image_path = "test_cat.jpg";  // Default test image
    }
    
    // Print header
    std::cout << "\n========================================\n";
    std::cout << "   ResNet Inference Engine (C++)\n";
    std::cout << "========================================\n\n";
    
    try {
        // ========== 1. Validate Files ==========
        std::cout << "[1] Validating files...\n";
        
        if (!file_exists(image_path)) {
            throw std::runtime_error("Image file not found: " + image_path);
        }
        if (!file_exists(model_path)) {
            throw std::runtime_error("Model file not found: " + model_path);
        }
        
        std::cout << "    Image: " << image_path << "\n";
        std::cout << "    Model: " << model_path << "\n";
        
        // ========== 2. Load Image ==========
        std::cout << "\n[2] Loading image...\n";
        cv::Mat img = cv::imread(image_path);
        
        if (img.empty()) {
            throw std::runtime_error("Failed to decode image: " + image_path);
        }
        std::cout << "    Size: " << img.cols << "x" << img.rows << " pixels\n";
        
        // ========== 3. Initialize ONNX Runtime ==========
        std::cout << "\n[3] Initializing ONNX Runtime...\n";
        
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ResNetInference");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        
        // Convert model path to wide string for Windows
        std::wstring model_path_w(model_path.begin(), model_path.end());
        
        auto start_load = std::chrono::high_resolution_clock::now();
        Ort::Session session(env, model_path_w.c_str(), session_options);
        auto end_load = std::chrono::high_resolution_clock::now();
        
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count();
        std::cout << "    Model loaded in " << load_time << " ms\n";
        
        // ========== 4. Preprocess Image ==========
        std::cout << "\n[4] Preprocessing image...\n";
        
        auto start_preprocess = std::chrono::high_resolution_clock::now();
        std::vector<float> input_data = preprocess_image(img);
        auto end_preprocess = std::chrono::high_resolution_clock::now();
        
        auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(end_preprocess - start_preprocess).count();
        std::cout << "    Done in " << preprocess_time / 1000.0 << " ms\n";
        
        // ========== 5. Create Input Tensor ==========
        std::vector<int64_t> input_shape = {1, 3, 224, 224};
        const char* input_names[] = {"input_tensor"};
        const char* output_names[] = {"output_tensor"};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());
        
        // ========== 6. Run Inference ==========
        std::cout << "\n[5] Running inference...\n";
        
        // Warm-up run
        session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        
        // Timed inference
        const int num_runs = 10;
        auto start_infer = std::chrono::high_resolution_clock::now();
        
        std::vector<Ort::Value> outputs;
        for (int i = 0; i < num_runs; i++) {
            outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        }
        
        auto end_infer = std::chrono::high_resolution_clock::now();
        auto infer_time = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer).count();
        double avg_infer_time = infer_time / 1000.0 / num_runs;
        
        std::cout << "    Inference time: " << avg_infer_time << " ms (avg of " << num_runs << " runs)\n";
        std::cout << "    Throughput: " << 1000.0 / avg_infer_time << " FPS\n";
        
        // ========== 7. Get Results ==========
        std::cout << "\n[6] Results:\n";
        
        float* output_data = outputs[0].GetTensorMutableData<float>();
        auto top_results = get_top_k(output_data, 1000, top_k);
        
        std::cout << "    Top-" << top_k << " Predictions:\n";
        for (size_t i = 0; i < top_results.size(); i++) {
            int class_id = top_results[i].first;
            float score = top_results[i].second;
            std::string class_name = get_class_name(class_id);
            
            std::cout << "      " << (i + 1) << ". " << class_name 
                      << " (score: " << score << ")\n";
        }
        
        // ========== Summary ==========
        std::cout << "\n========================================\n";
        std::cout << "   Performance Summary\n";
        std::cout << "========================================\n";
        std::cout << "  Model load:    " << load_time << " ms\n";
        std::cout << "  Preprocess:    " << preprocess_time / 1000.0 << " ms\n";
        std::cout << "  Inference:     " << avg_infer_time << " ms\n";
        std::cout << "  Total:         " << (load_time + preprocess_time/1000.0 + avg_infer_time) << " ms\n";
        std::cout << "========================================\n\n";
        
        return 0;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "\n[ERROR] ONNX Runtime error: " << e.what() << "\n";
        return 1;
    } catch (const cv::Exception& e) {
        std::cerr << "\n[ERROR] OpenCV error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n[ERROR] Unknown error occurred\n";
        return 1;
    }
}