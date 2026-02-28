#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "inference.h"
#include "preprocessor.h"

namespace py = pybind11;

/**
 * @brief Python wrapper for the Inference Engine
 */
class ResNetWrapper {
public:
    ResNetWrapper(const std::string& model_path, int num_threads = 4, bool use_gpu = false, bool use_tensorrt = false) {
        engine_ = std::make_unique<inference::Engine>(model_path, num_threads, use_gpu, use_tensorrt);
    }

    /**
     * @brief Run inference on raw image bytes received from Python
     * @param image_bytes Raw JPG/PNG image bytes
     * @return List of 1000 float scores
     */
    std::vector<float> predict(const py::bytes& image_bytes) {
        // Thread-local buffer: ensures buffer reuse for performance 
        // while remaining thread-safe for Web Server (FastAPI) concurrency.
        thread_local preprocessor::DataVector thread_local_input_buffer;

        // Convert Python bytes to std::vector<uchar>
        std::string bytes_str = image_bytes;
        std::vector<uchar> data(bytes_str.begin(), bytes_str.end());

        // Decode image directly from memory (no disk I/O)
        cv::Mat img = cv::imdecode(data, cv::IMREAD_COLOR);
        if (img.empty()) {
            throw std::runtime_error("Failed to decode image bytes. Invalid format?");
        }

        // Preprocess reusing the thread-local pinned memory buffer
        preprocessor::preprocess_image(img, thread_local_input_buffer);

        // Run inference
        // Note: Ort::Session::Run is thread-safe for inference
        return engine_->run(thread_local_input_buffer.data(), thread_local_input_buffer.size());
    }

private:
    std::unique_ptr<inference::Engine> engine_;
};

// Define the Python module
PYBIND11_MODULE(resnet_engine, m) {
    m.doc() = "ResNet ONNX Inference Engine with CUDA/TensorRT Support";

    py::class_<ResNetWrapper>(m, "Engine")
        .def(py::init<const std::string&, int, bool, bool>(),
             py::arg("model_path"),
             py::arg("num_threads") = 4,
             py::arg("use_gpu") = false,
             py::arg("use_tensorrt") = false,
             "Initialize the Inference Engine")
        .def("predict", &ResNetWrapper::predict, 
             "Run inference on raw image bytes. Returns 1000 class scores.",
             py::arg("image_bytes"));
}
