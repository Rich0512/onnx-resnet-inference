# ONNX ResNet Inference Engine

A high-performance image classification system built with C++ and ONNX Runtime.

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-orange.svg)

## Features

-  **High Performance**: ~150 FPS on CPU
-  **Image Classification**: ResNet18 with ImageNet 1000 classes
-  **Benchmarking**: Built-in performance measurement
-  **Cross-platform**: Windows with Visual Studio Build Tools

## Performance

| Metric | Value |
|--------|-------|
| Model Load Time | ~100 ms |
| Preprocessing | ~0.7 ms |
| Inference | ~7 ms |
| Throughput | ~150 FPS |

## Prerequisites

- Windows 11
- Visual Studio 2022 Build Tools
- CMake 3.10+
- ONNX Runtime 1.16+
- OpenCV 4.10+

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rich0512/onnx-resnet-inference.git
cd onnx-resnet-inference
```

### 2. Download dependencies

- **ONNX Runtime**: [Download](https://github.com/microsoft/onnxruntime/releases) → Extract to `./onnxruntime`
- **OpenCV**: [Download](https://opencv.org/releases/) → Extract to `./opencv`

### 3. Configure environment

Copy `.env.example` to `.env` and adjust paths if needed:

```bash
cp .env.example .env
```

### 4. Build

```bash
.\build.bat
```

## Usage

```bash
cd build\Release

# Run with default test image
.\main.exe

# Run with custom image
.\main.exe path\to\your\image.jpg

# Run with custom model
.\main.exe image.jpg custom_model.onnx
```

### Example Output

```
========================================
   ResNet Inference Engine (C++)
========================================

[1] Loading image: test_cat.jpg
    Image size: 600x801

[2] Initializing ONNX Runtime...
    Model loaded in 102 ms

[3] Preprocessing image...
    Preprocessing done in 0.754 ms

[4] Running inference...
    Inference time: 7.02 ms (avg of 10 runs)
    Throughput: 142.45 FPS

[5] Results:
    Top-5 Predictions:
      1. Class 287 (score: 13.84)  # Lynx
      2. Class 282 (score: 13.11)  # Tiger cat
      3. Class 281 (score: 12.82)  # Tabby cat
      4. Class 285 (score: 11.42)  # Egyptian cat
      5. Class 288 (score: 11.01)  # Leopard
```

## Project Structure

```
onnx-resnet-inference/
├── main.cpp          # Main inference code
├── CMakeLists.txt    # CMake configuration
├── build.bat         # Build script
├── run.bat           # Run script
├── model.py          # Python script to generate ONNX model
├── .env.example      # Environment configuration template
├── onnxruntime/      # ONNX Runtime library (not in repo)
├── opencv/           # OpenCV library (not in repo)
└── build/            # Build output (not in repo)
```

## Technical Details

### Image Preprocessing Pipeline

1. **Resize** to 224×224 (ResNet input size)
2. **Color conversion** BGR → RGB
3. **Normalization** [0, 255] → [0, 1]
4. **ImageNet standardization**
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]
5. **Format conversion** HWC → CHW (PyTorch format)

### ONNX Runtime Configuration

- **Execution Provider**: CPU (default)
- **Intra-op threads**: 4
- **Memory allocator**: Arena allocator

## Roadmap

- [ ] GPU acceleration (CUDA ExecutionProvider)
- [ ] INT8 quantization
- [ ] TensorRT optimization
- [ ] Multi-model support
- [ ] Batch inference

## License

MIT License

## Author

Richard - [GitHub](https://github.com/Rich0512)
