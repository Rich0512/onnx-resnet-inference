# ONNX ResNet Inference Engine

A high-performance image classification system built with C++ and ONNX Runtime.

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-orange.svg)

## Features

- 🚀 **High Performance**: ~150 FPS on CPU
- 🖼️ **Image Classification**: ResNet18 with ImageNet 1000 classes
- ⏱️ **Benchmarking**: Built-in performance measurement
- 🔧 **Cross-platform**: Windows with Visual Studio Build Tools

## Project Structure

```
onnx-resnet-inference/
├── src/                 # Source code
│   └── main.cpp
├── include/             # Header files (future)
├── scripts/             # Build and run scripts
│   ├── build.bat
│   ├── run.bat
│   └── model.py
├── config/              # Configuration files
│   └── .env.example
├── third_party/         # External libraries
│   ├── onnxruntime/
│   └── opencv/
├── models/              # Model files
│   └── resnet18.onnx
├── data/                # Test data
├── build/               # Build output (auto-generated)
├── CMakeLists.txt
├── README.md
└── .gitignore
```

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

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rich0512/onnx-resnet-inference.git
cd onnx-resnet-inference
```

### 2. Download dependencies

Download and extract to `third_party/`:

- **ONNX Runtime**: [Download](https://github.com/microsoft/onnxruntime/releases) → `third_party/onnxruntime`
- **OpenCV**: [Download](https://opencv.org/releases/) → `third_party/opencv`

### 3. Configure environment

```bash
cp config/.env.example .env
# Edit .env if needed
```

### 4. Download model

```bash
# Run Python script to generate model
python scripts/model.py
# Move to models folder
move resnet18.onnx models/
```

### 5. Build

```bash
scripts\build.bat
```

## Usage

```bash
# Run with test image
scripts\run.bat

# Run with custom image
scripts\run.bat path\to\image.jpg
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
    Preprocessing done in 0.7 ms

[4] Running inference...
    Inference time: 7 ms (avg of 10 runs)
    Throughput: 142 FPS

[5] Results:
    Top-5 Predictions:
      1. Class 287 - Lynx
      2. Class 282 - Tiger cat
      3. Class 281 - Tabby cat
```

## Technical Details

### Image Preprocessing

1. Resize to 224×224
2. BGR → RGB conversion
3. Normalize [0, 255] → [0, 1]
4. ImageNet standardization (Mean/Std)
5. HWC → CHW format

### ONNX Runtime Config

- Execution Provider: CPU
- Intra-op threads: 4

## Roadmap

- [ ] GPU acceleration (CUDA)
- [ ] INT8 quantization
- [ ] TensorRT optimization
- [ ] Multi-model support

## License

MIT License

## Author

Richard - [GitHub](https://github.com/Rich0512)
