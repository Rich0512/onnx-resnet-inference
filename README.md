# ONNX ResNet Inference Engine

A high-performance image classification system built with C++ and ONNX Runtime, featuring GPU acceleration and model quantization.

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.23-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg)

## ✨ Features

- 🚀 **High Performance**: 150 FPS (CPU) / 630 FPS (GPU)
- 🎮 **GPU Acceleration**: CUDA ExecutionProvider support
- � **Benchmark Mode**: Min/Max/Avg timing statistics
- 🗜️ **Model Quantization**: INT8 dynamic quantization
- �🖼️ **Image Classification**: ResNet18 with ImageNet 1000 classes
- ⚡ **Input Validation**: Robust error handling

## 📁 Project Structure

```
onnx-resnet-inference/
├── src/
│   ├── main.cpp           # Entry point + CLI
│   ├── inference.cpp      # ONNX Runtime engine
│   └── preprocessor.cpp   # Image preprocessing
├── include/
│   ├── inference.h
│   ├── preprocessor.h
│   └── imagenet_labels.h
├── scripts/
│   ├── build.bat          # Build script (CPU/GPU)
│   ├── run.bat            # Run script
│   ├── model.py           # Generate ONNX model
│   └── quantize.py        # INT8 quantization
├── models/
│   └── resnet18.onnx
├── third_party/
│   ├── onnxruntime/
│   └── opencv/
└── CMakeLists.txt
```

## 🚀 Performance

| Mode | Inference | Throughput |
|------|-----------|------------|
| CPU (4 threads) | 6.5 ms | 150 FPS |
| GPU (CUDA) | 1.6 ms | 630 FPS |
| **GPU (TensorRT)** | **~1.0 ms** | **1000+ FPS** |
| INT8 Model | 8.4 ms | 120 FPS |

## 📋 Prerequisites

- Windows 11
- Visual Studio 2022 Build Tools
- CMake 3.10+
- CUDA 12.x + cuDNN 9.x (for GPU)

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Rich0512/onnx-resnet-inference.git
cd onnx-resnet-inference
```

### 2. Download dependencies

| Library | Download | Location |
|---------|----------|----------|
| ONNX Runtime GPU | [Releases](https://github.com/microsoft/onnxruntime/releases) | `third_party/onnxruntime` |
| OpenCV | [Releases](https://opencv.org/releases/) | `third_party/opencv` |

### 3. Generate model

```bash
python scripts/model.py
move resnet18.onnx models/
```

### 4. Build

```bash
# CPU version
scripts\build.bat

# GPU version
scripts\build.bat --gpu
```

## 📖 Usage

```bash
# Basic usage
.\main.exe image.jpg

# GPU acceleration (CUDA)
.\main.exe --gpu image.jpg

# TensorRT optimization (Fastest!)
.\main.exe --tensorrt image.jpg

# Benchmark mode (100 runs)
.\main.exe --gpu --benchmark image.jpg

# Custom model
.\main.exe -m custom.onnx image.jpg

# Show help
.\main.exe --help
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-g, --gpu` | Enable GPU acceleration |
| `-b, --benchmark` | Run 100 iterations with statistics |
| `-m, --model PATH` | Specify ONNX model path |
| `-t, --top N` | Show top N predictions |

### Example Output

```
========================================
   ResNet Inference Engine (C++)
========================================

[1] Validating files...
    Image: test_cat.jpg
    Model: resnet18.onnx

[2] Loading image...
    Size: 600x801 pixels

[3] Initializing ONNX Runtime...
    [GPU mode: CUDA enabled]
    Model loaded in 75 ms

[4] Preprocessing image...
    Done in 0.7 ms

[5] Running inference...
    [Benchmark Mode]
    Runs: 100
    Min:  1.5 ms
    Max:  2.8 ms
    Avg:  1.7 ms
    Throughput: 591 FPS

[6] Results:
    Top-5 Predictions:
      1. lynx (score: 13.84)
      2. tiger cat (score: 13.11)
      3. tabby cat (score: 12.82)

========================================
   Performance Summary
========================================
  Mode:        GPU (CUDA)
  Model load:  75 ms
  Preprocess:  0.7 ms
  Inference:   1.7 ms
========================================
```

## 🗜️ Model Quantization

```bash
# Create INT8 quantized model
python scripts/quantize.py --type int8

# Use quantized model
.\main.exe -m resnet18_int8.onnx image.jpg
```

| Model | Size | Notes |
|-------|------|-------|
| FP32 | 44.6 MB | Original |
| INT8 | 11.2 MB | 75% smaller |

## 🔬 Technical Details

### Image Preprocessing Pipeline

1. Resize to 224×224
2. BGR → RGB conversion
3. Normalize [0,255] → [0,1]
4. ImageNet standardization (mean/std)
5. HWC → CHW format conversion

### ONNX Runtime Configuration

- CPU: 4 threads (IntraOp)
- GPU: CUDA ExecutionProvider

## 📜 License

MIT License

## 👤 Author

Richard - [GitHub](https://github.com/Rich0512)
