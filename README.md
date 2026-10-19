# ONNX ResNet Inference Engine

A high-performance image classification system built with C++ and ONNX Runtime, featuring GPU acceleration, hardware timeline profiling, and operator-fused preprocessing.

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.23-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.10-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900.svg)
![Nsight Systems](https://img.shields.io/badge/Nsight%20Systems-2024.5-76B900.svg)

## ✨ Features

- 🚀 **High Performance**: End-to-end low-latency inference with sub-millisecond preprocessing
- 🎮 **GPU Acceleration**: CUDA & TensorRT ExecutionProvider with multi-stream async pipeline
- ⚡ **Advanced Memory Management**: Zero-Copy Pinned Memory Allocator (`cudaHostAlloc`) & Buffer Reuse
- 🔬 **Hardware-Level Timeline Profiling**: Validated with **NVIDIA Nsight Systems (`nsys`)** for true computation/communication overlap
- 🛠️ **Operator-Fused Preprocessor**: Single-pass traversal, FMA arithmetic, and zero dynamic memory allocations
- 📊 **Benchmark Mode**: Comprehensive Min/Max/Avg/FPS timing statistics
- 🗜️ **Model Quantization**: INT8 dynamic quantization support

---

## 📚 Deep Dive Technical Notes (學習優化專題)

本專案對標 AI Compiler / AI Systems 職缺之底層核心觀念，沉澱了完整系統調優技術筆記：

| 專題文件 | 核心技術領域 | 涵蓋重點 |
| :--- | :--- | :--- |
| 📖 [**Station 1: Runtime 系統架構與記憶體管理**](docs/station1_runtime_memory_notes.md) | **Memory & HW-SW Interface** | • Paging vs. Pinned Memory (`cudaHostAlloc`) 原理<br>• GPU DMA 實體位址直抽與 Zero-Copy Tensor (`Ort::Value`)<br>• RAII PinnedAllocator 與多 CUDA Streams 非同步佇列 |
| 🔬 [**Station 2: Nsight Systems 硬體時間軸剖析**](docs/station2_nsight_systems_notes.md) | **Timeline Profiling & Overlap** | • 拒絕 Wall-Clock 盲測，以 CUPTI 微秒時間軸實證<br>• Copy Engine (PCIe) 與 Compute Engine (Tensor Core) 真重疊<br>• 抓包 `nchwToNhwcKernel` (26.8% 開銷) 與 CPU 瓶頸定位 |
| ⚡ [**Station 3: CPU 前處理算子融合極限優化**](docs/station3_simd_preprocessor_notes.md) | **Operator Fusion & Cache Locality** | • 將 5 次記憶體進出融合成單次走訪 (Single-Pass)<br>• 徹底消滅暫存 `cv::Mat` (0 Heap Allocation)<br>• `constexpr` + 硬體 FMA 單週期指令 (`x * scale + bias`) |
| 🚀 [**Station 4: AI Compiler 與 TensorRT 落地實戰**](docs/station4_tensorrt_compiler_notes.md) | **Engine Serialization & FP16** | • Kernel Fusion 與硬體 Auto-Tuning 機制<br>• Engine 序列化快取，啟動時間 21s ➔ 399ms (加速 54x)<br>• FP16 半精度加速，突破 2000+ FPS 大關 (0.49 ms) |
| 🏎️ [**Station 5: 100% 純 C++ 原生 TensorRT 引擎**](docs/station5_native_tensorrt_notes.md) | **Native C++ API & Zero Overhead** | • 擺脫 ONNX Runtime 與 Python 包裝，純 C++ 原生引擎<br>• `IRuntime` ➔ `ICudaEngine` ➔ `IExecutionContext` ➔ `enqueueV3`<br>• 零調度開銷，延遲壓至 **0.39 ms (2516+ FPS)** |
| ⚡ [**Station 6: GPU 原生自訂 CUDA Kernel 前處理**](docs/station6_cuda_preprocessor_notes.md) | **Custom CUDA Kernel & Memory Zero-Copy** | • 將 Resize + BGR轉RGB + Normalize + NCHW 轉置融合為單一 CUDA Kernel<br>• 2D Thread Grid (14x14 Blocks) 映射與雙線性插值<br>• 前處理延遲從 0.23 ms 暴降至 **0.0093 ms (>100,000 FPS)** |

---

## 🚀 Performance

### 1. End-to-End Pipeline Latency

| 階段 / 模式 | 優化前 (Baseline) | 優化後 (Optimized) | 改善幅度 |
| :--- | :--- | :--- | :--- |
| **CPU 影像前處理** | 4.36 ms (OpenCV 多步驟) | **0.23 ms** (算子融合單次走訪) | 🚀 **18.9x 狂飆** |
| **GPU 影像前處理 (自訂 CUDA Kernel)** | 4.36 ms (OpenCV 多步驟) | **0.0093 ms** (單 Kernel 融合) | 🚀 **468x 狂飆 (>100k FPS)** |
| **GPU 推論 (CUDA FP32)** | 1.60 ms | **1.32 ms** | 1.2x |
| **GPU 推論 (ORT-TRT FP16)** | - | **0.49 ms** | 🚀 **突破 2,000 FPS！** |
| **GPU 推論 (Native TensorRT 10 Pure C++)** | - | **0.37 ms** | 🚀 **突破 2,700 FPS (極致)** |
| **純 GPU 端到端 Pipeline (Preprocess + TRT + D2H)** | 49.36 ms (全 CPU 基準) | **0.42 ms** (純 GPU 直通) | 🚀 **117x 加速 (2375 FPS)** |
| **多 Stream 非同步吞吐量 (ORT-TRT)** | 150 FPS (CPU) | **2000 FPS** (Dual-Stream ORT) | **13.3x** |
| **多 Stream 非同步吞吐量 (Native TRT)** | 150 FPS (CPU) | **3000+ FPS** (Dual-Stream Pure C++) | 🚀 **20.1x (極限吞吐)** |

### 2. Model Modes Comparison

| Mode | Inference | Throughput | Improvement |
|------|-----------|------------|-------------|
| CPU (4 threads) | 6.5 ms | 150 FPS | 1x |
| GPU (CUDA FP32) | 1.32 ms | 760 FPS | 5.1x |
| GPU (TensorRT FP16 + Engine Cache) | 0.49 ms | 2036 FPS | 13.5x |
| GPU (Native TensorRT 10 Pure C++ Single Stream) | 0.37 ms | 2723 FPS | 18.1x |
| **GPU (Native TensorRT 10 Pure C++ Dual-Stream)** | **0.33 ms** | **3018+ FPS** | 🚀 **20.1x (極致高峰)** |
| **Station 6 GPU CUDA Preprocessor (Alone)** | **0.0093 ms** | **106,700+ FPS** | 🚀 **468x vs Baseline** |
| INT8 Model | 8.4 ms | 120 FPS | - |

---

## 📁 Project Structure

```
onnx-resnet-inference/
├── src/
│   ├── main.cpp           # Entry point + CLI benchmark
│   ├── inference.cpp      # ONNX Runtime engine (CUDA / Streams)
│   └── preprocessor.cpp   # Operator-fused image preprocessing
├── include/
│   ├── inference.h
│   ├── pinned_allocator.h # Custom CUDA Pinned Memory STL Allocator
│   ├── preprocessor.h
│   └── imagenet_labels.h
├── docs/                  # In-depth architectural & profiling notes
│   ├── station1_runtime_memory_notes.md
│   ├── station2_nsight_systems_notes.md
│   └── station3_simd_preprocessor_notes.md
├── scripts/
│   ├── build.bat          # One-click build script (MSVC / CUDA)
│   ├── run.bat            # Inference launch script
│   ├── model.py           # PyTorch to ONNX exporter
│   └── quantize.py        # INT8 quantization script
├── models/
│   └── resnet18.onnx
└── CMakeLists.txt
```

---

## 📋 Prerequisites

- Windows 10 / 11
- Visual Studio 2022 (with C++ Desktop Development)
- CMake 3.10+
- CUDA Toolkit 12.x + cuDNN 9.x
- NVIDIA Nsight Systems 2024.x (Optional, for timeline profiling)

---

## 🔧 Build & Run

### 1. Build

```bash
# GPU version (Recommended)
.\scripts\build.bat --gpu

# CPU-only version
.\scripts\build.bat
```

### 2. Run Inference & Benchmark

```bash
cd build\Release

# Run inference on test image with GPU benchmark
.\main.exe --gpu -b test_cat.jpg

# Run with TensorRT acceleration
.\main.exe --tensorrt test_cat.jpg
```

---

## 🔬 Hardware Timeline Profiling (Nsight Systems)

To capture hardware execution traces with CUDA streams and memory transfers:

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe" profile `
    -t cuda,nvtx `
    -o profile_run `
    --force-overwrite true `
    .\build\Release\main.exe --gpu -b test_cat.jpg

# Open timeline in GUI
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\host-windows-x64\nsys-ui.exe" profile_run.nsys-rep
```

---

## 📜 License

MIT License

## 👤 Author

Richard - [GitHub](https://github.com/Rich0512)
