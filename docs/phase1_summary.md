# 🎓 C++ AI 工程師養成計畫 - 第 1 階段總複習

本文件為 **AI 軟韌體工程師第一階段 (Phase 1)** 的完整技術總結。這份筆記將是你未來面試、複習的重要資產。

---

## 🚀 專案成果：高效能 ResNet 推論引擎

我們從零開始，打造了一個專業級的 C++ AI 推論專案。

### 效能終極對決 (RTX 3060 Ti)

| 模式 | 推論時間 | Throughput | 加速倍率 | 備註 |
|------|---------|------------|----------|------|
| **CPU (基準)** | 6.6 ms | 150 FPS | 1x | 適合低負載場景 |
| **CUDA** | 1.6 ms | 630 FPS | 4.2x | 通用 GPU 加速 |
| **INT8 (CPU)** | 8.4 ms | 120 FPS | 0.8x | 需 AVX-512 才能發揮優勢 |
| **TensorRT** | **~0.6 ms** | **1600 FPS** | **10x** | **極致效能首選** |

> **關鍵發現**: 在高階 GPU 上，TensorRT 的圖優化 (Graph Optimization) 與層融合 (Layer Fusion) 能帶來超越原生 CUDA 的巨大效能提升。

---

## 🧠 核心技術回顧

### 1. ONNX Runtime API (C++)
掌握度：⭐⭐⭐⭐⭐

*   **環境建立**: `Ort::Env` 是整個 Runtime 的環境管理。
*   **載入模型**: `Ort::Session` 負責載入與執行模型。
*   **輸入輸出**:
    *   **Input**: `Ort::Value::CreateTensor` 將 `std::vector` 包裝成 Tensor。
    *   **MemoryInfo**: `Ort::MemoryInfo::CreateCpu` 指定記憶體位置。
*   **執行**: `session.Run` 是同步 blocking call。

### 2. 優化器 (Execution Providers)

我們實作了三種 backend 的切換邏輯：

```cpp
// 1. TensorRT (最快)
OrtTensorRTProviderOptions trt_opts;
trt_opts.trt_max_partition_iterations = 1000; // 強制多切分
trt_opts.trt_min_subgraph_size = 1;          // 允許小圖
session_options.AppendExecutionProvider_TensorRT(trt_opts);

// 2. CUDA (次快 - Fallback)
OrtCUDAProviderOptions cuda_opts;
session_options.AppendExecutionProvider_CUDA(cuda_opts);

// 3. CPU (最慢 - 預設)
// 什麼都不用做，預設即為 CPU
```

### 3. TensorRT 整合眉角 (The Tricky Parts)

*   **環境相依**:
    *   ONNX Runtime 本身只是 Wrapper，底層需要 `onnxruntime_providers_tensorrt.dll`。
    *   更底層需要 TensorRT 原生函式庫：`nvinfer.dll`, `nvonnxparser.dll` 等。
*   **版本匹配**: CUDA 版本必須精確匹配 (CUDA 12.x vs 11.x 互不相容)。
*   **首次延遲**: TensorRT 第一次執行需要編譯 P-Code 為相容該 GPU 的 Binary Engine，所以載入時間較長 (數秒至數十秒)。

### 4. 圖片預處理 pipeline
標準 ImageNet 預處理流程：
1.  **Read**: `cv::imread` (BGR, HWC, uint8)
2.  **Resize**: 調整至 224x224
3.  **Convert**: BGR → RGB
4.  **Normalize**: `(pixel / 255.0 - mean) / std`
5.  **Reorder**: HWC → CHW (Channel-First 對 GPU 更友善)

---

## 🛠️ C++ 工程實力養成

### 1. 現代 C++ 特性
- **Smart Pointers**: 使用 `std::unique_ptr<Ort::Session>` 管理資源生命週期。
- **Exception Handling**: `try-catch` 包覆 ONNX 呼叫，攔截 `Ort::Exception`。
- **STL**: 熟練使用 `std::vector`, `std::string`, `std::chrono`。

### 2. 建置系統 (CMake)
- **多平台**: 使用 CMake 產生 Visual Studio 專案。
- **變數控制**: `option(USE_CUDA "Enable GPU" ON)` 控制編譯流程。
- **前處理器**: `add_definitions(-DUSE_CUDA)` 讓 C++ code 知道是否啟用 GPU。

### 3. 專案架構
- **標頭檔分離**: `.h` 宣告介面，`.cpp` 實作邏輯 (Pimpl 精神)。
- **模組化**: `inference` (推論), `preprocessor` (資料處理), `main` (應用邏輯) 分離。

---

## 📝 面試模擬題庫 (Self-Check)

**Q1: 為什麼 TensorRT 比單純 CUDA 快？**
> A: TensorRT 會進行 **Layer Fusion** (把 Conv+BN+ReLU 合併成一個 kernel 減少記憶體存取) 和 **Kernel Auto-tuning** (針對特定 GPU 型號選擇最佳演算法)。

**Q2: 如何處理 CPU 到 GPU 的資料傳輸瓶頸？**
> A: 目前我們用的是一般的 `std::vector` (CPU 分頁記憶體)。未來可以使用 **Pinned Memory** (Page-locked memory) 來加速 host-to-device 傳輸。

**Q3: 動態量化 (Dynamic Quantization) 在 GPU 上為何變慢？**
> A: 因為 GPU 擅長大規模平行浮點運算。動態量化需要在推論時即時計算 scale factor，這在 GPU 上反而成了 overhead。GPU 通常使用 **靜態量化 (Static Quantization)** 或 **INT8 Tensor Core** 加速才有效。

---

## 🔮 下一步：Phase 2

你已經準備好進入 **AI 系統架構** 的領域了。
- **記憶體優化**: Pinned Memory
- **平行化**: CUDA Streams, Multi-threading
- **服務化**: gRPC / HTTP Server

**Keep Coding!** 🚀
