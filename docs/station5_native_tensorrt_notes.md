# 第五站：100% 純 C++ 原生 TensorRT (Native C++ API) 實戰筆記

> 本文件記錄如何徹底擺脫 ONNX Runtime 與 Python 包裝，使用 NVIDIA 原生 C++ API 打造零框架調度開銷、端到端極致推論引擎。

---

## 一、 架構演進全貌：五大階段效能里程碑

| 階段 / 架構 | 前處理耗時 | GPU 推論耗時 | 總端到端延遲 | 吞吐量 (FPS) | 核心技術與最佳化 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **初始基準 (CPU)** | 4.36 ms | 45.0 ms | 49.36 ms | ~20 FPS | OpenCV CPU + ONNX Runtime CPU |
| **第二站 (CUDA EP)** | 4.36 ms | 1.34 ms | 5.70 ms | 746 FPS | CUDA 串流非同步推論，發現 27.4% 轉置損耗 |
| **第三站 (算子融合)** | **0.23 ms** | 1.34 ms | 1.57 ms | 746 FPS | 單次遍歷、常數摺疊、Raw Pointer、免多餘分配 |
| **第四站 (ORT-TRT EP)** | **0.23 ms** | 0.51 ms | 0.74 ms | 1960 FPS | 圖融合、FP16 Tensor Core、佈局離線轉置消除 |
| **第五站 (原生 Pure C++)** | **0.23 ms** | **0.39 ms** | **0.62 ms** | **2516 FPS** | **脫離 ORT 包裝、原生 enqueueV3、零調度開銷** |

---

## 二、 方案 A (ORT-TRT) vs 方案 B (Native TensorRT C++) 深度對比

```text
[方案 A: ONNX Runtime + TensorRT EP]
Client Code -> Ort::Session::Run() -> Provider Interface -> Subgraph Partition -> nvinfer.dll
  (缺點: 每一幀都要走 ONNX Runtime 的 Ort::Value 封裝與派發檢查，框架延遲約 0.12 ~ 0.15 ms)

[方案 B: 原生 Native TensorRT C++ (main_trt.exe)]
Client Code -> context->enqueueV3(stream) -> GPU Hardware 直接執行
  (優點: 零中介層、GPU 顯存指標直接綁定，推論延遲壓至 0.39 ms，吞吐量突破 2500 FPS)
```

---

## 三、 原生 TensorRT 10 C++ 核心實作五部曲

在 [src/trt_inference.cpp](file:///c:/Users/Richard/Desktop/C++/onnx-resnet-inference/src/trt_inference.cpp) 中，實作了標準的工業級原生流程：

### 1. 建立日誌記錄器 (ILogger)
```cpp
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;
```

### 2. 反序列化 Engine 與建立 Execution Context
```cpp
// 讀取二進位 .engine 檔案
std::ifstream file(engine_path, std::ios::binary);
// ... 讀取至 std::vector<char> engine_data ...

std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(gLogger));
std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(engine_data.data(), size));
std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
```

### 3. 分配顯存與 TensorRT 10 原生位址綁定 (`setTensorAddress`)
```cpp
void* d_input = nullptr;
void* d_output = nullptr;
cudaMalloc(&d_input, 1 * 3 * 224 * 224 * sizeof(float));
cudaMalloc(&d_output, 1000 * sizeof(float));

// TensorRT 10 廢棄舊版 setBindingDimensions / enqueueV2
// 全面採用名稱綁定：
context->setTensorAddress("input_tensor", d_input);
context->setTensorAddress("output_tensor", d_output);
```

### 4. 結合第三站算子融合 C++ 前處理與異步 H2D 拷貝
```cpp
// 1. 第三站 0.23 ms 極速前處理
preprocessor::preprocess_image(image, h_input_buffer_);

// 2. 異步拷貝至 GPU 顯存
cudaMemcpyAsync(d_input, h_input_buffer_.data(), input_size, cudaMemcpyHostToDevice, stream);
```

### 5. 異步推理入列 (`enqueueV3`)
```cpp
// 3. 異步發送 Kernel 到 CUDA Stream
context->enqueueV3(stream);

// 4. 異步將輸出拷貝回 Host
cudaMemcpyAsync(h_output_buffer_.data(), d_output, output_size, cudaMemcpyDeviceToHost, stream);

// 5. 等待串流完成
cudaStreamSynchronize(stream);
```

---

## 四、 實戰中遇到的核心踩坑與面試精選 Q&A

### Q1: 為什麼 TensorRT 載入時會報 `Serialization assertion stdVersionRead == kSERIALIZATION_VERSION failed. Version tag does not match (239 vs 240)`？
* **現象**：用 Python TRT 10.7 (Version Tag 239) 嘗試讀取由 C++ ORT TRT (Version Tag 240) 編譯的 `.engine` 時發生崩潰。
* **底層原因**：
  * TensorRT 的 `.engine` 檔案**不是通用模型格式**，它是特定硬體微架構 (SM86)、特定 CUDA 驅動、以及特定 TensorRT 編譯器內部版本**緊密綁定**的二進位機器碼與執行計畫 (Execution Plan)。
  * NVIDIA 為了確保穩定性與硬體相容性，在 Engine 檔頭寫入了 `kSERIALIZATION_VERSION`。即便同為 TensorRT 10，只要內部 patch 版本不同，都會被拒絕載入。
* **工業界標準做法**：
  * **不要跨環境傳輸 `.engine` 檔案**。
  * 在部署伺服器初次啟動時，由部署程式在該台機器的實體 GPU 上將 `.onnx` 即時編譯並序列化至本機快取；後續啟動時直接讀取本地快取。

### Q2: 既然方案 B 比方案 A 快了將近 30%，那為什麼很多公司還是選擇方案 A (ONNX Runtime + TRT)？
* **架構取捨考量 (Trade-offs)**：
  1. **模型相容性與算子退化 (Fallback)**：
     * 原生 TensorRT（方案 B）要求模型中的每一個 Operator 必須被 TensorRT 完全支援，只要有一個算子不支援，編譯就會報錯（除非自己寫 C++ CUDA Custom Plugin）。
     * 方案 A (ORT-TRT) 具備「自動子圖切分與退化」功能：支援的層自動跑 TensorRT，不支援的層自動 Fallback 到 CUDA 或 CPU，不會導致整個服務崩潰。
  2. **多硬體相容性**：
     * 方案 A 只需要換一個 Provider 參數（`AppendExecutionProvider_CPU` / `AppendExecutionProvider_OpenVINO`），同一套 C++ 程式碼就能在 Intel CPU、AMD GPU、NVIDIA GPU 跑。
     * 方案 B 是 NVIDIA GPU 專用硬體綁定，程式碼移植到非 NVIDIA 環境必須重寫。

---

## 五、 如何執行純 C++ 原生引擎

在 `build/Release` 目錄下執行：

```powershell
# 執行原生 TensorRT 10 C++ 引擎（自動搜尋快取引擎）
.\main_trt.exe -b test_cat.jpg
```

**輸出結果驗證**：
```text
========================================
   Native TensorRT Engine (Option B - Pure C++)
========================================
[1] Validating files...
    Image:  test_cat.jpg
    Engine: trt_cache\TensorrtExecutionProvider_TRTKernel_..._fp16_sm86.engine
[2] Loading image...
    Size: 600x801 pixels
[3] Initializing Native TensorRT Engine...
    Engine loaded & deserialized in 136.7 ms
[4] Running End-to-End Prediction (Station 3 Fused C++ Preprocessor)...
    Preprocessing: 0.24 ms
    Inference:     0.39 ms
[5] Running Benchmark Mode (100 Iterations)...
    Avg:  0.397 ms
    FPS:  2516.19 FPS
[6] Top-5 Predictions:
    1. lynx (score: 13.7891, id: 287)
    2. tiger cat (score: 13.1172, id: 282)
    3. tabby cat (score: 12.8203, id: 281)
========================================
```
