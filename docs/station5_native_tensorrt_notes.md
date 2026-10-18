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

---

## 五、 原生多 Streams 併發管線化 (Multi-Stream Pipelining)

剛才我們在 [src/trt_inference.cpp](file:///c:/Users/Richard/Desktop/C++/onnx-resnet-inference/src/trt_inference.cpp#L199-L270) 實作了雙 Streams 併發，將吞吐量推升到全新高峰：

### 核心架構：共享 Engine，獨立 Context
```text
                          +------------------------+
                          |   ICudaEngine (唯讀)    |
                          +------------------------+
                                /            \
                               /              \
            +---------------------+        +---------------------+
            | IExecutionContext 1 |        | IExecutionContext 2 |
            | d_input_1 / stream1 |        | d_input_2 / stream2 |
            +---------------------+        +---------------------+
                       |                              |
            context_1->enqueueV3()         context_2->enqueueV3()
                       \                              /
                        v                            v
                      GPU Tensor Core 同時分時交錯運算 (2950 FPS)
```

| 模式 | 執行緒 / Streams | 100 次總耗時 | 吞吐量 (Throughput) |
| :--- | :--- | :--- | :--- |
| **單 Stream 原生推論** | 1 Thread + 1 Stream | 37.6 ms | 2657 FPS |
| **雙 Streams 併發管線** | **2 Threads + 2 Streams** | **33.8 ms** | **2950 FPS** |

---

## 六、 新手解密：寫給 Python / 演算法工程師的「C++ 關鍵語法白話速查」

> **心態建立**：學 AI Systems / HPC 的核心是**「理解硬體在幹嘛」**，而不是「默背 C++ 艱深的語法」。在推論引擎中，C++ 只是我們用來「遙控 GPU」最直接的一把螺絲起子。

以下把專案中你看到的所有 C++ 語法，用 Python 直覺一對一翻譯：

| C++ 語法 | 實際長相 | Python 對應直覺 | 為什麼這樣寫？（硬體目的） |
| :--- | :--- | :--- | :--- |
| **`std::unique_ptr`** | `std::unique_ptr<nvinfer1::IRuntime> runtime;` | 類似 `with ... as f:` 離開時自動 `del` | **防記憶體洩漏**。當函式結束時，C++ 會自動幫你把 GPU/CPU 記憶體乾淨釋放，不用手動 `free()`。 |
| **指標與取址 `*` 與 `&`** | `cudaMalloc(&d_input, size)`<br>`context->setTensorAddress(..., d_input)` | 傳遞記憶體「門牌號碼」 | GPU 顯存非常龐大，我們不能像 Python 那樣動不動傳遞整個物件，而是給它門牌號碼（指標），GPU 直接去該位址讀寫。 |
| **`std::vector`** | `std::vector<float> buffer(1000);` | 就是 Python 的 `list`（可連續存取的陣列） | 在 CPU 記憶體中開一塊連續空間放資料。 |
| **`std::async` 與 `std::future`** | `auto f = std::async(worker);`<br>`f.get();` | 類似 `threading.Thread` 或 `asyncio.create_task` | 在 CPU 背景另外開一條執行緒去呼叫 GPU，最後用 `.get()` 等它做完。 |
| **`cudaStream_t`** | `cudaStreamCreate(&stream);`<br>`enqueueV3(stream);` | 類似非同步任務隊列 (Task Queue) | GPU 就像一條工廠輸送帶，Stream 就是輸送帶編號。多個 Stream 代表多條輸送帶同時送貨進工廠。 |
| **`cudaStreamSynchronize`** | `cudaStreamSynchronize(stream);` | 類似 `await task` 或 `thread.join()` | 因為對 GPU 發送指令是非同步（發完 CPU 就繼續往下跑了），所以當我們要印出預測結果時，必須喊一聲「等 GPU 輸送帶跑完再看結果」。 |

---

## 七、 如何執行純 C++ 原生引擎

在 `build/Release` 目錄下執行：

```powershell
# 執行原生 TensorRT 10 C++ 引擎（單 Stream + 雙 Streams 自動跑分）
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
    Engine loaded & deserialized in 130.3 ms
[4] Running End-to-End Prediction (Station 3 Fused C++ Preprocessor)...
    Preprocessing: 0.24 ms
    Inference:     0.37 ms
[5] Running Benchmark Mode (100 Iterations)...
    [Native TensorRT Benchmark Mode]
    Avg:  0.376 ms (2657 FPS)
    [Native TensorRT Multi-Threaded Async Benchmark (2 Streams)]
    Total time for 100 runs: 33.89 ms
    Throughput: 2950.17 FPS
[6] Top-5 Predictions:
    1. lynx (score: 13.7891, id: 287)
    2. tiger cat (score: 13.1172, id: 282)
    3. tabby cat (score: 12.8203, id: 281)
========================================
```

