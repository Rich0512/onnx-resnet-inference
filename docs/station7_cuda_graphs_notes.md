# 第七站：CUDA Graphs 零調度開銷硬體圖錄製 (Station 7)

> 本文件記錄如何利用 NVIDIA 旗艦技術 CUDA Graphs，將「自訂 CUDA 前處理 Kernel + TensorRT FP16 推論 + D2H 顯存拷貝」一次性固化為 GPU 靜態執行拓撲（DAG），消滅 CPU Launch 開銷，將單串流端到端延遲從 0.43 ms 壓低至 0.30 ms，單串流吞吐量突破 3,300 FPS。

---

## 一、 七大架構演進效能總覽

| 階段 / 架構 | 前處理耗時 | GPU 推論耗時 | 總端到端延遲 | 單串流 FPS | 核心技術里程碑 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **初始基準 (CPU)** | 4.36 ms | 45.0 ms | 49.36 ms | ~20 FPS | OpenCV CPU 多重拷貝 + ORT CPU |
| **第二站 (CUDA EP)** | 4.36 ms | 1.34 ms | 5.70 ms | 175 FPS | ORT CUDA 串流非同步推論，發現 27.4% 轉置損耗 |
| **第三站 (算子融合 CPU)** | 0.23 ms | 1.34 ms | 1.57 ms | 636 FPS | 單次走訪、常數摺疊、Raw Pointer、無多餘分配 |
| **第四站 (ORT-TRT EP)** | 0.23 ms | 0.51 ms | 0.74 ms | 1,350 FPS | 圖融合、FP16 Tensor Core、離線消除轉置 |
| **第五站 (Native TRT 10)** | 0.23 ms | 0.37 ms | 0.60 ms | 1,666 FPS | 脫離 ORT 包裝、原生 enqueueV3、雙串流 3000+ FPS |
| **第六站 (GPU CUDA Kernel)** | **0.0093 ms** | 0.37 ms | 0.43 ms | 2,311 FPS | 端到端顯存直通、單一 Kernel 算子融合 |
| **第七站 (CUDA Graphs)** | **0.0093 ms** | **0.29 ms** | **0.30 ms** | **3,318 FPS** | **硬體拓撲預編譯、消滅 CPU 發射氣泡、單串流破 3300 FPS** |

---

## 二、 為什麼需要 CUDA Graphs？（硬體級痛點剖析）

在第六站中，前處理與推論皆已移至 GPU 執行。然而在微秒級（Microsecond）的高頻推論中，出現了「CPU 發送指令速度追不上 GPU 執行速度」的新瓶頸：

### 1. CPU Kernel Launch Overhead（發射開銷）
在傳統 CUDA 串流模式下，CPU 必須依序呼叫：
1. `launch_cuda_preprocess`（耗時 ~5 微秒）
2. `context->enqueueV3`（內部拆分成數十個層的 Kernel，累積耗時 ~15 微秒）
3. `cudaMemcpyAsync`（耗時 ~5 微秒）

CPU 每次發射指令，都要透過作業系統驅動程式打包封包、透過 PCIe 匯流排推送到 GPU 命令佇列。
這導致了一種窘境：**GPU 運算只要 0.25 ms，但有 0.13 ms 都在等待 CPU 發送下一個指令（CPU Overhead Bubble）**。

### 2. CUDA Graphs 的解決方案
CUDA Graphs 引入「錄製一次、無限次重播」架構：
* **錄製期（Capture）**：CPU 將前處理、推論、D2H 拷貝整個工作流登記為一張有向無環圖（DAG, `cudaGraph_t`）。
* **實例化（Instantiate）**：驅動程式將 DAG 編譯為 GPU 硬體專用的實體執行物件（`cudaGraphExec_t`），預先將所有依賴關係與顯存位址燒錄進 GPU。
* **重播期（Replay / Launch）**：每一幀推論，CPU 僅需發出一條極短的硬體指令 `cudaGraphLaunch`（耗時小於 1 微秒），GPU 硬體排程器便自動無縫接力執行整條管線！

---

## 三、 核心程式碼逐行詳解

### 1. 資源宣告 ([include/trt_inference.h](file:///c:/Users/Richard/Desktop/C++/onnx-resnet-inference/include/trt_inference.h))
```cpp
cudaGraph_t graph_ = nullptr;        // 抽象依賴圖句柄 (DAG)
cudaGraphExec_t graph_exec_ = nullptr; // 預編譯的硬體可執行檔 (Executable Graph)
bool graph_captured_ = false;         // 旗標：標記是否已經完成錄製與實例化
```
* `cudaGraph_t`：儲存節點（Kernel 節點、Memcpy 節點）及其依賴邊的資料結構。
* `cudaGraphExec_t`：經由驅動程式分析、驗證拓撲正確性後的二進位執行計畫，是真正傳給 GPU 重播的實體。

---

### 2. 圖錄製與實例化實作 ([src/trt_inference.cpp](file:///c:/Users/Richard/Desktop/C++/onnx-resnet-inference/src/trt_inference.cpp))
```cpp
void TrtEngine::capture_cuda_graph(const cv::Mat& sample_image) {
    // 1. 暖身執行（極重要：TensorRT 與 CUDA 驅動需要先在串流上跑過一次，
    //    完成內部靜態顯存與 Scratchpad 的初始化，否則無法捕捉）
    cuda_preprocess::launch_cuda_preprocess(...);
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(...);
    cudaStreamSynchronize(stream_);

    // 2. 開啟串流捕捉模式
    cudaError_t err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

    // 3. 發射工作流（此時任務不會真正執行，而是被捕捉並記錄成 Graph 的節點）
    cuda_preprocess::launch_cuda_preprocess(...);  // 節點 1
    context_->enqueueV3(stream_);                  // 節點 2 ~ N
    cudaMemcpyAsync(...);                          // 節點 N+1

    // 4. 結束捕捉，產生依賴圖 (graph_)
    err = cudaStreamEndCapture(stream_, &graph_);

    // 5. 實例化：將 graph_ 編譯成可在硬體直跑的 graph_exec_
    err = cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    graph_captured_ = true;
}
```

---

### 3. 重播執行實作
```cpp
// 只需要 1 次極速呼叫，CPU 發射開銷趨近於 0
cudaGraphLaunch(graph_exec_, stream_);
cudaStreamSynchronize(stream_);
```

---

## 四、 實測成績驗證

在同一張測試圖片（1504x846 解析度）、100 次連續測試下：

```text
========================================
   Performance Comparison (Station 6 vs Station 7)
========================================

[Station 6: 傳統純 GPU 串流管線 (逐次發射)]
  Avg Latency:  0.4326 ms
  Throughput:   2311.12 FPS

[Station 7: CUDA Graph 零調度硬體重播 (全圖固化)]
  Avg Latency:  0.3013 ms
  Throughput:   3318.04 FPS  <--- 單串流狂飆超過 1,000 FPS！

========================================
```

* **延遲改善**：從 0.432 ms 壓低至 **0.301 ms**，單幀延遲直接砍掉 **30.3%**（消滅了約 130 微秒的 CPU 發射氣泡與排程空隙）。
* **吞吐量大躍進**：在單一 Stream、無多線程介入的情況下，FPS 從 2311 直接飆破 **3318 FPS**！
