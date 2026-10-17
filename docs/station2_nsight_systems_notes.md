# 第二站：NVIDIA Nsight Systems 硬體時間軸剖析與 Stream 重疊實戰 (Station 2 Notes)

> 本文件延續第一站的記憶體管理與非同步概念，深入硬體層級，使用 **NVIDIA Nsight Systems (`nsys`)** 親眼驗證 GPU 內部 **Copy Engine** 與 **Compute Engine** 的並行重疊（Overlap），並掌握 AI Systems / AI Compiler 必備的 Timeline Profiling 核心技能。

---

## 🎯 職缺需求與今日學習對照

| 職缺技能要求 (JD) | 今日專案對照代碼 / 工具 | 掌握的底層核心概念 |
| :--- | :--- | :--- |
| **Performance Profiling & Bottleneck Analysis** | `nsys profile`<br>`nsys stats`<br>`nsys-ui` | • 牆上時間 (Wall-Clock) vs. GPU 硬體真實時間軸<br>• CUPTI (CUDA Profiling Tools Interface) 採集機制<br>• GPU 雙引擎模型 (Copy Engine vs. Compute Engine) |
| **Concurrency & Pipeline Overlapping** | `src/main.cpp` (2 Streams)<br>`src/inference.cpp` (`run_async`) | • 驗證多 CUDA Streams 是否真正物理並行 (Overlap)<br>• 識別隱式同步 (Implicit Synchronization) 陷阱<br>• Default Stream (Stream 0) 造成的管線排空效應 |
| **Code Instrumentation & Observability** | NVTX (NVIDIA Tools Extension)<br>`nvtxRangePushA` / `nvtxRangePop` | • 將 C++ 業務邏輯（前處理、排程、後處理）投影至硬體時間軸<br>• 精準對齊 CPU 執行緒事件與 GPU Kernel 發射點 |

---

## 🧠 一、 為什麼只用 `std::chrono` 不夠？

在第一站中，我們用 `std::chrono::high_resolution_clock` 量出了推論時間（例如 ~1.6 ms）。但這種測量有致命盲點：

1. **非同步延遲欺騙（Asynchronous Illusion）**：
   * CUDA 呼叫（如 `cudaMemcpyAsync` 或 `session_->Run`）往往是**非同步發射（Enqueue）**的。CPU 執行這行程式碼可能只需 5 微秒，但 GPU 實際上要花 1,000 微秒才算完。
   * 如果沒有在前後加上精確的硬體事件同步（`cudaEventSynchronize`），`std::chrono` 量的往往只是 CPU 丟工單的時間，而非硬體真實執行時間。
2. **黑盒子問題（Black Box Issue）**：
   * 即使在最後加上 `cudaDeviceSynchronize()` 量得總耗時，你依然不知道：
     * **GPU 到底何時在算？何時在等資料（PCIe Stalling）？**
     * **兩個 CUDA Streams 到底是一前一後串行，還是同時開跑？**
     * **哪一個 Kernel 是最嚴重的效能瓶頸（Bottleneck）？**

> **AI Systems 工程師準則**：
> 「如果沒有用硬體 Profiler 親眼看過時間軸，你宣稱的『非同步並行』很可能只是美麗的誤會。」

---

## 🏗️ 二、 GPU 硬體架構：雙引擎並行模型

在 NVIDIA 顯卡（如 RTX 30 / 40 系列，Ampere/Ada 架構）硬體內部，運算與傳輸是由**完全獨立的實體硬體單元**負責的：

```
             ┌────────────────────────────────────────────────────────┐
             │                     NVIDIA GPU                         │
             │                                                        │
 Host RAM ───┼─► [Copy Engine (DMA HtoD)] ──┐                         │
(Pinned Mem) │                              ▼                         │
             │                         [VRAM / L2] ◄──► [Compute Engine]
             │                              ▲           (SM / Tensor Cores)
 Host RAM ◄──┼── [Copy Engine (DMA DtoH)] ──┘                         │
             └────────────────────────────────────────────────────────┘
```

1. **Copy Engine（DMA 搬運單元）**：
   * 負責走 PCIe 總線在 Host RAM 與 GPU VRAM 之間傳遞資料。
   * **獨立運作**，完全不佔用 GPU 的 CUDA Cores / Tensor Cores 運算資源。
2. **Compute Engine（SM 串流多處理器 / Tensor Core）**：
   * 負責執行矩陣乘法、卷積（GEMM/Winograd）、ReLU、Pooling 等算子。
3. **黃金並行（True Overlap）**：
   * **Stream 1** 讓 **Compute Engine** 運算 Batch A 的 ResNet 卷積。
   * 同一瞬間，**Stream 2** 讓 **Copy Engine** 透過 PCIe 將 Batch B 的圖片資料拷貝進顯卡。
   * 傳輸時間被運算時間完全隱藏（Latency Hiding），達到兩倍的吞吐量！

---

## 🔬 三、 Nsight Systems 實戰指令與資料剖析

### 1. 採集 Profiling 資料
在 Windows 環境下，使用 Nsight Systems CLI 對推論程式進行抓取：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe" profile `
    -t cuda,nvtx `
    -o profile_run `
    --force-overwrite true `
    .\build\Release\main.exe --gpu -b test_cat.jpg
```

* `-t cuda,nvtx`：指定追蹤 CUDA API/Kernel 以及 NVTX 標籤。
* `-o profile_run`：產生 `profile_run.nsys-rep` 檔案。

### 2. 用指令直接產出關鍵統計摘要 (`nsys stats`)
無需開啟 GUI，直接在終端機提取統計報表：

```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\target-windows-x64\nsys.exe" stats `
    profile_run.nsys-rep `
    --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum
```

#### 本專案真實採集數據解讀：

1. **最耗時的 GPU Kernel（Top 1）**：
   * `sm86_xmma_fprop_implicit_gemm_indexed_tf32f32_...`
   * **真相**：這就是 ResNet 最核心的卷積層！Ampere 架構（sm86）使用 **Tensor Core** 以 **TF32** 精度進行隱式 GEMM（Implicit GEMM），單次執行耗時僅約 **29 ~ 32 微秒 (µs)**，佔據了 GPU 總運算時間的 **37.8%**。
2. **記憶體轉移時間分佈**：
   * `[CUDA memcpy Host-to-Device]` 佔據了搬移時間的 **94.1%**，平均每次耗時約 **97 ~ 112 微秒 (µs)**。
   * **結論**：PCIe 傳輸時間遠大於個別單一算子！因此將 PCIe 搬移與計算重疊（Overlap）是提升整體 Throughput 的唯一手段。

---

## 🖥️ 四、 GUI 時間軸圖解指南（如何看懂 `nsys-ui`）

啟動 GUI 工具：
```powershell
& "C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.5.1\host-windows-x64\nsys-ui.exe" .\build\Release\profile_run.nsys-rep
```

打開檔案後，將視窗橫向拉大，按住 `Ctrl + 滑鼠滾輪` 放大時間軸：

```
時間軸 (Timeline) ────────────────────────────────────────────────────────►
[CUDA HW]
 ├─ Stream 14: [Memcpy HtoD]───►[sm86_gemm Kernel]───►[cudnn_relu]
 ├─ Stream 15:       [Memcpy HtoD]───────►[sm86_gemm Kernel]
                     ▲
                     └─ 關鍵觀察點：Stream 15 的傳輸是否與 Stream 14 的運算在同一個垂直時間線上？
```

### 必備的診斷檢查清單（Checklist）：
1. **多 Stream 是否重疊？**
   * 檢查是否有兩個不同 Stream 軌道上的色塊在同一垂直時間區間內並行。
2. **GPU 氣泡（Pipeline Bubbles / Gaps）**：
   * 如果 Kernel 與 Kernel 之間有大段空白，代表 GPU 處於飢餓狀態（Starvation），通常是因為 CPU 前處理太慢，或是 Host-to-Device 傳輸阻塞。
3. **隱式同步（Implicit Synchronization）**：
   * 若發現所有 Stream 突然在一條垂線上全部停擺等待，檢查程式碼中是否呼叫了同步 API（例如沒有指定 Stream 的 `cudaMemcpy`、`cudaMalloc`、或是讀取了非非同步的結果）。

---

## 🏷️ 五、 專業進階技能：NVTX 標籤注入

若要在時間軸上明確看見每個階段，可引入 NVIDIA Tools Extension (NVTX)：

```cpp
#include <nvtx3/nvToolsExt.h>

void process_frame() {
    nvtxRangePushA("1. Preprocessing");
    // OpenCV resize, normalize...
    nvtxRangePop();

    nvtxRangePushA("2. Enqueue Inference");
    engine.run_async(data, size, stream);
    nvtxRangePop();
}
```
在 Nsight Systems 中，時間軸頂端會出現一條名為 **NVTX** 的彩色橫帶，清楚標示出當前 CPU 與 GPU 分別正在處理哪一個任務。

---

## 🧭 下一步預告 (Next Steps)
* **第三站**：深入 CPU 前處理效能瓶頸！重構 `src/preprocessor.cpp` 的三層影像迴圈，學習快取友善（Cache-Friendly）走訪與 **SIMD / RISC-V Vector (RVV)** 向量化思考。
