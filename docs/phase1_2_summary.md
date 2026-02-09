# Phase 1 & 2 總複習：從零到高性能推論引擎

這份文件旨在複習我們這段時間以來的所有技術細節與優化里程碑。建議您在進入 API 服務化 (Phase 3) 前仔細閱讀，鞏固基礎。

## 1. 專案演進與里程碑 (Evolution)

我們見證了推論速度從 150 FPS 飆升至 1800+ FPS 的過程。

| 階段 | 核心技術 | 效能提升 | 關鍵瓶頸與解決 |
|---|---|---|---|
| **Phase 0** | **Base (CPU)** | 1x (150 FPS) | 純 CPU 運算。瓶頸在運算力不足。 |
| **Phase 1** | **GPU (CUDA)** | 4.2x (630 FPS) | 瓶頸轉移到資料傳輸 (PCIe) 與 Kernel Launch。 |
| **Phase 2a** | **TensorRT** | 6.6x (990 FPS) | 透過 Layer Fusion 和 Kernel Auto-tuning 優化 GPU 運算。 |
| **Phase 2b** | **Pinned Memory** | 10.6x (1600 FPS) | **瓶頸**: Pageable Memory Copy 慢。<br>**解法**: 使用 `cudaHostAlloc` 鎖定記憶體，啟用 DMA。 |
| **Phase 2c** | **Buffer Reuse** | **12.5x (1876 FPS)** | **瓶頸**: `cudaHostAlloc` 分配太慢 (1ms)。<br>**解法**: 預先分配並重複使用，消除分配時間。 |
| **Phase 2d** | **Async Streams** | Saturation | **瓶頸**: GPU 計算飽和 & CPU 同步。<br>**解法**: 多 Stream 並行 (在大負載下才顯著)。 |

---

## 2. 關鍵技術回顧 (Technology Deep Dive)

### 2.1 ONNX Runtime 架構
*   **Session**: 載入模型，管理推理環境 (`Ort::Session`)。
*   **Execution Provider (EP)**: 決定由誰來執行運算 (CPU, CUDA, TensorRT)。
*   **Memory Info**: 描述數據在哪裡 (CPU, GPU, Pinned)。

### 2.2 記憶體優化 (Pinned Memory)
我們寫了一個自定義分配器 `PinnedAllocator`：
```cpp
// 簡單來說：
// malloc/new -> 系統記憶體 (Pageable)，GPU 存取要經過 CPU copy。
// cudaHostAlloc -> 鎖定記憶體 (Pinned)，GPU 可以直接 DMA 抓取 (Zero-Copy)。

template <typename T>
struct PinnedAllocator {
    T* allocate(size_t n) {
        T* ptr;
        cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault); // 魔法在這
        return ptr;
    }
    // ...
};
```
**代價**: 分配速度慢。
**對策**: Buffer Reuse (只分配一次)。

### 2.3 異步推論 (Async Inference)
我們試圖打破「做完一個再做下一個」的模式。
```cpp
// 傳統 (Sequential)
Preprocess -> Inference -> Preprocess -> Inference

// 異步 (Pipeline)
Stream 1: Inference [A]
CPU     : Preprocess [B]  <-- 重疊執行 (Overlap)
```
雖然在 ResNet18 (小模型) 上提升不明顯，但在 API Server 中這將是處理高併發請求的關鍵。

---

## 3. C++ 實戰技巧 (Coding Skills)

在這專案中，我們運用了許多 Modern C++ 技巧：
1.  **RAII**: 使用 `std::vector` 和 `std::unique_ptr` (雖然 `PinnedAllocator` 是手動管理的，但封裝在 vector 裡)。
2.  **Template**: `PinnedAllocator<T>` 和 `run<Alloc>()` 讓我們能通用化程式碼。
3.  **Namespace**: `inference`, `preprocessor` 分離關注點。
4.  **CLI Parsing**: 使用簡單的 `check_cmd_option` 處理 `--gpu`, `--benchmark`。
5.  **Benchmarking**: 使用 `std::chrono` 進行精確計時。

---

## 4. 展望 Phase 3 (API Service)

現在我們有一個強大的「引擎」(Engine)。
Phase 3 的目標是造「車殼」(Web Server)，讓使用者可以透過網路 (HTTP REST API) 來呼叫這個引擎。

**挑戰**:
1.  **Concurrency**: Web Server 是多執行緒的，我們的 Engine 是 Thread-Safe 的嗎？
    *   ONNX Runtime `Run` 是 Thread-Safe 的。
    *   但我們的 `BufferSize` / `input_buffer` 在 Buffer Reuse 時如果是 global 或 member variable，就會有 Race Condition！
    *   **解法**: 每個 Request (Thread) 要有自己的 Buffer。
2.  **Queueing**: 當請求太多時，如何排隊？(Task Queue + Worker Pool)。

準備好了嗎？我們將把這個 C++ 程式變成一個 Microservice！
