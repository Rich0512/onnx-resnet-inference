# 第一站：Runtime 系統架構與記憶體管理學習筆記 (Station 1 Notes)

> 本文件記錄了從現有 `onnx-resnet-inference` 專案出發，對標 AI Compiler / AI Systems 職缺中 **Runtime Systems** 與 **Hardware-Software Interface** 所建立的底層核心觀念。

---

## 🎯 職缺需求與今日學習對照

| 職缺技能要求 (JD) | 今日專案對照代碼 | 掌握的底層核心概念 |
| :--- | :--- | :--- |
| **Runtime Systems & Memory Management** | `include/pinned_allocator.h`<br>`src/inference.cpp` | • 作業系統虛擬記憶體分頁 (Paging)<br>• 晶片硬體 DMA (Direct Memory Access) 限制<br>• Pinned Memory (`cudaHostAlloc`) 與鎖頁原理<br>• RAII 資源生命週期管理與 C++ Allocator 規範 |
| **HW-SW Interface & High-Throughput** | `src/inference.cpp`<br>`src/main.cpp` | • Zero-Copy Tensor 封裝 (`Ort::Value`)<br>• 非同步硬體任務佇列 (`cudaMemcpyAsync`)<br>• 多 CUDA Streams 平行排程與管線重疊 (Overlap) |

---

## 🧠 一、 作業系統與硬體底層核心機制

### 1. Pageable Memory（可分頁記憶體）vs. Pinned Memory（鎖頁記憶體）
* **Pageable Memory（普通 `malloc` / `new`）**：
  * 使用者看到的是虛擬位址。
  * 作業系統（OS）會根據系統負載隨時將實體分頁移位，甚至換頁至硬碟（Swap/Pagefile）。
  * CPU 存取普通記憶體時有硬體 **MMU（記憶體管理單元）** 與 OS 核心在背後處理 Page Fault，因此安全無虞。
* **Pinned Memory（`cudaHostAlloc` / 鎖頁記憶體）**：
  * 向 OS 宣告將實體分頁標記為 `PG_locked`。
  * **OS 保證：絕對不換頁至硬碟，實體位址（Physical Address）終生固定不變**。

### 2. 為什麼加速器（GPU / NPU）傳輸必須使用 Pinned Memory？
* 顯卡上的 **DMA（Direct Memory Access）控制器** 是純硬體電路，**只認實體位址，且沒有 MMU 與 OS 介入保護**。
* **傳輸普通記憶體的下場**：
  * DMA 無法直接讀寫普通記憶體（怕傳到一半分頁被 OS 換掉導致資料損毀或藍屏）。
  * **CUDA Driver 被迫使用二傳手（Staging Buffer）**：
    1. CPU 親自執行 `memcpy` 將資料複製進 Driver 的秘密 Pinned Staging Buffer。
    2. DMA 才從 Staging Buffer 搬移至 GPU VRAM。
    3. **代價**：多花一次 CPU 記憶體拷貝，CPU 執行緒被死死卡住，無法做到非同步。
* **傳輸 Pinned Memory 的優勢**：
  * DMA 控制器直接拿著實體位址直抽資料至顯卡（**零額外中間拷貝，達到 PCIe 理論頻寬極限**）。

### 3. 為什麼不能在每次推論迴圈都 `new` / `cudaHostAlloc`？
* **時間開銷（Latency Overhead）**：
  * 普通 `malloc` 耗時約數十奈秒（ns）。
  * `cudaHostAlloc` 必須透過系統呼叫（System Call）打進 OS 核心修改分頁表，一次耗時高達 **0.5 ms ~ 1.0 ms**。
  * 相比之下，ResNet-18 的 GPU 推論本身只要 **0.53 ms**。若每次迴圈都重新配置，效能直接暴跌 3 倍！
* **系統工程黃金準則**：
  > **「啟動時預先配置，迴圈中終生複用（Allocate up-front, reuse in the loop）。」**

---

## 🛠️ 二、 C++ 系統層與語法細節

### 1. `PinnedAllocator` 的設計哲學
```cpp
template <typename T>
struct PinnedAllocator {
    using value_type = T; // 對標準庫亮出型別身分證，符合 C++ Allocator 規範
    
    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        #ifdef USE_CUDA
            // 第一志願：申請最高速的 Pinned Memory
            cudaError_t err = cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault);
            if (err != cudaSuccess) {
                // 第二志願（Fallback）：降級為普通 heap 記憶體，防止程式崩潰
                ptr = static_cast<T*>(::operator new(n * sizeof(T)));
            }
        #else
            ptr = static_cast<T*>(::operator new(n * sizeof(T)));
        #endif
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) {
        #ifdef USE_CUDA
            cudaFreeHost(ptr); // 歸還實體鎖頁記憶體
        #else
            ::operator delete(ptr);
        #endif
    }
};
```
* **RAII（資源獲取即初始化）**：
  * 將 `cudaHostAlloc` 與 `cudaFreeHost` 封裝進 Allocator，搭配 `std::vector` 使用。
  * 離開大括號或遇到 C++ 異常（Exception）時，自動呼叫解構子釋放，**徹底杜絕鎖頁實體記憶體洩漏**。
* **`::operator new` vs. `new T[n]`**：
  * `::operator new` 僅負責分配純粹的實體記憶體空間，不浪費 CPU 週期呼叫建構子初始化無意義的預設值。

### 2. 零拷貝（Zero-Copy）Tensor 包裝 (`src/inference.cpp`)
```cpp
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, 
    const_cast<float*>(input_data), 
    input_size, 
    input_node_dims.data(), 
    input_node_dims.size());
```
* **底層真相**：
  * 600 KB 的圖片陣列（$1 \times 3 \times 224 \times 224 \times 4 \text{ bytes} \approx 602 \text{ KB}$）**完全沒有被拷貝**。
  * `Ort::Value` 僅是一個幾十 bytes 的輕量結構體，內部儲存 `input_data` 指標與維度標籤（如同在箱子上貼便條紙）。
* **`const_cast<float*>` 的角色**：
  * 因第三方 C++ API 缺乏 const-correctness，為了在零拷貝前提下傳遞唯讀輸入，使用 C++ 內建的 `const_cast` 去除 const 屬性。

---

## ⚡ 三、 非同步並行與硬體串流排程

### 1. `cudaMemcpy` vs. `cudaMemcpyAsync`
* **`cudaMemcpy`（同步阻塞）**：CPU 執行緒原地凍結，死等 PCIe 傳輸完畢。
* **`cudaMemcpyAsync`（非同步）**：CPU 僅花費 0.5 微秒將傳輸工單丟進指定 Stream，隨即返回執行後續 C++ 代碼；硬體 DMA 控制器在背景自主傳輸。

### 2. 多 CUDA Streams（多線道高速公路）
```cpp
Ort::RunOptions run_options;
std::string stream_ptr_str = std::to_string((unsigned long long)stream);
run_options.AddConfigEntry("gpu_compute_stream", stream_ptr_str.c_str());
session_->Run(run_options, ...);
```
* **CUDA Stream**：GPU 硬體內部的命令佇列（Command Queue）。
* **並行重疊（Overlap）**：
  * Stream 1：GPU Tensor Core 正在運算第 1 張圖片。
  * Stream 2：DMA 控制器正在透過 PCIe 傳輸第 2 張圖片。
  * 傳輸與計算同時發生，硬體管線利用率最大化。

---

## 🔍 四、 編譯器與程式碼嚴謹度細節

1. **雙重指標傳遞**：
   * `cudaHostAlloc` 的簽名為 `cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags)`。
   * 欲修改結構體內的指標成員 `data`，必須傳入其位址 `&data`。
2. **作用域與具體型別**：
   * 非模板類別中需使用具體型別 `sizeof(float)`，模板類別中才使用 `sizeof(T)`。

---

## 🧭 下一步預告 (Next Steps)
* **第二站**：使用 **NVIDIA Nsight Systems (`nsys`)** 親眼觀察多 Stream 在硬體時間軸上的真實重疊狀況（Timeline Profiling）。
* **第三站**：重構 `src/preprocessor.cpp` 的三層迴圈，練習指標走訪優化與 **RISC-V Vector (RVV) / SIMD** 向量化思考。
