# Phase 2 程式碼深度解析 (Code Deep Dive)

這份文件不講高大上的概念，我們直接看 Code。為什麼要這樣寫？這行 Code 在做什麼？

## 1. 魔法的起源：`pinned_allocator.h`

這是讓 GPU 飛起來的關鍵。即使你看不懂大部分 Template 語法，只要懂這一個核心概念：

```cpp
// pinned_allocator.h
template <typename T>
struct PinnedAllocator {
    // ... 標準 Allocator boilerplates ...

    // 核心：分配記憶體
    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        #ifdef USE_CUDA
            // [關鍵點] 
            // 這裡不用 new，改用 CUDA API。
            // cudaHostAllocDefault: 表示這塊記憶體是 "Page-locked" (鎖定) 的。
            // OS 承諾：我不會把這塊資料移到硬碟 (Swap)。
            // GPU 看到這承諾：太好了，那我可以不透過 CPU，直接拉線過來讀 (DMA)！
            cudaError_t err = cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault);
            if (err != cudaSuccess) {
                throw std::bad_alloc(); // 失敗就報錯
            }
        #else
            // 如果沒開 CUDA，就只是一個普通的 new
            ptr = static_cast<T*>(::operator new(n * sizeof(T)));
        #endif
        return ptr;
    }

    // 核心：釋放記憶體
    void deallocate(T* p, std::size_t) noexcept {
        #ifdef USE_CUDA
            // 必須用專用的 API 來還地
            cudaFreeHost(p);
        #else
            ::operator delete(p);
        #endif
    }
};
```

---

## 2. 應用魔法：`preprocessor.h`

我們怎麼讓 `std::vector` 使用上面那個魔法？

```cpp
// preprocessor.h
namespace preprocessor {

#ifdef USE_CUDA
    // [關鍵語法]
    // std::vector<T, Allocator>
    // 我們告訴 vector：請用我的 PinnedAllocator 來管理記憶體！
    using DataVector = std::vector<float, inference::PinnedAllocator<float>>;
#else
    // 普通版：使用預設分配器 (std::allocator)
    using DataVector = std::vector<float>;
#endif
// ...
}
```

這行 `using` 是一個開關。只要定義了 `DataVector`，後面的程式碼就不用管細節，直接用 `DataVector` 就好。

---

## 3. 避免浪費： Buffer Reuse (`preprocessor.cpp` & `main.cpp`)

我們發現 `cudaHostAlloc` 很貴 (1ms)。所以我們改寫了流程。

**舊版 (慢)**：
```cpp
// 每次呼叫都只有 return，代表內部 new 了一個新的 vector
DataVector result = preprocess(img); 
// -> PinnedAllocator::allocate() 被呼叫 -> 耗時 1ms
```

**新版 (快)**：
```cpp
// 1. 在 main 裡先準備好一個空盤子 (只在程式啟動時 allocate 一次)
preprocessor::DataVector input_buffer; 

// 2. 傳 Reference 進去
preprocess_image(img, input_buffer, ...);

// 3. 在函數內部：
void preprocess_image(..., DataVector& output_buffer) {
    output_buffer.clear(); // 清空內容，但「不釋放記憶體」 (capacity 不變)
    
    // 如果容量夠，這行其實什麼都不做 (Zero allocation)
    // 如果不夠，才會擴充 (只發生在第一次)
    output_buffer.reserve(3 * 224 * 224); 

    // 塞資料...
    // 因為 capacity 夠大，push_back 只是移動指標，超快
    output_buffer.push_back(...); 
}
```

這就是為什麼 Preprocess 時間從 1.0ms 降到 0.48ms 的秘密：**我們避開了 OS 的記憶體管理開銷**。

---

## 4. 引擎介接：`inference.cpp`

Engine 怎麼接收這個特殊的 Vector？

`inference.h` 的設計很巧妙：
```cpp
// 它不接受 vector，只接受 pointer
std::vector<float> run(const float* input_data, size_t input_size);
```

這樣無論你的 vector 是 Pinned, 普通的, 還是其他奇怪的 memory，只要能拿到 `data()` 指標，Engine 都能吃。

在 `inference.cpp` 透過 ONNX Runtime API 告訴底層：
```cpp
// 建立 Memory Info，說盡管這是 CPU 指標，但請放心讀取
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

// 建立 Tensor (這個動作只是包裝指標，不會拷貝數據！)
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, 
    const_cast<float*>(input_data), // 指向 Pinned Memory
    input_size, 
    ...
);

// 當執行 session_->Run() 時：
// ONNX Runtime 發現這個 Tensor 在 Pinned Memory
// 它就會直接啟動 DMA 傳輸到 GPU
```

---

## 5. 異步的奧義：`run_async`

這段最抽象。

```cpp
// run_async
Ort::RunOptions run_options;
// 告訴 ORT：這個任務請丟到這條生產線 (Stream) 上
run_options.AddConfigEntry("gpu_compute_stream", stream_ptr_str.c_str());

session_->Run(run_options, ...);
```

這行 `Run` 送出後，理論上 CPU 應該要馬上返回。
但我們範例中為了拿結果 (`GetTensorMutableData`)，所以還是 blocked 住。

真正的 Pipeline 應用是：
1. Thread 1: `Engine.RunAsync(image1)` -> 返回
2. Thread 1: 去做其他事 (例如讀 image2)
3. Thread 2 (Callback): 收到 image1 完成通知 -> `GetResults`

這比較複雜，目前的 `run_async` 只是第一步：證明我們可以指定 Stream。

---

這就是所有優化的底層邏輯。
不是魔法，而是對 OS、CUDA、C++ 記憶體模型的精確控制。
