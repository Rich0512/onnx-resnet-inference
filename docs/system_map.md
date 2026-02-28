# 系統地圖與資料流 (System Map & Data Flow)

為了在進入 API 服務開發前有一個清晰的藍圖，這裡將我們目前的系統整理成一張高階的資料流動圖。
你可以把它想像成是一條「汽車組裝流水線」。

## 🏗️ 核心架構圖 (Mermaid)

```mermaid
graph TD
    %% 定義節點與樣式
    classDef cpu fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    classDef gpu fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef memory fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    
    subgraph "External"
        U["User / Command Line"]
        IMG["Image File (JPG/PNG)"]
    end

    subgraph "Main Controller (main.cpp)"
        M[main function]:::cpu
        U -- "--tensorrt test.jpg" --> M
        M -- "1. Load Image" --> IMG
    end

    subgraph "Preprocessor (preprocessor.cpp)"
        P[preprocess_image]:::cpu
        M -- "2. Pass cv::Mat" --> P
        P -- "Resize to 224x224" --> P1[Resize]:::cpu
        P1 -- "BGR to RGB" --> P2[Color Convert]:::cpu
        P2 -- "Scale & Normalize" --> P3[Normalize]:::cpu
        P3 -- "HWC to CHW" --> P4[Format]:::cpu
    end

    subgraph "Memory Bank (pinned_allocator.h)"
        BUF[("DataVector<br>Pinned Memory")]:::memory
        P4 -- "3. Write directly to" --> BUF
        note1> "Buffer Reuse: Created once by Main,<br>Reused for every image to save 1ms"
    end

    subgraph "Inference Engine (inference.cpp)"
        E[Engine::run_async]:::cpu
        M -- "4. Trigger Inference" --> E
        
        subgraph "Hardware (GPU / TensorRT)"
            GPU[NVIDIA GPU]:::gpu
            E -- "5. Submit Kernel via Stream" --> GPU
            BUF -- "6. Zero-Copy DMA Transfer" --> GPU
            GPU -- "7. Compute ResNet18" --> GPU
        end
        
        GPU -- "8. Return 1000 Scores" --> OUT[("Scores Array")]:::memory
    end

    M -- "9. Map Scores to Labels" --> L[Label Decoder]:::cpu
    L -- "10. Top 5 Results" --> U
```

---

## 🗺️ 流程白話文解析

1. **進場 (`main.cpp`)**: 
   - 程式啟動，`main` 函式就像總指揮。它先建好一個**純金托盤** (`DataVector input_buffer`，使用 Pinned Memory)。
   - 它從硬碟讀取圖片變成 `cv::Mat` 格式。
2. **切菜 (`preprocessor.cpp`)**: 
   - `main` 把圖片和那個**純金托盤**一起丟給 `preprocess_image` (廚房小弟)。
   - 小弟開始忙碌：縮小圖片、改顏色、作正規化數學運算，最後把處理好的食材一塊塊整齊排在**純金托盤**上。
3. **開火 (`inference.cpp`)**: 
   - `main` 看到托盤裝滿了，就呼叫 `engine.run_async`。
   - `Engine` 是一個外交官，它轉身按下了 ONNX Runtime (GPU 驅動) 的啟動按鈕。
   - **GPU 瞬間啟動！它直接探出頭，從那個純金托盤 (`Pinned Memory`) 上把食材吸過去 (DMA Zero-Copy)，完全不需要 CPU 幫忙拿。**
   - GPU 用極快的速度跑完 TensorRT 模型，吐出 1000 個小數點 (代表 1000 種貓狗的機率)。
4. **出餐 (`main.cpp`)**:
   - `main` 拿到這 1000 個數字，找出最大的 5 個。
   - 查字典 (`imagenet_labels.h`)，把它們翻譯成 "Tabby Cat"、"Tiger Cat" 印在螢幕上。

這就是我們前兩週的心血結晶！
接下来 Phase 3，我們要改寫的只有最上面的 **External** 和 **Main Controller**。
我們不要從 Command Line 讀圖片了，我們要從 **網路 (HTTP Request)** 接收圖片！
