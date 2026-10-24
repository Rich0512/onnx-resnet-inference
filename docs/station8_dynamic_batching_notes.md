# Station 8: 動態批次處理與 GPU 吞吐量飽和度壓測 (Dynamic Batching & Throughput Saturation)

在生產環境（如雲端推論伺服器、Triton Inference Server、影像搜尋引擎）中，系統追求的指標往往不僅僅是「單張影像的極限低延遲（Latency）」，更是「伺服器每秒能處理的最大影像數量（Throughput / FPS）」。

當我們以 `batch_size = 1` 執行推論時，GPU 的數千個 CUDA 核心與數百個 Tensor Core 其實處於嚴重的**算力飢餓（Underutilization）**狀態。本專題透過純 C++ 原生 TensorRT 與自訂 3D CUDA Grid Kernel，實現支援動態維度的端到端批次處理 Pipeline，並實測繪製 GPU 吞吐量飽和曲線。

---

## 1. 核心觀念：為什麼 Batch=1 餵不飽 GPU？

### 1.1 記憶體頻寬受限 (Memory Bandwidth Bound) vs. 算力受限 (Compute Bound)
在深度學習卷積層中，每層都需要從 GPU 顯存 (VRAM) 讀取卷積核權重 (Weights)。
- 當 `batch_size = 1` 時：GPU 讀取一次 ResNet-18 的數千萬個權重參數，卻只為了計算 1 張輸入影像。這意味著 GPU 大部分時間都在等待 VRAM 搬移權重資料（算力利用率常低於 20%~30%）。
- 當 `batch_size = 32` 時：GPU 讀取一次權重參數，能同時對 32 張輸入影像進行平行矩陣乘法（GEMM）。權重被載入到高速 Shared Memory / L2 Cache 後被重複利用了 32 次，**運算強度（Arithmetic Intensity = FLOPs / Byte）暴增**，直接將 GPU 推向 Tensor Core 的算力極限。

### 1.2 Kernel 發射與調度開銷佔比
在低 Batch 下，CPU 發射 Kernel 的時間、PCIe 傳輸的固定延遲（Latency Floor）佔了總時間的很大比例；而高 Batch 能將這些固定開銷平攤給 N 張影像，使得「平均每張影像的延遲（Per-Image Latency）」呈現斷崖式下降。

---

## 2. TensorRT Optimization Profile 與動態維度

### 2.1 導出帶有動態維度的 ONNX 模型
在 PyTorch 導出時，必須明確標記批次維度為動態軸：
```python
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=['input_tensor'],
    output_names=['output_tensor'],
    dynamic_axes={
        'input_tensor': {0: 'batch_size'},
        'output_tensor': {0: 'batch_size'}
    },
    opset_version=17
)
```

### 2.2 TensorRT 的最佳化配置 (Optimization Profile)
靜態 TensorRT Engine 在編譯期就固定了維度，無法在執行期更改。若要支援動態維度，必須在 Builder 配置階段建立 `IOptimizationProfile`，定義三個邊界：
- **`min`**：模型允許推論的最小形狀，例如 `(1, 3, 224, 224)`。
- **`opt`**：TensorRT 進行 Kernel Auto-Tuning 的目標形狀，例如 `(8, 3, 224, 224)`。TensorRT 會針對這個 Batch Size 挑選執行效率最高的 Tactic 與 GEMM 演算法。
- **`max`**：允許的最大形狀與內部緩衝區上限，例如 `(32, 3, 224, 224)`。

在 C++ 執行期，只需要在呼叫 `enqueueV3` 之前設定當前 Batch 大小：
```cpp
// 執行期動態設定當前維度
context_->setInputShape("input_tensor", nvinfer1::Dims4(batch_size, 3, 224, 224));
```

---

## 3. 自訂 3D CUDA Grid 批次前處理 Kernel

在 Station 6 中，我們設計了 2D Grid 的 CUDA Kernel 來處理單張影像。在 Station 8，我們要一口氣在 GPU 上平行處理 $N$ 張圖像，絕不能用 CPU 跑迴圈依序啟動 $N$ 次 Kernel。

我們將 CUDA Grid 擴充為 **3D Grid**：
- `gridDim.x = (dst_w + 15) / 16 = 14`
- `gridDim.y = (dst_h + 15) / 16 = 14`
- `gridDim.z = batch_size`
- `blockDim.x = 16, blockDim.y = 16, blockDim.z = 1`

### 3.1 核心 Kernel 實作重點
```cuda
__global__ void preprocess_batch_kernel(
    const uint8_t* __restrict__ src_batch,
    int src_w, int src_h, int src_stride,
    float* __restrict__ dst_batch,
    int dst_w, int dst_h,
    float scale_x, float scale_y,
    int batch_size)
{
    // 利用 blockIdx.z 直接定位當前 Thread 負責的 batch 索引
    int b = blockIdx.z;
    if (b >= batch_size) return;

    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_x >= dst_w || dst_y >= dst_h) return;

    // 計算該 batch 影像在記憶體中的起點
    size_t src_image_size = (size_t)src_stride * src_h;
    const uint8_t* src = src_batch + b * src_image_size;

    size_t dst_batch_stride = 3 * dst_w * dst_h;
    float* dst = dst_batch + b * dst_batch_stride;

    // 雙線性插值與標準化 (RGB + NCHW 輸出)
    // ...
}
```
透過這種設計，硬體排程器（GigaThread Engine）會在 GPU 的數十個 SM 上同時調度數百個 Block，一次將全部 Batch 的影像前處理在微秒內同時搞定！

---

## 4. 實測數據：GPU 吞吐量飽和度曲線 (Saturation Benchmark)

硬體環境：NVIDIA RTX 筆電 GPU，CUDA 12.6，TensorRT 10.7 FP16。
以 100 次預熱 + 1,000 次推論取平均值，測試不同 Batch Size 下的表現：

| Batch Size | Batch 延遲 (ms) | 每張影像延遲 (ms/img) | 系統吞吐量 (FPS) | 加速倍率 (Speedup) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.940 ms | 0.9404 ms | **1,063.4 FPS** | 1.00x |
| **2** | 0.805 ms | 0.4023 ms | **2,485.7 FPS** | 2.34x |
| **4** | 0.977 ms | 0.2443 ms | **4,092.8 FPS** | 3.85x |
| **8** | 1.529 ms | 0.1911 ms | **5,232.2 FPS** | 4.92x |
| **16** | 2.987 ms | 0.1867 ms | **5,356.3 FPS** | 5.04x |
| **32** | 5.278 ms | 0.1649 ms | **6,062.8 FPS** | **5.70x** |

---

## 5. 數據深入解讀：工程甜蜜點 (Sweet Spot)

從數據可以觀察到極為關鍵的高效能計算規律：

1. **Batch 1 到 4 的暴風成長階段**：
   - Batch 從 1 提升至 4，整批推論延遲幾乎沒有增加（從 0.94 ms 只微幅變為 0.97 ms）！
   - 這直接證明：在 Batch 1 時，GPU 運算核心有將近 75% 的時間是閒置在等待排程。因此增加到 Batch 4，幾乎只是在「填滿閒置的硬體計算流水線」，讓吞吐量從 1,063 FPS 直衝 **4,092 FPS**。

2. **Batch 8 到 16 的飽和過渡區（甘甜點 Sweet Spot）**：
   - 在 Batch = 8 時，Throughput 達到 **5,232 FPS**，每張影像只需 0.19 ms。
   - 當 Batch 增加到 16，Throughput 為 **5,356 FPS**，增長幅度開始趨於平緩。
   - 這代表 GPU 的 SM 佔用率（Occupancy）與記憶體頻寬已接近飽和，硬體從「延遲主導（Latency Bound）」正式切換為「計算密集飽和（Compute Bound）」。

3. **系統架構權衡（Trade-off: Throughput vs. Latency）**：
   - **低延遲即時系統（如自駕車即時辨識、即時機器人感知）**：應選擇 `batch_size = 1 ~ 2`，以換取小於 1 ms 的極致低響應延遲。
   - **高吞吐離線/服務系統（如圖片檢索後端、相片分類伺服器）**：應選擇 `batch_size = 8 ~ 16`，用稍微增加的整批響應時間（1.5 ~ 3.0 ms），換取高達 **5,300+ FPS** 的近 5.7 倍伺服器處理能力！
