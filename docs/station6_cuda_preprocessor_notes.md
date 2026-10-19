# 第六站：GPU 原生自訂 CUDA Kernel 前處理 (Station 6)

> 本文件記錄如何徹底消滅 CPU 前處理瓶頸，將「雙線性插值縮放 (Resize) + BGR 轉 RGB + 常態化 (Normalize) + HWC 轉 NCHW 佈局轉置」全部融合為單一極速 GPU CUDA Kernel，使前處理延遲從 0.23 ms 暴降至 0.009 ms (9.3 微秒)。

---

## 一、 六大架構演進效能里程碑

| 階段 / 架構 | 前處理耗時 | GPU 推論耗時 | 總端到端延遲 | 前處理吞吐量 | 核心技術里程碑 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **初始基準 (CPU)** | 4.36 ms | 45.0 ms | 49.36 ms | ~230 FPS | OpenCV CPU 多重拷貝 + ORT CPU |
| **第二站 (CUDA EP)** | 4.36 ms | 1.34 ms | 5.70 ms | ~230 FPS | ORT CUDA 串流非同步推論，發現 27.4% 轉置損耗 |
| **第三站 (算子融合 CPU)** | 0.23 ms | 1.34 ms | 1.57 ms | 4,340 FPS | 單次遍歷、常數摺疊、Raw Pointer、無多餘分配 |
| **第四站 (ORT-TRT EP)** | 0.23 ms | 0.51 ms | 0.74 ms | 4,340 FPS | 圖融合、FP16 Tensor Core、離線消除轉置 |
| **第五站 (Native TRT 10)** | 0.23 ms | 0.37 ms | 0.60 ms | 4,340 FPS | 脫離 ORT 包裝、原生 enqueueV3、雙串流 3000+ FPS |
| **第六站 (GPU CUDA Kernel)** | **0.0093 ms** | **0.37 ms** | **0.42 ms** | **>100,000 FPS** | **端到端顯存直通、單一 Kernel 算子融合、完全解放 CPU** |

---

## 二、 為什麼需要 GPU 前處理？

在第五站中，我們的 Native TensorRT GPU 推論耗時已經壓低到 0.37 ms。
然而，此時前處理仍然在 CPU 上執行（耗時 0.23 ms）：
* **阿姆達爾定律 (Amdahl's Law)**：前處理耗時佔了整個端到端時間的近 40% (0.23 / 0.60)。
* **資料傳輸開銷 (H2D Overhead)**：CPU 前處理完後的張量大小是 `1 * 3 * 224 * 224 * sizeof(float) = 602,112 bytes` (約 600 KB)，每幀都需要經由 PCIe 拷貝至 GPU。
* **高解析度相機場景**：如果相機輸入是 4K (3840x2160) 或 1080p 串流，CPU 做雙線性插值與色彩轉換會吃滿 CPU 核心，造成嚴重的 Pipeline 堵塞。

### 解決方案：GPU 端到端顯存直通 (Zero CPU Preprocessing)
1. 原始相機影像（或 DMA 緩衝區）直接傳輸至 GPU 顯存。
2. 啟動自訂 2D CUDA Kernel：在 GPU 上直接完成取樣、縮放、色彩重排、標準化，並直接輸出至 TensorRT 的輸入緩衝區 `d_input`。
3. 緊接著啟動 TensorRT `enqueueV3`，中間完全不經過 CPU，也不需要任何額外的暫存顯存。

---

## 三、 自訂 CUDA Kernel 算子融合架構設計

### 1. 2D 線程網格 (Grid & Block) 配置
ResNet-18 的輸入維度是 `224 * 224`：
* 我們採用 2D Block 架構：`dim3 block(16, 16)`（每個 Block 有 256 個 GPU 線程）。
* 網格維度計算：
  * `grid_x = (224 + 16 - 1) / 16 = 14`
  * `grid_y = (224 + 16 - 1) / 16 = 14`
  * 總共有 `14 * 14 = 196` 個 Blocks，總計 50,176 個 GPU Threads。
* 每個 Thread 剛好負責計算**目標輸出影像的一個像素 (x, y)** 的 R、G、B 三個通道！

### 2. 座標對齊與雙線性插值 (Bilinear Interpolation)
為了避免縮放後的幾何形狀產生偏移，我們使用中心對齊 (Half-pixel center alignment)：
* 映射公式：`src_x = (x + 0.5) * (src_width / dst_width) - 0.5`
* 計算鄰近四個整數座標：
  * `x_low = floor(src_x)`, `x_high = min(x_low + 1, src_width - 1)`
  * `y_low = floor(src_y)`, `y_high = min(y_low + 1, src_height - 1)`
* 雙線性權重係數：
  * `wx = src_x - x_low`
  * `wy = src_y - y_low`
* 取樣四個點的像素值並進行加權內插：
  `val = (1 - wy) * ((1 - wx) * P00 + wx * P01) + wy * ((1 - wx) * P10 + wx * P11)`

### 3. FMA (Fused Multiply-Add) 常態化數學變換
ImageNet 標準常態化原本是：
`output = (pixel / 255.0 - mean) / std`

在硬體底層，除法 (`/`) 比乘法運算器慢好幾倍。我們改用乘加融合形式 (FMA)：
`output = pixel * inv_scale - bias_term`
其中：
* `inv_scale = 1.0 / (255.0 * std)`
* `bias_term = mean / std`

三通道係數在編譯期計算完畢：
* R: `inv_scale = 1.0 / (255.0 * 0.229) = 0.017124f`, `bias = 0.485 / 0.229 = 2.117904f`
* G: `inv_scale = 1.0 / (255.0 * 0.224) = 0.017507f`, `bias = 0.456 / 0.224 = 2.035714f`
* B: `inv_scale = 1.0 / (255.0 * 0.225) = 0.017429f`, `bias = 0.406 / 0.225 = 1.804444f`

每個像素的常態化運算只需一個指令 `fmaf(val, inv_scale, -bias)`，在 GPU CUDA Core 上單週期即可完成！

### 4. 佈局直接輸出為 NCHW (Planar)
OpenCV 讀取的記憶體是 HWC (BGR packed: B0 G0 R0 B1 G1 R1 ...)。
TensorRT 要求的輸入格式是 NCHW (全 R, 全 G, 全 B)：
* R 通道記憶體位址偏移：`0 * 224 * 224 + y * 224 + x`
* G 通道記憶體位址偏移：`1 * 224 * 224 + y * 224 + x`
* B 通道記憶體位址偏移：`2 * 224 * 224 + y * 224 + x`

線程在計算出 `norm_r`, `norm_g`, `norm_b` 後，直接將數值寫入上述三個偏移位址，**同時消滅了色彩空間轉換 (BGR->RGB) 與記憶體佈局轉置 (HWC->NCHW)**！

---

## 四、 核心程式碼解析

### 1. CUDA Kernel 實作 ([src/cuda_preprocessor.cu](file:///c:/Users/Richard/Desktop/C++/onnx-resnet-inference/src/cuda_preprocessor.cu))
```cuda
__global__ void preprocess_kernel_fused(
    const uint8_t* __restrict__ src,
    int src_w, int src_h, int src_step,
    float* __restrict__ dst,
    int dst_w, int dst_h,
    float scale_x, float scale_y
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    // 1. 座標中心對齊
    float src_x = (x + 0.5f) * scale_x - 0.5f;
    float src_y = (y + 0.5f) * scale_y - 0.5f;

    src_x = fmaxf(0.0f, fminf(src_x, (float)(src_w - 1)));
    src_y = fmaxf(0.0f, fminf(src_y, (float)(src_h - 1)));

    int x_low = (int)src_x;
    int y_low = (int)src_y;
    int x_high = min(x_low + 1, src_w - 1);
    int y_high = min(y_low + 1, src_h - 1);

    float wx = src_x - x_low;
    float wy = src_y - y_low;

    // 2. 雙線性插值取樣 (BGR)
    // ... 四個角落像素權重計算 ...

    // 3. 乘加常態化 (FMA) 與色彩通道變換
    float norm_r = val_r * INV_SCALE_R - BIAS_R;
    float norm_g = val_g * INV_SCALE_G - BIAS_G;
    float norm_b = val_b * INV_SCALE_B - BIAS_B;

    // 4. 直接寫入 NCHW 佈局
    int plane_size = dst_w * dst_h;
    int pixel_idx = y * dst_w + x;

    dst[0 * plane_size + pixel_idx] = norm_r;
    dst[1 * plane_size + pixel_idx] = norm_g;
    dst[2 * plane_size + pixel_idx] = norm_b;
}
```

---

## 五、 實測效能驗證

使用 1504x846 解析度影像進行 100 次迭代測試：

```text
========================================
   Native TensorRT Engine (Option B - Pure C++)
========================================

[Station 6 GPU CUDA Preprocessor Alone]
Avg Kernel Time: 0.0093696 ms (106,728 FPS)

[Native TensorRT Benchmark Mode]
Runs: 100
Avg:  0.367184 ms
FPS:  2723.43 FPS

[Station 6 Pure GPU Pipeline (CUDA Preprocess + TRT FP16 + D2H)]
Runs: 100
Avg:  0.420906 ms
FPS:  2375.83 FPS

[Native TensorRT Multi-Threaded Async Benchmark (2 Streams)]
Total time for 100 runs: 33.1338 ms
Throughput: 3018.07 FPS

Top-5 Predictions (Verified with GPU CUDA Preprocessor):
    1. unknown (score: 11.3594, id: 284)
    2. Egyptian cat (score: 10.1406, id: 285)
    3. tabby cat (score: 8.46094, id: 281)
    4. Chihuahua (score: 7.95703, id: 151)
    5. unknown (score: 7.57422, id: 158)
```

### 關鍵效能分析：
1. **前處理時間極限突破**：
   * Station 1 (OpenCV CPU): 4.36 ms
   * Station 3 (Fused CPU): 0.23 ms (18.2 倍加速)
   * Station 6 (CUDA Kernel): **0.0093 ms** (24.7 倍於 Station 3，**468 倍於 Station 1**)！
2. **端到端純 GPU 吞吐量**：
   * 包含自訂 Kernel 前處理、TensorRT FP16 推論、輸出拷貝 D2H，整套單串流管線平均耗時僅 **0.42 ms**，單卡吞吐量高達 **2,375 FPS**！
3. **推論準確度 100% 一致**：
   * Top-1 預測為貓咪類別（id: 284 / 285），信心分數與數值輸出與 Station 3、Station 5 完全對齊。
