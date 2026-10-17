# 第三站：CPU 前處理極限效能優化與 SIMD / 向量化重構 (Station 3 Notes)

> 本文件記錄了從現有 `onnx-resnet-inference` 專案出發，針對前處理 CPU 瓶頸，進行 **算子融合（Operator Fusion）**、**單次走訪（Single-Pass Traversal）** 與 **硬體 FMA 數學優化** 的重構過程與底層原理。

---

## 🎯 職缺技能要求與本站學習對照

| 職缺技能要求 (JD) | 本站專案代碼 | 掌握的底層核心觀念 |
| :--- | :--- | :--- |
| **High Performance C++ & Memory Optimization** | `src/preprocessor.cpp`<br>`include/preprocessor.h` | • 徹底消除暫存 `cv::Mat` 頻繁向 OS 申請 Heap 的開銷<br>• 指標直接走訪（Raw Pointer）代替 `cv::Mat::at<float>`<br>• 快取友善循序存取（Cache-Friendly Sequential Access） |
| **Operator Fusion (算子融合思維)** | `preprocessor::preprocess_image` | • 將 BGR轉RGB、uint8轉float、歸一化、HWC轉CHW **5次記憶體讀寫融合成單次走訪 (Single-Pass)**<br>• 記憶體流量直接縮減 80% |
| **Hardware Arithmetic & SIMD Optimization** | AVX2 / FMA 指令思維 | • 算術代數簡化：以乘法常數代替昂貴的浮點除法指令（`divss`）<br>• 將歸一化公式轉為硬體單週期 **FMA (Fused Multiply-Add: `x * scale + bias`)**<br>• 連續無分支記憶體排布，賦能編譯器自動向量化（Auto-Vectorization） |

---

## 🧠 一、 原始代碼的四大效能黑洞剖析

在原始的 [src/preprocessor.cpp](file:///c:/Users/Richard/Desktop/C++/onnx-resnet-inference/src/preprocessor.cpp) 中，耗時高達 **4.3 ms**（遠遠慢於 GPU 推論的 1.3 ms），其背後有四大致命傷：

```cpp
// 1. 轉 RGB 加上轉 float (耗費一次 600KB 記憶體申請與全矩陣遍歷)
cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
rgb.convertTo(rgb, CV_32F, 1.0f / 255.0f);

// 2. 切成 3 個通道 (耗費三次 200KB 記憶體申請與全矩陣拷貝)
cv::Mat channels[3];
cv::split(rgb, channels);

// 3. 運算子重載 (每次 +-*/ 都偷偷配置暫存 Mat，引發大量 Heap 分配)
channels[0] = (channels[0] - MEAN_R) / STD_R;
channels[1] = (channels[1] - MEAN_G) / STD_G;
channels[2] = (channels[2] - MEAN_B) / STD_B;

// 4. 效能地獄：三重迴圈 + at<float> + push_back
input_data.clear();
input_data.reserve(3 * input_width * input_height);
for (int c = 0; c < 3; c++) {
    for (int h = 0; h < input_height; h++) {
        for (int w = 0; w < input_width; w++) {
            input_data.push_back(channels[c].at<float>(h, w));
        }
    }
}
```

### 1. 記憶體頻寬殺手（Memory Bandwidth Bottleneck）
* **同一張圖片被讀寫了整整 5 次**：
  1. `cvtColor`：讀取 BGR，寫入 RGB（第 1 遍）
  2. `convertTo`：讀取 uint8，寫入 float32（第 2 遍）
  3. `split`：讀取交錯 RGB，寫入三個獨立 channels（第 3 遍）
  4. 減均值除標準差：讀取通道，計算後寫回（第 4 遍）
  5. 三重迴圈：讀取通道，寫入 vector（第 5 遍）
* **真相**：CPU 大多數時間不是在算數學，而是在等待資料在 L3 Cache 與 DRAM 之間搬運。

### 2. 頻繁 Heap 動態記憶體分配（Memory Thrashing）
* `cv::Mat rgb`、`channels[3]` 以及矩陣加減乘除的暫存物件，每次呼叫都透過 `malloc` 向 OS 申請數百 KB 記憶體，算完立即釋放，產生龐大的 System Call 開銷。

### 3. `cv::Mat::at<float>(h, w)` 的隱藏成本
* 每次呼叫 `at(h, w)` 都需要計算行跨度乘法：`h * step + w * sizeof(float)`。
* 且在 Debug 模式或未完全關閉檢查時，每次都會觸發邊界檢查（Bounds Checking）。

### 4. `push_back` 阻礙編譯器向量化
* 每次寫入都伴隨 `vector::size` 的更新與容量檢查，編譯器根本無法將其展開為 AVX2 / SIMD 向量暫存器操作。

---

## 🚀 二、 重構方案：算子融合與單次走訪 (Operator Fusion & Single-Pass)

### 1. 核心哲學：當像素在快取中時，一次算完！
當 CPU 從 `resized` 讀取一個像素的 `(B, G, R)` 時，這 3 個 bytes 已經熱騰騰地躺在 CPU L1 Data Cache 中了。我們可以直接完成所有的計算並直接寫入最終目標記憶體。

### 2. 代數簡化：除法轉 FMA（乘加指令）
原本的 ImageNet 歸一化公式：
$$\text{Output} = \frac{\frac{\text{Pixel}}{255.0} - \text{Mean}}{\text{Std}}$$

我們可以在**編譯期（`constexpr`）**預先算出兩個浮點常數：
$$\text{Scale} = \frac{1.0}{255.0 \times \text{Std}}$$
$$\text{Bias} = -\frac{\text{Mean}}{\text{Std}}$$

原本複雜的除法直接化簡為**一次乘法 + 一次加法**：
$$\text{Output} = \text{Pixel} \times \text{Scale} + \text{Bias}$$

在現代 x86 CPU 上，這會被編譯為硬體原生的 **`vfmadd`（Fused Multiply-Add）** 指令，**1 個 Clock 週期即可完成**！

---

## 🛠️ 三、 重構後代碼實作解析

```cpp
// 1. 編譯期預先計算 FMA 係數
constexpr float scale_r = 1.0f / (255.0f * STD_R);
constexpr float bias_r  = -MEAN_R / STD_R;
constexpr float scale_g = 1.0f / (255.0f * STD_G);
constexpr float bias_g  = -MEAN_G / STD_G;
constexpr float scale_b = 1.0f / (255.0f * STD_B);
constexpr float bias_b  = -MEAN_B / STD_B;

const int total_pixels = input_width * input_height;

// 2. 一次性鎖定最終記憶體大小（複用 Pinned Memory）
input_data.resize(3 * total_pixels);

// 3. 取得 CHW 三個通道的平坦記憶體指標（Raw Pointers）
float* r_ptr = input_data.data();
float* g_ptr = r_ptr + total_pixels;
float* b_ptr = g_ptr + total_pixels;

// 4. 連續記憶體單次走訪（Single-Pass）
const uint8_t* src_ptr = resized.data;

for (int i = 0; i < total_pixels; ++i) {
    uint8_t b = *src_ptr++;
    uint8_t g = *src_ptr++;
    uint8_t r = *src_ptr++;

    *r_ptr++ = static_cast<float>(r) * scale_r + bias_r;
    *g_ptr++ = static_cast<float>(g) * scale_g + bias_g;
    *b_ptr++ = static_cast<float>(b) * scale_b + bias_b;
}
```

### 優化戰果對比：
* **記憶體讀寫次數**：5 遍 ➔ **1 遍**（減少 80% 記憶體頻寬）。
* **動態記憶體申請**：數次 Heap 分配 ➔ **0 次**（全場複用傳入的 Buffer）。
* **計算效率**：複雜浮點除法 ➔ **FMA 單週期指令**。
* **向量化潛力**：連續指標無分支寫入，現代編譯器（MSVC / GCC / Clang）在 `/O2` 下會自動展開為 **AVX2 / SSE 向量暫存器操作**。

---

## 🎙️ 四、 面試應對指南（Interview Playbook）

### 面試必考題：
> **「在端到端推論系統中，如何解決 CPU 前處理成為效能瓶頸的問題？」**

### 滿分回答結構：
1. **指出問題**：
   * 「在 Profile 過程中，我發現 GPU 算力很強（1.3 ms），但傳統 OpenCV 多步驟前處理（cvtColor, convertTo, split, normalize）花費了 4.3 ms，導致嚴重的 GPU 飢餓（Starvation）。」
2. **闡述解決方案**：
   * 「我實施了**算子融合（Operator Fusion）與單次遍歷（Single-Pass）**：
     * 將原本需進出記憶體 5 次的流程，融合成一次循序指標走訪，**直接將影像資料在 CPU L1 快取中一氣呵成轉換為 CHW 佈局並寫入 Pinned Memory**。
     * 將標準化公式化簡為 `pixel * scale + bias`，運用硬體 **FMA 指令** 取代昂貴的除法指令。
     * 消滅所有中途暫存的 `cv::Mat` 物件，達成 **Zero Heap Allocation**。」
3. **總結收益**：
   * 「大幅釋放了記憶體頻寬壓力，使 CPU 能及時餵飽 GPU 佇列，徹底消除硬體時間軸上的 Idle Gaps。」
