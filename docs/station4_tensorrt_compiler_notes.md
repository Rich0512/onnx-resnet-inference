# 第四站：AI Compiler 與 NVIDIA TensorRT 深度實戰 (Station 4 Notes)

> 本文件記錄了從現有 `onnx-resnet-inference` 專案出發，針對模型載入開銷、算子融合優化與精度加速，實作 **TensorRT Engine 序列化快取（Serialization & Caching）** 與 **FP16 半精度加速** 的完整系統架構與實證數據。

---

## 🎯 職缺技能要求與本站學習對照

| 職缺技能要求 (JD) | 本站專案代碼 | 掌握的底層核心觀念 |
| :--- | :--- | :--- |
| **AI Compiler & Graph Optimization** | `src/inference.cpp`<br>`OrtTensorRTProviderOptions` | • 傳統執行期（ORT/PyTorch）vs. 編譯期最佳化（TensorRT/TVM）<br>• 算子垂直與水平融合（Conv + Bias + BatchNorm + ReLU 熔接）<br>• 徹底消除第二站抓出的 `nchwToNhwcKernel` 轉置開銷 |
| **Engine Serialization & AOT Compilation** | `trt_engine_cache_enable`<br>`trt_engine_cache_path` | • AOT (Ahead-of-Time) 編譯 vs. JIT 暖機代價<br>• 序列化為二進制硬體專用 Engine 檔案（.engine / .plan）<br>• 解決線上生產環境（Production）冷啟動延遲難題 |
| **Mixed Precision & Tensor Core Acceleration** | `trt_fp16_enable = 1` | • FP32 ➔ FP16 半精度加速原理與吞吐量翻倍機制<br>• Tensor Core 硬體架構在 FP16 上的極限發揮 |

---

## 🧠 一、 為什麼 TensorRT 第一次載入要等 21 秒？（AI Compiler 底層黑盒子）

在開啟 `--tensorrt` 時，終端機顯示了兩個截然不同的數字：
* **普通 CUDA 載入**：僅需 **71 ms**（因為它只是單純照著譜彈琴，不做圖最佳化）。
* **TensorRT 首次載入**：耗時高達 **21,733 ms (約 21.7 秒)**！

### 這 21 秒它到底在後台狂忙什麼？
1. **算子大融合（Operator Fusion）**：
   * 原本的 ResNet 包含數十個獨立層（Conv、Bias、BN、ReLU）。
   * TensorRT 將這些算子全部熔接成一個一體成型的超大 Kernel，中間數值直接在 GPU 暫存器流轉，**徹底消滅進出顯存的記憶體頻寬浪費**。
2. **硬體實機試跑競速（Kernel Auto-Tuning，最耗時核心）**：
   * 針對同一個卷積層，NVIDIA 底層有幾十種不同的數學算法（Direct、Winograd、不同尺寸的 Implicit GEMM）。
   * TensorRT 在這 20 秒內，**真的把所有候選算法在你這張特定的顯卡晶片上試跑了一遍**，拿碼錶實測誰最快，選出冠軍硬體排程！
3. **Layout 轉置蒸發術**：
   * 在離線構建期直接把權重與計算圖編譯成 Tensor Core 最愛的 **NHWC**，第二站吃掉 26.8% 的 `nchwToNhwcKernel` 徹底蒸發！

---

## 💾 二、 工業級實戰：Engine 序列化快取（Serialization）

線上伺服器不可能每次重啟都等 21 秒！我們在 [src/inference.cpp](file:///c:/Users/Richard/Desktop/C++/onnx-resnet-inference/src/inference.cpp) 加入了快取開關：

```cpp
// 1. 確保快取目錄存在
std::filesystem::create_directories("./trt_cache");

// 2. 開啟 TensorRT 序列化快取
OrtTensorRTProviderOptions trt_options{};
trt_options.device_id = 0;
trt_options.trt_max_partition_iterations = 1000;
trt_options.trt_min_subgraph_size = 1;
trt_options.trt_fp16_enable = 1;
trt_options.trt_engine_cache_enable = 1;             // 啟用 Engine 快取
trt_options.trt_engine_cache_path = "./trt_cache";  // 指定持久化磁碟路徑
session_options_.AppendExecutionProvider_TensorRT(trt_options);
```

### 產出的實體硬體引擎檔案：
執行後，在 `build/Release/trt_cache/` 產生了：
`TensorrtExecutionProvider_TRTKernel_graph_main_graph_..._fp16_sm86.engine` (約 23.6 MB)
* **`fp16`**：已成功量化為 16 位元半精度。
* **`sm86`**：專門為本機 Ampere 架構微架構量身打造的二進制機器碼。

---

## 📊 三、 實測戰果對比（突破 2,000 FPS 大關！）

| 模式 / 階段 | 第一次載入 (無快取) | 第二次載入 (使用快取) | 提升幅度 |
| :--- | :--- | :--- | :--- |
| **Model Load 載入耗時** | **21,733 ms** (AOT 編譯) | **399 ms** (直接讀取快取) | 🚀 **載入加速 54 倍！** |
| **Inference 推論耗時** | 0.6024 ms | **0.4911 ms** | 突破 0.5 ms 大關！ |
| **Throughput 吞吐量** | 1660 FPS | **2036.25 FPS** | 🚀 **突破 2,000 FPS！** |
| **CPU 前處理耗時** | - | **0.227 ms** | 保持極速 |

---

## 🎙️ 四、 面試應對指南（Interview Playbook）

### 面試必考題：
> **「TensorRT 與普通 ONNX Runtime/CUDA Provider 相比，為什麼推論速度能快這麼多？首次啟動很慢的問題你如何解決？」**

### 滿分回答結構：
1. **闡述加速核心（AI Compiler 價值）**：
   * 「TensorRT 扮演了神經網絡編譯器的角色，做了兩大核心最佳化：
     1. **Kernel Fusion**：將 Conv + Bias + ReLU 等垂直熔接，減少顯存讀寫次數。
     2. **Kernel Auto-Tuning**：在目標硬體上對卷積算法（如 Winograd 與 Tensor Core TF32/FP16 GEMM）進行實機基準測試，選出最優 Tactic。
     3. **消除 Layout 轉置**：在編譯期直接將圖與權重調整為硬體偏好的 NHWC 格式，消除執行期的轉置開銷。」
2. **解決冷啟動問題（工程落地能力）**：
   * 「針對首次建構需耗時數十秒的問題，我在系統中實作了 **Engine 序列化快取（Serialization Cache）**：
     * 將離線編譯好的專用硬體二進制檔案（`.engine`）持久化至磁碟。
     * 重啟時直接反序列化載入，**將啟動時間從 21 秒壓縮至 399 毫秒，同時享有 2000+ FPS 的極致吞吐量**。」
