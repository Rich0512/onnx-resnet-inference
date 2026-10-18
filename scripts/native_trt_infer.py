"""
Native TensorRT Inference Benchmark (Option B)
Comparing pure TensorRT C++/Python API pipeline against Option A (ONNX Runtime TRT EP).
"""
import os
import time
import numpy as np
import cv2
import tensorrt as trt
import torch

ENGINE_PATH = "models/resnet18_native_fp16.engine"
ONNX_PATH = "models/resnet18.onnx"
IMAGE_PATH = "build/Release/test_cat.jpg"

def build_engine(onnx_file_path, engine_file_path):
    print(f"[Build] Building Native TensorRT Engine from {onnx_file_path}...")
    start_time = time.time()
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_file_path, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None
            
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB Workspace
    
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Failed to build serialized network!")
        
    with open(engine_file_path, "wb") as f:
        f.write(plan)
        
    build_time = time.time() - start_time
    print(f"[Build] Engine built in {build_time:.2f} s, saved to {engine_file_path} (size: {plan.nbytes / (1024*1024):.2f} MB)")

def run_native_inference():
    print("=" * 50)
    print("   Native TensorRT Inference Engine (Option B)")
    print("=" * 50)
    
    if not os.path.exists(ENGINE_PATH):
        build_engine(ONNX_PATH, ENGINE_PATH)
    else:
        print(f"[1] Found existing serialized engine: {ENGINE_PATH}")

    # [1] Load & Deserialize Engine
    t0 = time.perf_counter()
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    load_time_ms = (time.perf_counter() - t0) * 1000
    print(f"[2] Deserialized Engine in {load_time_ms:.2f} ms")

    context = engine.create_execution_context()

    # Inspect bindings / IO tensors
    input_name = "input_tensor"
    output_name = "output_tensor"
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    print(f"    Input:  '{input_name}', Shape: {input_shape}, Dtype: {engine.get_tensor_dtype(input_name)}")
    print(f"    Output: '{output_name}', Shape: {output_shape}, Dtype: {engine.get_tensor_dtype(output_name)}")

    # [2] Allocate GPU Memory (Device Tensors)
    # ResNet-18 input: (1, 3, 224, 224) float32, output: (1, 1000) float32
    d_input = torch.empty(tuple(input_shape), dtype=torch.float32, device="cuda")
    d_output = torch.empty(tuple(output_shape), dtype=torch.float32, device="cuda")
    stream = torch.cuda.Stream()

    # Bind tensor device pointers (Modern TensorRT 10 API)
    context.set_tensor_address(input_name, d_input.data_ptr())
    context.set_tensor_address(output_name, d_output.data_ptr())

    # [3] Preprocessing (Image -> NCHW Float)
    t_pre0 = time.perf_counter()
    img = cv2.imread(IMAGE_PATH)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # -> NCHW
    t_pre_ms = (time.perf_counter() - t_pre0) * 1000
    print(f"[3] Preprocessing completed in {t_pre_ms:.3f} ms")

    # Copy to GPU device memory
    d_input.copy_(torch.from_numpy(img).cuda())

    # [4] Warmup
    for _ in range(10):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    # [5] Benchmark 100 Runs
    latencies = []
    print("[4] Running 100 inference iterations on Native TensorRT...")
    for _ in range(100):
        start = time.perf_counter()
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    avg_ms = np.mean(latencies)
    min_ms = np.min(latencies)
    max_ms = np.max(latencies)
    fps = 1000.0 / avg_ms

    # Get results back
    output = d_output.cpu().numpy()[0]
    top5_idx = np.argsort(output)[::-1][:5]

    print("\n" + "=" * 50)
    print("   Performance Summary (Option B: Native TensorRT)")
    print("=" * 50)
    print(f"  Engine Load Time: {load_time_ms:.2f} ms")
    print(f"  Min Latency:      {min_ms:.3f} ms")
    print(f"  Max Latency:      {max_ms:.3f} ms")
    print(f"  Avg Latency:      {avg_ms:.3f} ms")
    print(f"  Throughput:       {fps:.2f} FPS")
    print("=" * 50)
    print("[5] Top-5 Predictions:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"    {rank}. Class #{idx} (score: {output[idx]:.4f})")
    print("=" * 50)

if __name__ == "__main__":
    run_native_inference()
