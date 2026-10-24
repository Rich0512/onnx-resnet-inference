"""
Station 8: Export Dynamic Batch ResNet-18 ONNX and build TensorRT Engine.
Supports batch sizes from min=1 to opt=8 to max=32.
"""
import os
import time
import torch
import torchvision.models as models
import tensorrt as trt

ONNX_DYNAMIC_PATH = "models/resnet18_dynamic.onnx"
ENGINE_DYNAMIC_PATH = "models/resnet18_dynamic_fp16.engine"

def export_dynamic_onnx():
    print("=" * 60)
    print("[1/2] Exporting Dynamic Batch ResNet-18 to ONNX...")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    
    # 1. Load pretrained PyTorch ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    
    # 2. Dummy input with batch size 1
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 3. Export with dynamic batch dimension
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_DYNAMIC_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_tensor'],
        output_names=['output_tensor'],
        dynamic_axes={
            'input_tensor': {0: 'batch_size'},
            'output_tensor': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported {ONNX_DYNAMIC_PATH} with dynamic batch dimension!")

def build_dynamic_trt_engine():
    print("\n" + "=" * 60)
    print("[2/2] Building Dynamic TensorRT FP16 Engine (min=1, opt=8, max=32)...")
    print("=" * 60)
    
    start_time = time.time()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(ONNX_DYNAMIC_PATH, "rb") as f:
        if not parser.parse(f.read()):
            print("[ERROR] Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False
            
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB Workspace
    
    # Setup Optimization Profile for Dynamic Batch Dimensions
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_tensor",
        (1, 3, 224, 224),   # Minimum batch size
        (8, 3, 224, 224),   # Optimal batch size (targeted for highest throughput)
        (32, 3, 224, 224)  # Maximum batch size
    )
    config.add_optimization_profile(profile)
    
    print("Building serialized network with TensorRT Auto-Tuner (this may take ~20-30s)...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Failed to build serialized network!")
        
    with open(ENGINE_DYNAMIC_PATH, "wb") as f:
        f.write(plan)
        
    duration = time.time() - start_time
    size_mb = os.path.getsize(ENGINE_DYNAMIC_PATH) / (1024 * 1024)
    print(f"Successfully generated {ENGINE_DYNAMIC_PATH} in {duration:.2f}s ({size_mb:.2f} MB)")
    return True

if __name__ == "__main__":
    export_dynamic_onnx()
    build_dynamic_trt_engine()
