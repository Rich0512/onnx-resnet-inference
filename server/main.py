import os
import sys
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager

# Add Pybind module and its DLL dependencies paths
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))
sys.path.append(build_dir)

if os.name == 'nt':
    os.add_dll_directory(build_dir)
    opencv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'opencv', 'build', 'x64', 'vc16', 'bin'))
    if os.path.exists(opencv_bin): os.add_dll_directory(opencv_bin)
    onnx_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'onnxruntime', 'lib'))
    if os.path.exists(onnx_bin): os.add_dll_directory(onnx_bin)
    
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if os.path.exists(cuda_bin): os.add_dll_directory(cuda_bin)
    
    # CuDNN may also be required by the ONNX Runtime CUDA provider
    cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.18\bin\12.9\x64"
    if os.path.exists(cudnn_bin): os.add_dll_directory(cudnn_bin)

try:
    import resnet_engine
except ImportError as e:
    raise RuntimeError(f"Failed to load C++ Engine module: {e}")

# Global Engine Instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Events
    global engine
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet18.onnx'))
    
    print(f"Loading ONNX Model from: {model_path} into C++ Engine...")
    engine = resnet_engine.Engine(
        model_path=model_path, 
        num_threads=4, 
        use_gpu=False,  # Fallback to CPU for now until CUDA paths are fully resolved
        use_tensorrt=False
    )
    print("Engine initialization complete!")
    yield
    # Shutdown Events
    print("Shutting down the server, releasing C++ engine...")

app = FastAPI(title="ONNX ResNet Fast Inference API", lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Welcome to C++ ONNX Inference API via FastAPI!"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG/PNG)")

    # Read image into bytes
    image_bytes = await file.read()
    
    start_time = time.perf_counter()
    try:
        # Send raw bytes directly down to C++ layer for decode and infer
        # This operates on the thread_local DataVector buffer in C++ !
        scores = engine.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Engine Error: {str(e)}")
    end_time = time.perf_counter()

    # Find the top predicted class index and score
    max_idx = scores.index(max(scores))
    max_score = scores[max_idx]

    # In a real app we'd map this max_idx to imagenet_classes.txt
    
    return {
        "success": True,
        "inference_time_ms": round((end_time - start_time) * 1000, 3),
        "prediction": {
            "class_id": max_idx,
            "confidence": round(max_score, 4)
        }
    }
