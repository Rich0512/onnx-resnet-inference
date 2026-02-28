import os
import sys

# Add the build/Release directory to the Python path so it can find the .pyd file
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'Release'))
sys.path.append(build_dir)

# On Windows Python 3.8+, DLL dependencies must be explicitly added
if os.name == 'nt':
    os.add_dll_directory(build_dir)
    # Also add the OpenCV bin directory if it's not in PATH
    opencv_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'opencv', 'build', 'x64', 'vc16', 'bin'))
    if os.path.exists(opencv_bin):
        os.add_dll_directory(opencv_bin)
        
    # Add ONNX Runtime lib directory (contains onnxruntime.dll and provider DLLs)
    onnx_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'third_party', 'onnxruntime', 'lib'))
    if os.path.exists(onnx_bin):
        os.add_dll_directory(onnx_bin)
        
    # Add CUDA / TensorRT bin directory
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if os.path.exists(cuda_bin):
        os.add_dll_directory(cuda_bin)


try:
    import resnet_engine
    print("Successfully imported resnet_engine!")
    print(f"Docstring: {resnet_engine.__doc__}")

    # Resolve model path relative to script
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'resnet18.onnx'))
    
    # Note: Using USE_GPU=False for a quick load test to bypass TensorRT warmup time
    engine = resnet_engine.Engine(
        model_path=model_path, 
        num_threads=4, 
        use_gpu=False, 
        use_tensorrt=False
    )
    print("Successfully initialized C++ Engine from Python!")

    # Test full prediction pipeline
    import cv2
    import numpy as np

    print("Generating a dummy image for testing...")
    # Create a 224x224 RGB image (dummy data)
    dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (50, 50), (150, 150), (0, 255, 0), -1)
    
    # Encode to JPEG bytes (simulating an uploaded file)
    success, encoded_img = cv2.imencode('.jpg', dummy_img)
    if not success:
        raise RuntimeError("Failed to encode dummy image")
        
    img_bytes = encoded_img.tobytes()
    
    print("Running inference...")
    scores = engine.predict(img_bytes)
    print(f"Prediction successful! Received {len(scores)} scores.")
    
    # Get top 1 result
    max_idx = scores.index(max(scores))
    print(f"Top prediction index: {max_idx}, Score: {scores[max_idx]:.2f}")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
