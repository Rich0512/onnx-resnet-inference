import os
import urllib.request

headers = [
    "NvInfer.h",
    "NvInferImpl.h",
    "NvInferLegacyDims.h",
    "NvInferPlugin.h",
    "NvInferPluginBase.h",
    "NvInferPluginUtils.h",
    "NvInferRuntime.h",
    "NvInferRuntimeBase.h",
    "NvInferRuntimeCommon.h",
    "NvInferRuntimePlugin.h",
    "NvInferVersion.h",
    "NvOnnxConfig.h"
]

target_dir = "third_party/tensorrt/include"
os.makedirs(target_dir, exist_ok=True)

base_url = "https://raw.githubusercontent.com/NVIDIA/TensorRT/release/10.7/include/"

for h in headers:
    dest = os.path.join(target_dir, h)
    url = base_url + h
    print(f"Downloading {h}...")
    urllib.request.urlretrieve(url, dest)

print(f"All {len(headers)} official NVIDIA TensorRT headers downloaded to {target_dir} successfully!")
