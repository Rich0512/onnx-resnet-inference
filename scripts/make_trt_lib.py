import os
import subprocess
import re

dll_path = "build/Release/nvinfer_10.dll"
def_path = "third_party/nvinfer_10.def"
lib_path = "third_party/nvinfer_10.lib"

print(f"Reading exports from {dll_path}...")
out = subprocess.check_output(["dumpbin", "/exports", dll_path]).decode("utf-8", errors="ignore")
lines = out.splitlines()
exports = []
start = False
for line in lines:
    if "ordinal" in line and "hint" in line:
        start = True
        continue
    if start:
        m = re.match(r"\s*\d+\s+[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+(\S+)", line)
        if m:
            exports.append(m.group(1))
        elif "Summary" in line:
            break

print(f"Found {len(exports)} exported symbols.")
os.makedirs("third_party", exist_ok=True)
with open(def_path, "w") as f:
    f.write("LIBRARY nvinfer_10.dll\nEXPORTS\n")
    for exp in exports:
        f.write(f"    {exp}\n")

print(f"Generating {lib_path} via MSVC lib tool...")
subprocess.check_call(["lib", f"/def:{def_path}", "/machine:x64", f"/out:{lib_path}"])
print("Successfully generated nvinfer_10.lib!")
