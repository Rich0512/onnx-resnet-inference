"""
INT8 Model Quantization Script

This script quantizes a FP32 ONNX model to INT8 format using dynamic quantization.
Dynamic quantization is simpler and doesn't require calibration data.

Usage:
    python quantize.py
    python quantize.py --input model.onnx --output model_int8.onnx
"""

import os
import argparse
from pathlib import Path

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
except ImportError:
    print("Error: onnxruntime not installed.")
    print("Install with: pip install onnxruntime")
    exit(1)


def quantize_model(input_path: str, output_path: str, type: str) -> None:
    """
    Quantize ONNX model from FP32 to INT8.
    
    Args:
        input_path: Path to input FP32 ONNX model
        output_path: Path to output INT8 ONNX model
    """
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    # Get original model size
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"\nOriginal model size: {original_size:.2f} MB")
    
    # Perform quantization
    print("\nQuantizing model...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8 if type == "uint8" else QuantType.QInt8,
        per_channel=False
    )

    
    # Get quantized model size
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    print("\n✓ Quantization complete!")

    print("\n=== 量化前後比較 ===")
    print(f"| 項目 | 原始 | 量化後 |")
    print(f"| 大小 | {original_size:.2f} MB | {quantized_size:.2f} MB |")

def main():
    parser = argparse.ArgumentParser(description="Quantize ONNX model to INT8")
    parser.add_argument("--input", "-i", default="models/resnet18.onnx",
                        help="Input FP32 model path")
    parser.add_argument("--output", "-o", default="models/resnet18_int8.onnx",
                        help="Output INT8 model path")
    parser.add_argument("--type", "-t", choices=["int8", "uint8"], default="int8",
                        help="Quantization type (int8 or uint8)")
    args = parser.parse_args()
    
    # Resolve relative paths
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    
    if not input_path.exists():
        print(f"Error: Input model not found: {input_path}")
        exit(1)
    
    if not output_path.parent.exists():
        print(f"Error: Output directory not found: {output_path.parent}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        

    quantize_model(str(input_path), str(output_path), args.type)

if __name__ == "__main__":
    main()
