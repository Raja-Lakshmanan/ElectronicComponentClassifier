"""
export_v5_int8_weights.py
-------------------------
Phase 2 -- Export actual quantized signed INT8 weights for V5-Spatial
(ComponentCNNSpatial) into fpga_export/v5_spatial/integer_params/.

Exports:
  - conv1_weight_int8.npy and conv1_weight_int8.txt (features.0.weight, [16, 3, 3, 3])
  - conv2_weight_int8.npy and conv2_weight_int8.txt (features.3.weight, [32, 16, 3, 3])
  - conv3_weight_int8.npy and conv3_weight_int8.txt (features.6.weight, [64, 32, 3, 3])
  - fc1_weight_int8.npy   and fc1_weight_int8.txt   (classifier.0.weight, [64, 256])
  - fc2_weight_int8.npy   and fc2_weight_int8.txt   (classifier.2.weight, [10, 64])

Formula:
  scale = max(|w_fp32|) / 127.0
  w_int8 = clip(round(w_fp32 / scale), -128, 127).astype(np.int8)

Computes SHA-256 checksums for every exported file.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = PROJECT_DIR / "fpga_export" / "v5_spatial"
INT_PARAMS_DIR = EXPORT_DIR / "integer_params"
MANIFEST_PATH = INT_PARAMS_DIR / "manifest_int8_weights.json"

WEIGHT_CONFIGS = [
    {
        "layer_id": "conv1",
        "display_name": "Conv Block 1 Weight",
        "raw_npy": "features_0_weight.npy",
        "out_npy": "conv1_weight_int8.npy",
        "out_txt": "conv1_weight_int8.txt",
        "param_key": "features.0.weight",
        "shape": [16, 3, 3, 3],
        "elements": 432,
    },
    {
        "layer_id": "conv2",
        "display_name": "Conv Block 2 Weight",
        "raw_npy": "features_3_weight.npy",
        "out_npy": "conv2_weight_int8.npy",
        "out_txt": "conv2_weight_int8.txt",
        "param_key": "features.3.weight",
        "shape": [32, 16, 3, 3],
        "elements": 4608,
    },
    {
        "layer_id": "conv3",
        "display_name": "Conv Block 3 Weight",
        "raw_npy": "features_6_weight.npy",
        "out_npy": "conv3_weight_int8.npy",
        "out_txt": "conv3_weight_int8.txt",
        "param_key": "features.6.weight",
        "shape": [64, 32, 3, 3],
        "elements": 18432,
    },
    {
        "layer_id": "fc1",
        "display_name": "FC Hidden (FC1) Weight",
        "raw_npy": "classifier_0_weight.npy",
        "out_npy": "fc1_weight_int8.npy",
        "out_txt": "fc1_weight_int8.txt",
        "param_key": "classifier.0.weight",
        "shape": [64, 256],
        "elements": 16384,
    },
    {
        "layer_id": "fc2",
        "display_name": "FC Output (FC2) Weight",
        "raw_npy": "classifier_2_weight.npy",
        "out_npy": "fc2_weight_int8.npy",
        "out_txt": "fc2_weight_int8.txt",
        "param_key": "classifier.2.weight",
        "shape": [10, 64],
        "elements": 640,
    },
]


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    INT_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "description": "Exported INT8 weights for V5-Spatial (ComponentCNNSpatial)",
        "quantization_scheme": "Symmetric per-tensor to signed INT8 [-128, 127]",
        "rounding_rule": "Round-to-nearest-even (np.round / IEEE 754)",
        "formula": "w_int8 = clip(round(w_fp32 / scale), -128, 127)",
        "weights": {},
    }

    print("=" * 82)
    print("  Exporting Quantized Signed INT8 Weights for V5-Spatial")
    print("=" * 82)
    print(f"{'Layer':<8} {'Shape':<16} {'Scale':<14} {'Min':<6} {'Max':<6} {'NPY File':<24} {'SHA256 (first 8)'}")
    print("-" * 82)

    for cfg in WEIGHT_CONFIGS:
        raw_path = EXPORT_DIR / cfg["raw_npy"]
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw weight file not found: {raw_path}")

        w_fp32 = np.load(raw_path)
        assert w_fp32.shape == tuple(cfg["shape"]), f"Shape mismatch in {raw_path.name}"

        max_abs = float(np.abs(w_fp32).max())
        scale = max_abs / 127.0 if max_abs > 0 else 1.0

        w_int8 = np.clip(np.round(w_fp32 / scale), -128, 127).astype(np.int8)

        # Save .npy
        npy_path = INT_PARAMS_DIR / cfg["out_npy"]
        np.save(npy_path, w_int8)

        # Save readable .txt
        txt_path = INT_PARAMS_DIR / cfg["out_txt"]
        w_flat = w_int8.flatten()
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(f"# Quantized INT8 weights for {cfg['display_name']} ({cfg['param_key']})\n")
            f_txt.write(f"# Shape: {cfg['shape']}, dtype: int8, elements: {cfg['elements']}\n")
            f_txt.write(f"# Scale: {scale:.12e}, max_abs_fp32: {max_abs:.10f}\n")
            f_txt.write(f"# Min: {int(w_int8.min())}, Max: {int(w_int8.max())}\n")
            f_txt.write("# index: int8_value\n")
            for idx, val in enumerate(w_flat):
                f_txt.write(f"{idx:6d}: {int(val):4d}\n")

        # Checksums
        npy_sha = sha256_file(npy_path)
        txt_sha = sha256_file(txt_path)

        manifest["weights"][cfg["layer_id"]] = {
            "display_name": cfg["display_name"],
            "param_key": cfg["param_key"],
            "shape": cfg["shape"],
            "elements": cfg["elements"],
            "dtype": "int8",
            "scale": scale,
            "max_abs_fp32": max_abs,
            "int8_range": [int(w_int8.min()), int(w_int8.max())],
            "npy_file": cfg["out_npy"],
            "npy_sha256": npy_sha,
            "txt_file": cfg["out_txt"],
            "txt_sha256": txt_sha,
        }

        print(f"{cfg['layer_id']:<8} {str(cfg['shape']):<16} {scale:<14.8f} {int(w_int8.min()):<6d} {int(w_int8.max()):<6d} {cfg['out_npy']:<24} {npy_sha[:8]}")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("-" * 82)
    print(f"Saved manifest: {MANIFEST_PATH.resolve()}")
    print("All 5 INT8 weight tensors exported and hashed successfully.")


if __name__ == "__main__":
    main()
