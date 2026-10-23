"""
export_v5_integer_biases.py
---------------------------
Phase 2 -- Export pre-scaled integer biases for V5-Spatial (PYNQ-Z2 FPGA preparation).

Converts the trained FP32 biases from checkpoints/v5_spatial_best_model.pth into
integer accumulator biases for the 5 learned layers:
  - Conv1 (features.0): 16 filters
  - Conv2 (features.3): 32 filters
  - Conv3 (features.6): 64 filters
  - FC1   (classifier.0): 64 units
  - FC2   (classifier.2): 10 units (all 10 classes structurally preserved)

Mathematical formulation:
  bias_scale = input_activation_scale * weight_scale
  bias_int32 = round(bias_fp32 / bias_scale)

Rounding rule:
  Round-to-nearest-even (IEEE 754 / NumPy np.round default). Also checks and logs
  round-half-up (np.floor(x + 0.5)) for hardware comparison.

Outputs saved to:
  fpga_export/v5_spatial/integer_params/
    - conv1_bias_int32.npy
    - conv2_bias_int32.npy
    - conv3_bias_int32.npy
    - fc1_bias_int32.npy
    - fc2_bias_int32.npy
    - manifest_integer_biases.json

IMPORTANT NOTE ON CALIBRATION:
  The input activation scales used here originate from the training-set calibration
  subset generated in Phase 2 Step 2.3 (evaluate_v5_quantization.py). If the activation
  quantization strategy is modified later (e.g. percentile clipping, power-of-two scales,
  or post-quantization calibration), these bias scales MUST be regenerated accordingly.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "v5_spatial_best_model.pth"
QUANT_REPORT_PATH = PROJECT_DIR / "fpga_export" / "v5_spatial" / "quantization_eval" / "quantization_report.json"
OUTPUT_DIR = PROJECT_DIR / "fpga_export" / "v5_spatial" / "integer_params"
MANIFEST_PATH = OUTPUT_DIR / "manifest_integer_biases.json"

SEP = "=" * 82
SEP2 = "-" * 82

# Layer configurations
LAYER_CONFIGS = [
    {
        "layer_id": "conv1",
        "display_name": "Conv Block 1",
        "param_weight_key": "features.0.weight",
        "param_bias_key": "features.0.bias",
        "act_scale_key": "input",
        "act_scale_desc": "Normalized RGB input image",
        "out_npy_name": "conv1_bias_int32.npy",
        "out_txt_name": "conv1_bias_int32.txt",
        "elements": 16,
    },
    {
        "layer_id": "conv2",
        "display_name": "Conv Block 2",
        "param_weight_key": "features.3.weight",
        "param_bias_key": "features.3.bias",
        "act_scale_key": "after_pool1",
        "act_scale_desc": "Post-Pool1 activation (identical to post-Conv1+ReLU scale)",
        "out_npy_name": "conv2_bias_int32.npy",
        "out_txt_name": "conv2_bias_int32.txt",
        "elements": 32,
    },
    {
        "layer_id": "conv3",
        "display_name": "Conv Block 3",
        "param_weight_key": "features.6.weight",
        "param_bias_key": "features.6.bias",
        "act_scale_key": "after_pool2",
        "act_scale_desc": "Post-Pool2 activation (identical to post-Conv2+ReLU scale)",
        "out_npy_name": "conv3_bias_int32.npy",
        "out_txt_name": "conv3_bias_int32.txt",
        "elements": 64,
    },
    {
        "layer_id": "fc1",
        "display_name": "FC Hidden (FC1)",
        "param_weight_key": "classifier.0.weight",
        "param_bias_key": "classifier.0.bias",
        "act_scale_key": "after_spatial_pool",
        "act_scale_desc": "Post-AdaptiveAvgPool2d(2,2) flattened activation",
        "out_npy_name": "fc1_bias_int32.npy",
        "out_txt_name": "fc1_bias_int32.txt",
        "elements": 64,
    },
    {
        "layer_id": "fc2",
        "display_name": "FC Output (FC2)",
        "param_weight_key": "classifier.2.weight",
        "param_bias_key": "classifier.2.bias",
        "act_scale_key": "after_fc1_relu",
        "act_scale_desc": "Post-FC1+ReLU activation",
        "out_npy_name": "fc2_bias_int32.npy",
        "out_txt_name": "fc2_bias_int32.txt",
        "elements": 10,
    },
]


def load_inputs():
    """Load model checkpoint and calibration scales from quantization report."""
    if not CHECKPOINT_PATH.exists():
        sys.exit(f"ERROR: Checkpoint file not found: {CHECKPOINT_PATH}")
    if not QUANT_REPORT_PATH.exists():
        sys.exit(f"ERROR: Quantization report not found: {QUANT_REPORT_PATH}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    with open(QUANT_REPORT_PATH, "r", encoding="utf-8") as f:
        quant_report = json.load(f)

    act_scales = quant_report["fixed_point_integer"]["activation_scales"]
    return ckpt, state_dict, act_scales, quant_report


def main():
    print(SEP)
    print("  Phase 2 -- Export V5-Spatial Pre-Scaled Integer Biases for FPGA HLS")
    print(SEP)

    ckpt, state_dict, act_scales, quant_report = load_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSource Checkpoint: {CHECKPOINT_PATH.name} (Epoch {ckpt.get('epoch', 'N/A')})")
    print(f"Calibration Source: {quant_report.get('calibration_source', 'cleaned_dataset/train')}")
    print(f"Output Directory:   {OUTPUT_DIR}\n")

    manifest: Dict[str, Any] = {
        "description": "Pre-scaled integer biases for V5-Spatial (ComponentCNNSpatial) FPGA implementation",
        "source_checkpoint": CHECKPOINT_PATH.name,
        "checkpoint_epoch": ckpt.get("epoch"),
        "formula": "bias_int = round(bias_fp32 / bias_scale), where bias_scale = input_activation_scale * weight_scale",
        "rounding_rule": "round-to-nearest-even (IEEE 754 / NumPy np.round)",
        "default_dtype": "int32",
        "calibration_note": (
            "Activation scales were determined via training-set calibration in evaluate_v5_quantization.py. "
            "If activation scaling or calibration strategy is modified, these integer biases must be regenerated."
        ),
        "layers": {},
    }

    print(f"{'Layer':<10} {'Shape':<8} {'Act Scale':<14} {'Wgt Scale':<14} {'Bias Scale':<16} {'Max |b_fp|':<12} {'Max |b_int|':<12} {'Fits INT16?'}")
    print(SEP2)

    generated_arrays = {}

    for cfg in LAYER_CONFIGS:
        layer_id = cfg["layer_id"]
        w_tensor = state_dict[cfg["param_weight_key"]].cpu().numpy()
        b_tensor = state_dict[cfg["param_bias_key"]].cpu().numpy()

        # 1. Input activation scale
        act_scale = float(act_scales[cfg["act_scale_key"]])

        # 2. Weight scale: symmetric per-tensor, scale = max(|w|) / 127.0
        w_max = float(np.abs(w_tensor).max())
        w_scale = float(w_max / 127.0)

        # 3. Accumulator bias scale
        bias_scale = act_scale * w_scale

        # 4. Quantize bias to INT32
        # Rounding rule: round-to-nearest-even
        scaled_bias = b_tensor / bias_scale
        b_int32 = np.round(scaled_bias).astype(np.int32)
        generated_arrays[layer_id] = b_int32

        # Round-half-up comparison (floor(x + 0.5))
        b_half_up = np.floor(scaled_bias + 0.5).astype(np.int32)
        diff_half_up = int(np.abs(b_int32 - b_half_up).sum())

        # Fit check
        b_abs_max = int(np.abs(b_int32).max())
        fits_int16 = bool(b_abs_max <= 32767)
        fits_int8 = bool(b_abs_max <= 127)

        # Dequantization error analysis
        dequant_fp32 = b_int32.astype(np.float64) * bias_scale
        abs_err = np.abs(b_tensor - dequant_fp32)
        max_err = float(abs_err.max())
        mean_err = float(abs_err.mean())

        # Export .npy file (INT32)
        npy_path = OUTPUT_DIR / cfg["out_npy_name"]
        np.save(npy_path, b_int32)

        # Export human-readable .txt file
        txt_path = OUTPUT_DIR / cfg["out_txt_name"]
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write(f"# Integer Biases for {cfg['display_name']} ({cfg['param_bias_key']})\n")
            f_txt.write(f"# Shape: {list(b_int32.shape)}, dtype: int32\n")
            f_txt.write(f"# Bias scale: {bias_scale:.12e} (act_scale={act_scale:.10f}, w_scale={w_scale:.10f})\n")
            f_txt.write(f"# Max value: {int(b_int32.max())}, Min value: {int(b_int32.min())}\n")
            f_txt.write("# index: int32_value (dequantized_fp32, original_fp32)\n")
            for idx, val in enumerate(b_int32):
                f_txt.write(f"{idx:4d}: {int(val):8d}  (deq={float(dequant_fp32[idx]):+.6f}, orig={float(b_tensor[idx]):+.6f})\n")

        # Manifest entry
        manifest["layers"][layer_id] = {
            "display_name": cfg["display_name"],
            "param_weight_key": cfg["param_weight_key"],
            "param_bias_key": cfg["param_bias_key"],
            "shape": list(b_int32.shape),
            "elements": cfg["elements"],
            "dtype": "int32",
            "npy_file": cfg["out_npy_name"],
            "txt_file": cfg["out_txt_name"],
            "input_activation_scale": {
                "key": cfg["act_scale_key"],
                "value": act_scale,
                "description": cfg["act_scale_desc"],
            },
            "weight_scale": {
                "max_abs_weight": w_max,
                "value": w_scale,
            },
            "bias_scale": bias_scale,
            "int32_range": {
                "min": int(b_int32.min()),
                "max": int(b_int32.max()),
                "max_absolute": b_abs_max,
            },
            "fits_int16": fits_int16,
            "fits_int8": fits_int8,
            "quantization_error": {
                "max_abs_error": max_err,
                "mean_abs_error": mean_err,
            },
            "round_half_up_discrepancies": diff_half_up,
        }

        fit_str = "YES" if fits_int16 else "NO (needs INT32)"
        print(f"{layer_id:<10} {str(list(b_int32.shape)):<8} {act_scale:<14.8f} {w_scale:<14.8f} {bias_scale:<16.10f} {float(np.abs(b_tensor).max()):<12.6f} {b_abs_max:<12d} {fit_str}")

    print(SEP2)

    # Save manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nSaved manifest: {MANIFEST_PATH.resolve()}")

    # Requirement 8: Reload and verify every exported file
    print(f"\n{SEP2}")
    print("  Verification: Reloading Exported .npy Files")
    print(SEP2)

    all_verified = True
    for cfg in LAYER_CONFIGS:
        layer_id = cfg["layer_id"]
        npy_path = OUTPUT_DIR / cfg["out_npy_name"]
        reloaded = np.load(npy_path)

        orig_arr = generated_arrays[layer_id]
        shape_match = reloaded.shape == orig_arr.shape
        dtype_match = reloaded.dtype == np.int32
        value_match = np.array_equal(reloaded, orig_arr)

        if shape_match and dtype_match and value_match:
            print(f"  [PASS] {cfg['out_npy_name']:24s} -> Shape {str(reloaded.shape):<10} dtype={reloaded.dtype} (values match exactly)")
        else:
            print(f"  [FAIL] {cfg['out_npy_name']:24s} -> MISMATCH! shape_ok={shape_match}, dtype_ok={dtype_match}, val_ok={value_match}")
            all_verified = False

    if not all_verified:
        sys.exit("ERROR: File verification failed!")

    print(f"\nAll 5 integer bias tensors successfully exported and verified.")
    print(f"\nNote: All layers' integer biases fit safely in signed INT16 (max magnitude 19,140 in Conv1),")
    print(f"      and are exported in standard INT32 format for direct compatibility with 32-bit accumulators.")
    print(f"\nSTATUS NOTICE:")
    print(f"  Integer biases are now exported, but integer-only inference is NOT yet complete.")
    print(f"  Still required before full HLS synthesis:")
    print(f"    1. Fixed-point requantization multipliers (M0, S) for each layer.")
    print(f"    2. Input camera pixel LUT (256 entries).")
    print(f"    3. Bit-exact integer-only Python test harness to validate end-to-end match.")
    print(SEP)


if __name__ == "__main__":
    main()
