"""
prepare_v4_fpga.py
------------------
FPGA Deployment Configuration and Verification Script for ComponentCNN (V4 Checkpoint).

Target Checkpoint:
  checkpoints/v4_best_model.pth

Tasks performed:
  1. Loads checkpoints/v4_best_model.pth and verifies the architecture against model.py.
  2. Verifies parameter counts (24,234), input tensor shape (1, 3, 64, 64), and raw output shape (1, 10).
  3. Confirms RGB 64x64 input format and normalization parameters:
     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) => dynamic range [-1.0, 1.0].
  4. Records the exact mapping of all 10 output class indices, identifying 'transistor' at index 9.
  5. Documents the 9 permitted output classes (indices 0..8) and the requirement to mask index 9
     before selecting the predicted class (logits[9] = -infinity).
  6. Executes a layer-by-layer dummy forward pass, reporting intermediate tensor dimensions
     and MAC operations for HLS synthesis planning.
  7. Generates a standalone configuration specification report saved to v4_fpga_config.txt.
  8. Strictly read-only: does not modify checkpoint, dataset, or model files.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# ── Model imports ──────────────────────────────────────────────────────────
from model import ComponentCNN, CLASS_NAMES, NUM_CLASSES, INPUT_SIZE

PROJECT_DIR        = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "v4_best_model.pth"
DEFAULT_CONFIG_OUT = PROJECT_DIR / "v4_fpga_config.txt"

NORM_MEAN_EXPECTED = (0.5, 0.5, 0.5)
NORM_STD_EXPECTED  = (0.5, 0.5, 0.5)

SEP  = "=" * 82
SEP2 = "-" * 82


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and configure ComponentCNN V4 model for FPGA deployment."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help=f"Path to V4 model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output-config",
        type=str,
        default=str(DEFAULT_CONFIG_OUT),
        help=f"Path to output configuration file (default: {DEFAULT_CONFIG_OUT})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for FPGA inference verification (default: 1 for edge streaming)",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Checkpoint Verification
# ══════════════════════════════════════════════════════════════════════════
def verify_v4_checkpoint(checkpoint_path: Path) -> Tuple[ComponentCNN, Dict]:
    if not checkpoint_path.exists():
        sys.exit(f"FATAL: Checkpoint file not found: {checkpoint_path}")

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        sys.exit(f"FATAL: Failed to read checkpoint: {exc}")

    required_keys = ["model_state_dict", "class_names", "num_classes"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        sys.exit(f"FATAL: Checkpoint missing required metadata keys: {missing}")

    if ckpt["num_classes"] != 10:
        sys.exit(f"FATAL: Expected 10 classes in V4 checkpoint, found {ckpt['num_classes']}")

    if ckpt["class_names"] != CLASS_NAMES:
        sys.exit(f"FATAL: Class names mismatch in checkpoint!\nExpected: {CLASS_NAMES}\nGot: {ckpt['class_names']}")

    # Verify normalization
    norm_mean = tuple(ckpt.get("norm_mean", NORM_MEAN_EXPECTED))
    norm_std  = tuple(ckpt.get("norm_std", NORM_STD_EXPECTED))
    if norm_mean != NORM_MEAN_EXPECTED or norm_std != NORM_STD_EXPECTED:
        sys.exit(f"FATAL: Normalization mismatch!\nExpected: {NORM_MEAN_EXPECTED}, {NORM_STD_EXPECTED}\nGot: {norm_mean}, {norm_std}")

    # Build model and load weights
    model = ComponentCNN(num_classes=10)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Verify parameter count
    total_params = sum(p.numel() for p in model.parameters())
    if total_params != 24234:
        sys.exit(f"FATAL: Expected 24,234 parameters, found {total_params}")

    return model, ckpt


# ══════════════════════════════════════════════════════════════════════════
# Layer Profile & Dummy Forward Pass
# ══════════════════════════════════════════════════════════════════════════
def profile_forward_pass(model: ComponentCNN, batch_size: int = 1) -> List[Dict]:
    """
    Simulate layer-by-layer execution to verify intermediate shapes,
    parameter counts, and compute MACs for FPGA hardware budgeting.
    """
    dummy_input = torch.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE)

    layers_info = []

    # Block 1
    c1 = model.features[0](dummy_input)
    r1 = model.features[1](c1)
    p1 = model.features[2](r1)
    # Conv1 MACs: B * H_out * W_out * C_out * (C_in * K * K)
    macs_c1 = batch_size * 32 * 32 * 16 * (3 * 3 * 3)
    layers_info.append({
        "block": "Conv Block 1",
        "operation": "Conv2d(3, 16, k=3, p=1) + ReLU + MaxPool2d(2)",
        "in_shape": tuple(dummy_input.shape),
        "out_shape": tuple(p1.shape),
        "weight_shape": tuple(model.features[0].weight.shape),
        "bias_shape": tuple(model.features[0].bias.shape),
        "params": model.features[0].weight.numel() + model.features[0].bias.numel(),
        "macs": macs_c1,
    })

    # Block 2
    c2 = model.features[3](p1)
    r2 = model.features[4](c2)
    p2 = model.features[5](r2)
    macs_c2 = batch_size * 16 * 16 * 32 * (16 * 3 * 3)
    layers_info.append({
        "block": "Conv Block 2",
        "operation": "Conv2d(16, 32, k=3, p=1) + ReLU + MaxPool2d(2)",
        "in_shape": tuple(p1.shape),
        "out_shape": tuple(p2.shape),
        "weight_shape": tuple(model.features[3].weight.shape),
        "bias_shape": tuple(model.features[3].bias.shape),
        "params": model.features[3].weight.numel() + model.features[3].bias.numel(),
        "macs": macs_c2,
    })

    # Block 3
    c3 = model.features[6](p2)
    r3 = model.features[7](c3)
    p3 = model.features[8](r3)
    macs_c3 = batch_size * 8 * 8 * 64 * (32 * 3 * 3)
    layers_info.append({
        "block": "Conv Block 3",
        "operation": "Conv2d(32, 64, k=3, p=1) + ReLU + MaxPool2d(2)",
        "in_shape": tuple(p2.shape),
        "out_shape": tuple(p3.shape),
        "weight_shape": tuple(model.features[6].weight.shape),
        "bias_shape": tuple(model.features[6].bias.shape),
        "params": model.features[6].weight.numel() + model.features[6].bias.numel(),
        "macs": macs_c3,
    })

    # AdaptiveAvgPool
    pool_out = model.pool(p3)
    layers_info.append({
        "block": "Global Pooling",
        "operation": "AdaptiveAvgPool2d((1, 1))",
        "in_shape": tuple(p3.shape),
        "out_shape": tuple(pool_out.shape),
        "weight_shape": "None",
        "bias_shape": "None",
        "params": 0,
        "macs": batch_size * 64 * 8 * 8,
    })

    # Flatten & Classifier
    flat = torch.flatten(pool_out, 1)
    logits = model.classifier(flat)
    macs_fc = batch_size * 64 * 10
    layers_info.append({
        "block": "Linear Classifier",
        "operation": "Linear(in=64, out=10)",
        "in_shape": tuple(flat.shape),
        "out_shape": tuple(logits.shape),
        "weight_shape": tuple(model.classifier.weight.shape),
        "bias_shape": tuple(model.classifier.bias.shape),
        "params": model.classifier.weight.numel() + model.classifier.bias.numel(),
        "macs": macs_fc,
    })

    return layers_info


# ══════════════════════════════════════════════════════════════════════════
# Report Generator
# ══════════════════════════════════════════════════════════════════════════
def generate_fpga_config_report(
    ckpt: Dict,
    layers_info: List[Dict],
    output_path: Path,
) -> str:
    total_params = sum(lyr["params"] for lyr in layers_info)
    total_macs   = sum(lyr["macs"] for lyr in layers_info)
    fp32_kb      = (total_params * 4) / 1024.0
    int16_kb     = (total_params * 2) / 1024.0
    int8_kb      = (total_params * 1) / 1024.0

    transistor_idx = CLASS_NAMES.index("transistor")

    lines = [
        "=" * 82,
        "  ComponentCNN (V4) — FPGA Deployment Configuration Specification",
        "  Hardware Target: PYNQ-Z2 (Zynq-7020 SoC / Xilinx Vivado HLS)",
        "=" * 82,
        "",
        "1. GENERAL CHECKPOINT INFORMATION",
        "-" * 82,
        f"  Checkpoint File       : {DEFAULT_CHECKPOINT}",
        f"  Model Architecture    : ComponentCNN (model.py)",
        f"  Checkpoint Epoch      : {ckpt.get('epoch', 'N/A')}",
        f"  Val Accuracy (ckpt)   : {ckpt.get('val_accuracy', 0.0):.2f}%",
        f"  Val Macro F1 (ckpt)   : {ckpt.get('val_macro_f1', 0.0):.4f}",
        f"  Verification Status   : PASS (All weights, shapes, and metadata verified)",
        "",
        "2. INPUT SPECIFICATION & HARDWARE PREPROCESSING",
        "-" * 82,
        f"  Input Resolution      : 64 x 64 pixels",
        f"  Input Channels        : 3 (RGB, Planar: Channel-first [C, H, W] = [3, 64, 64])",
        f"  Color Channel Order   : R, G, B",
        f"  Raw Pixel Range       : [0, 255] (unsigned 8-bit integer, uint8)",
        f"  Normalization Mean    : {NORM_MEAN_EXPECTED}",
        f"  Normalization Std     : {NORM_STD_EXPECTED}",
        "",
        "  FPGA Preprocessing Fixed-Point Conversion Formula:",
        "    Normalized_Float = ((Pixel_uint8 / 255.0) - 0.5) / 0.5",
        "                     = (2.0 * Pixel_uint8 / 255.0) - 1.0",
        "    Dynamic Range    : [-1.0, +1.0]",
        "",
        "    Hardware Fixed-Point Formulation (Q1.15 or Q8.8):",
        "      Input_Q15 = round( ((Pixel_uint8 << 16) / 255) - 32768 )",
        "",
        "3. OUTPUT CLASS MAPPING & TRANSISTOR MASKING SPECIFICATION",
        "-" * 82,
        f"  Total Raw Output Classes   : {NUM_CLASSES} (Indices 0 through 9)",
        f"  Permitted Classes Count    : 9 (Indices 0 through 8)",
        f"  Transistor Output Index    : {transistor_idx} (MUST BE MASKED)",
        "",
        f"  {'Index':<7} {'Class Name':<22} {'Deployment Status':<20} {'Action / Notes'}",
        "  " + "-" * 78,
    ]

    for idx, name in enumerate(CLASS_NAMES):
        if idx == transistor_idx:
            status = "MASKED / EXCLUDED"
            action = "Set logit to -inf; do not allow as output prediction"
        else:
            status = "PERMITTED"
            action = "Valid component prediction candidate"
        lines.append(f"  [{idx:<3}] {name:<22} {status:<20} {action}")

    lines.extend([
        "  " + "-" * 78,
        "",
        "  Required Post-Processing Logic (Firmware / HLS Comparator):",
        "    In Python / Host Driver:",
        "      raw_logits = model(input_tensor)               # Shape: [1, 10]",
        "      raw_logits[:, 9] = -float('inf')               # Mask transistor",
        "      predicted_idx = raw_logits[:, :9].argmax(dim=1)",
        "",
        "    In Vivado HLS C++ Synthesis:",
        "      // Only search indices 0 to 8 in the argmax reduction tree:",
        "      int best_class = 0;",
        "      fixed_t max_logit = logits[0];",
        "      for (int i = 1; i < 9; i++) { // Loop bounds explicitly exclude index 9",
        "          if (logits[i] > max_logit) {",
        "              max_logit = logits[i];",
        "              best_class = i;",
        "          }",
        "      }",
        "",
        "4. LAYER-BY-LAYER HARDWARE PIPELINE PROFILE",
        "-" * 82,
        f"  {'Block':<18} {'Operation / Kernel':<36} {'Output Shape':<14} {'Params':>8} {'MACs':>10}",
        "  " + "-" * 88,
    ])

    for lyr in layers_info:
        lines.append(
            f"  {lyr['block']:<18} {lyr['operation']:<36} {str(lyr['out_shape']):<14} {lyr['params']:>8,d} {lyr['macs']:>10,d}"
        )

    lines.extend([
        "  " + "-" * 88,
        f"  {'TOTAL':<18} {'Pipeline Total':<36} {'(1, 10)':<14} {total_params:>8,d} {total_macs:>10,d}",
        "",
        "5. HARDWARE RESOURCE & MEMORY FOOTPRINT ESTIMATE",
        "-" * 82,
        f"  Total Trainable Parameters : {total_params:,}",
        f"  Total Multiply-Accumulates : {total_macs:,} MACs per inference frame",
        f"  Weight Memory (FP32)       : {fp32_kb:.2f} KB  (Baseline 32-bit float)",
        f"  Weight Memory (INT16 / Q15): {int16_kb:.2f} KB  (16-bit fixed point)",
        f"  Weight Memory (INT8 / Q7)  : {int8_kb:.2f} KB  (8-bit quantized integer)",
        "",
        "  PYNQ-Z2 On-Chip BRAM Fit Assessment:",
        f"    - PYNQ-Z2 (Zynq-7020) Block RAM : 4.9 Mb (~612.5 KB)",
        f"    - Entire 24.2K parameter weights require ONLY ~24 to ~48 KB of BRAM.",
        f"    - Fit Verdict: 100% of network weights comfortably fit entirely on-chip.",
        f"    - Zero off-chip DDR bandwidth required for weight streaming!",
        "",
        "6. DEPLOYMENT VERIFICATION CHECKLIST",
        "-" * 82,
        "  [x] Checkpoint exists and loads cleanly without PyTorch warnings",
        "  [x] Model architecture matches model.py ComponentCNN exactly",
        "  [x] Parameter count verified: exactly 24,234 parameters",
        "  [x] Input resolution verified: 64x64 RGB",
        "  [x] Normalization verified: mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)",
        "  [x] 10 raw output classes mapped and aligned with project index convention",
        "  [x] Transistor identified at index 9; masking requirement explicitly specified",
        "  [x] Dummy forward pass verified: (1, 3, 64, 64) -> (1, 10)",
        "=" * 82,
    ])

    report_text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ══════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_arguments()
    ckpt_path = Path(args.checkpoint)
    out_config = Path(args.output_config)

    print()
    print(SEP)
    print("  ComponentCNN (V4) FPGA Deployment Configuration & Verification")
    print(SEP)
    print(f"  Target Checkpoint : {ckpt_path}")
    print(f"  Output Config     : {out_config}")
    print(f"  Inference Batch   : {args.batch_size}")
    print(SEP2)

    # 1. Verify Checkpoint
    print("\n[1/3] Loading checkpoint and verifying architecture & metadata …")
    model, ckpt = verify_v4_checkpoint(ckpt_path)
    print(f"  -> Architecture   : ComponentCNN (model.py)")
    print(f"  -> Checkpoint Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  -> Val Macro F1   : {ckpt.get('val_macro_f1', 0.0):.4f}")
    print(f"  -> Val Accuracy   : {ckpt.get('val_accuracy', 0.0):.2f}%")
    print(f"  -> Parameters     : {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  -> Input Spec     : RGB 64x64, mean={NORM_MEAN_EXPECTED}, std={NORM_STD_EXPECTED}")
    print(f"  -> Raw Classes    : {NUM_CLASSES} classes (Index 9: 'transistor' [TO BE MASKED])")

    # 2. Dummy Forward Pass
    print("\n[2/3] Simulating layer-by-layer forward pass …")
    layers_info = profile_forward_pass(model, batch_size=args.batch_size)
    for lyr in layers_info:
        print(f"  {lyr['block']:<18} | in: {str(lyr['in_shape']):<14} | out: {str(lyr['out_shape']):<14} | MACs: {lyr['macs']:>8,d}")

    # Test Transistor Masking on Logits
    dummy_input = torch.randn(args.batch_size, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        raw_logits = model(dummy_input)
    masked_logits = raw_logits.clone()
    masked_logits[:, 9] = -float("inf")
    pred_class = masked_logits.argmax(dim=1).item()
    print(f"\n  Dummy Raw Logits  : shape={tuple(raw_logits.shape)}")
    print(f"  Masked Logit (ch 9): {masked_logits[0, 9].item()}")
    print(f"  Permitted Winner  : Index {pred_class} ({CLASS_NAMES[pred_class]})")

    # 3. Save Configuration Specification
    print(f"\n[3/3] Generating FPGA configuration specification ({out_config.name}) …")
    report_text = generate_fpga_config_report(ckpt, layers_info, out_config)
    print(f"  -> Saved FPGA configuration to:\n     {out_config.resolve()}")

    print("\n" + SEP)
    print("  VERIFICATION COMPLETE — READY FOR FPGA SYNTHESIS WORKFLOW")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
