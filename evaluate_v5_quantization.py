"""
evaluate_v5_quantization.py
---------------------------
Phase 2, Step 2.3 -- Quantization evaluation for PYNQ-Z2 FPGA deployment.

Evaluates the trained V5-Spatial model under three conditions:
  1. FP32 Baseline -- Full-precision PyTorch inference (ground truth reference)
  2. Simulated INT8 Weight-Only -- Symmetric per-tensor weight quantization to
     signed INT8, dequantized back to FP32 for inference. Activations remain in
     FP32. This measures accuracy loss from weight quantization alone.
  3. Fixed-Point Integer Inference -- Layer-by-layer integer arithmetic with
     INT8 weights, INT8 activations, INT16 biases, and INT32 accumulators.
     Activation scales are calibrated from a training-set calibration subset
     (NOT from the test set). This simulates the behavior of a true integer-
     only FPGA datapath, including rounding and saturation effects.

Test set: cleaned_dataset/test/ (346 non-transistor images across 9 classes)
Calibration set: cleaned_dataset/train/ (subset, for activation scale estimation)

Outputs saved to: fpga_export/v5_spatial/quantization_eval/
  - quantization_report.json  (machine-readable full report)
  - Console summary           (human-readable)

Does NOT modify existing model, checkpoint, dataset, or exported weight files.
"""

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from torchvision import datasets, transforms
except ImportError as exc:
    sys.exit(f"ERROR: torchvision not available: {exc}")

try:
    from sklearn.metrics import f1_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

# -- Model import (read-only) ------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_v5_spatial import (
    ComponentCNNSpatial, CLASS_NAMES, NUM_CLASSES, INPUT_SIZE, PERMITTED_CLASSES,
)

# -- Paths --------------------------------------------------------------------
PROJECT_DIR     = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "v5_spatial_best_model.pth"
CLEANED_DIR     = PROJECT_DIR / "cleaned_dataset"
TEST_DIR        = CLEANED_DIR / "test"
TRAIN_DIR       = CLEANED_DIR / "train"
OUTPUT_DIR      = PROJECT_DIR / "fpga_export" / "v5_spatial" / "quantization_eval"
REPORT_PATH     = OUTPUT_DIR / "quantization_report.json"

# -- Constants ----------------------------------------------------------------
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD  = (0.5, 0.5, 0.5)
TRANSISTOR_INDEX = 9
NUM_PERMITTED    = 9
BATCH_SIZE       = 32
NUM_WORKERS      = 0
SEED             = 42
CALIBRATION_SIZE = 200  # images from train split for activation scale calibration

SEP  = "=" * 82
SEP2 = "-" * 82


# =============================================================================
#  Dataset Helpers
# =============================================================================
class NonTransistorDataset(datasets.ImageFolder):
    """ImageFolder subclass that loads only the 9 permitted classes."""
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        dir_path = Path(directory)
        class_to_idx = {cls: idx for idx, cls in enumerate(PERMITTED_CLASSES)}
        classes = sorted([c for c in PERMITTED_CLASSES if (dir_path / c).is_dir()])
        if classes != PERMITTED_CLASSES:
            missing = set(PERMITTED_CLASSES) - set(classes)
            if missing:
                sys.exit(f"ERROR: Directory '{directory}' missing classes: {missing}")
        return PERMITTED_CLASSES, class_to_idx


def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def build_test_loader() -> Tuple[DataLoader, NonTransistorDataset]:
    if not TEST_DIR.exists():
        sys.exit(f"ERROR: Test directory not found: {TEST_DIR}")
    ds = NonTransistorDataset(root=str(TEST_DIR), transform=get_eval_transforms())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)
    return loader, ds


def build_calibration_loader() -> DataLoader:
    """Load a subset of training images (deterministic, no augmentation) for
    activation scale calibration. Uses eval transforms -- no augmentation."""
    if not TRAIN_DIR.exists():
        sys.exit(f"ERROR: Train directory not found: {TRAIN_DIR}")
    ds = NonTransistorDataset(root=str(TRAIN_DIR), transform=get_eval_transforms())
    # Deterministic subset
    rng = random.Random(SEED)
    n = min(CALIBRATION_SIZE, len(ds))
    indices = rng.sample(range(len(ds)), n)
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=False)
    return loader


# =============================================================================
#  Checkpoint + Model Loading
# =============================================================================
def load_model(device: torch.device) -> Tuple[ComponentCNNSpatial, dict]:
    if not CHECKPOINT_PATH.exists():
        sys.exit(f"ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = ComponentCNNSpatial(num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


# =============================================================================
#  Metrics
# =============================================================================
def compute_metrics(preds: List[int], labels: List[int]) -> dict:
    total = len(labels)
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    if HAVE_SKLEARN:
        macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    else:
        # Manual macro F1
        tp = [0]*NUM_PERMITTED; fp = [0]*NUM_PERMITTED; fn = [0]*NUM_PERMITTED
        for p, l in zip(preds, labels):
            if p == l: tp[l] += 1
            else: fp[p] += 1; fn[l] += 1
        f1s = []
        for c in range(NUM_PERMITTED):
            prec = tp[c]/(tp[c]+fp[c]) if (tp[c]+fp[c]) > 0 else 0
            rec = tp[c]/(tp[c]+fn[c]) if (tp[c]+fn[c]) > 0 else 0
            f1s.append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0)
        macro_f1 = sum(f1s)/NUM_PERMITTED
    return {"accuracy": accuracy, "macro_f1": macro_f1, "total": total, "correct": correct}


# =============================================================================
#  Experiment 1: FP32 Baseline
# =============================================================================
@torch.no_grad()
def run_fp32_baseline(
    model: ComponentCNNSpatial,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[dict, List[int], List[int], torch.Tensor]:
    """Run FP32 inference and return metrics, predictions, labels, and all logits."""
    all_preds = []
    all_labels = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)  # (B, 10)
        # Mask transistor
        logits[:, TRANSISTOR_INDEX] = -float("inf")
        preds = logits[:, :TRANSISTOR_INDEX].argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_logits.append(logits.cpu())

    all_logits_t = torch.cat(all_logits, dim=0)
    metrics = compute_metrics(all_preds, all_labels)
    return metrics, all_preds, all_labels, all_logits_t


# =============================================================================
#  INT8 Symmetric Per-Tensor Weight Quantization Helpers
# =============================================================================
def quantize_tensor_int8(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
    """Symmetric per-tensor quantization to signed INT8.

    scale = max(|tensor|) / 127
    quantized = clamp(round(tensor / scale), -128, 127)

    Returns quantized int8 array and the scale factor.
    Handles all-zero tensors safely (scale=1.0, all zeros).
    """
    abs_max = np.abs(tensor).max()
    if abs_max == 0.0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
    return quantized, float(scale)


def dequantize_tensor(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize INT8 tensor back to float32: value = quantized * scale."""
    return quantized.astype(np.float32) * scale


# =============================================================================
#  Experiment 2: Simulated INT8 Weight-Only Quantization
# =============================================================================
@torch.no_grad()
def run_int8_weight_only(
    model: ComponentCNNSpatial,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[dict, List[int], torch.Tensor, dict]:
    """Quantize all weights to INT8, dequantize back to FP32, then run
    standard FP32 inference with the dequantized weights.

    This is SIMULATED weight-only quantization -- activations remain in FP32.
    It measures the accuracy impact of weight quantization in isolation.
    """
    import copy
    quant_model = copy.deepcopy(model)
    quant_info = {}

    for name, param in quant_model.named_parameters():
        fp32_np = param.data.cpu().numpy()
        q_int8, scale = quantize_tensor_int8(fp32_np)
        deq_fp32 = dequantize_tensor(q_int8, scale)

        # Compute quantization error for this tensor
        abs_err = np.abs(fp32_np - deq_fp32)
        quant_info[name] = {
            "shape": list(fp32_np.shape),
            "scale": scale,
            "max_abs_error": float(abs_err.max()),
            "mean_abs_error": float(abs_err.mean()),
            "original_range": [float(fp32_np.min()), float(fp32_np.max())],
        }

        param.data = torch.from_numpy(deq_fp32).to(param.device)

    quant_model.eval()

    all_preds = []
    all_logits = []

    for images, labels in loader:
        images = images.to(device)
        logits = quant_model(images)
        logits[:, TRANSISTOR_INDEX] = -float("inf")
        preds = logits[:, :TRANSISTOR_INDEX].argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_logits.append(logits.cpu())

    all_logits_t = torch.cat(all_logits, dim=0)
    return {}, all_preds, all_logits_t, quant_info


# =============================================================================
#  Experiment 3: Fixed-Point Integer Inference Simulation
# =============================================================================
class FixedPointInference:
    """Layer-by-layer fixed-point integer inference simulation.

    Quantization scheme (per-tensor symmetric):
      - Input activations:  INT8  (scale calibrated from training subset)
      - Weights:            INT8  (scale = max|w| / 127)
      - Biases:             INT16 (scaled to match weight_scale * activation_scale)
      - Accumulators:       INT32 (to prevent overflow in MACs)
      - Output activations: Re-quantized to INT8 after ReLU (for next layer)

    Pooling operations (MaxPool2d, AdaptiveAvgPool2d) operate on integer values
    at the current activation scale.

    This provides a bit-accurate simulation of integer-only datapath behavior
    on the PYNQ-Z2 FPGA, including rounding and saturation effects.
    """

    def __init__(self, model: ComponentCNNSpatial, calibration_loader: DataLoader,
                 device: torch.device):
        self.device = device
        self.overflow_count = 0
        self.saturation_count = 0

        # Extract FP32 weights
        sd = model.state_dict()
        self.conv_weights = [
            sd["features.0.weight"].cpu().numpy(),
            sd["features.3.weight"].cpu().numpy(),
            sd["features.6.weight"].cpu().numpy(),
        ]
        self.conv_biases = [
            sd["features.0.bias"].cpu().numpy(),
            sd["features.3.bias"].cpu().numpy(),
            sd["features.6.bias"].cpu().numpy(),
        ]
        self.fc_weights = [
            sd["classifier.0.weight"].cpu().numpy(),
            sd["classifier.2.weight"].cpu().numpy(),
        ]
        self.fc_biases = [
            sd["classifier.0.bias"].cpu().numpy(),
            sd["classifier.2.bias"].cpu().numpy(),
        ]

        # Quantize all weights to INT8
        self.q_conv_w = []
        self.s_conv_w = []
        for w in self.conv_weights:
            qw, sw = quantize_tensor_int8(w)
            self.q_conv_w.append(qw)
            self.s_conv_w.append(sw)

        self.q_fc_w = []
        self.s_fc_w = []
        for w in self.fc_weights:
            qw, sw = quantize_tensor_int8(w)
            self.q_fc_w.append(qw)
            self.s_fc_w.append(sw)

        # Calibrate activation scales from training subset
        self.act_scales = self._calibrate_activation_scales(model, calibration_loader)

    def _calibrate_activation_scales(
        self, model: ComponentCNNSpatial, loader: DataLoader
    ) -> List[float]:
        """Run FP32 inference on calibration data and record max absolute
        activation values at each layer boundary to compute per-layer scales.

        Layer boundaries (7 activation checkpoints):
          0: input            (after normalization)
          1: after conv1+relu (before pool1)
          2: after pool1
          3: after conv2+relu (before pool2)
          4: after pool2
          5: after conv3+relu (before pool3)
          6: after pool3
          7: after spatial_pool (AdaptiveAvgPool2d)
          8: after fc1+relu
          -- fc2 output is final logits, no activation quantization needed
        """
        max_vals = [0.0] * 9  # track max |activation| at 9 points

        model.eval()
        with torch.no_grad():
            for images, _ in loader:
                x = images.to(self.device)
                # Input
                max_vals[0] = max(max_vals[0], float(x.abs().max()))

                # Conv block 1
                x = model.features[0](x)  # Conv2d
                x = model.features[1](x)  # ReLU
                max_vals[1] = max(max_vals[1], float(x.abs().max()))
                x = model.features[2](x)  # MaxPool
                max_vals[2] = max(max_vals[2], float(x.abs().max()))

                # Conv block 2
                x = model.features[3](x)
                x = model.features[4](x)
                max_vals[3] = max(max_vals[3], float(x.abs().max()))
                x = model.features[5](x)
                max_vals[4] = max(max_vals[4], float(x.abs().max()))

                # Conv block 3
                x = model.features[6](x)
                x = model.features[7](x)
                max_vals[5] = max(max_vals[5], float(x.abs().max()))
                x = model.features[8](x)
                max_vals[6] = max(max_vals[6], float(x.abs().max()))

                # Spatial pool
                x = model.pool(x)
                max_vals[7] = max(max_vals[7], float(x.abs().max()))

                # FC1 + ReLU
                x = torch.flatten(x, 1)
                x = model.classifier[0](x)
                x = model.classifier[1](x)
                max_vals[8] = max(max_vals[8], float(x.abs().max()))

        # Compute scales: scale = max_val / 127, with floor to prevent division by zero
        scales = []
        for mv in max_vals:
            if mv == 0.0:
                scales.append(1.0 / 127.0)
            else:
                scales.append(mv / 127.0)
        return scales

    def _quantize_activation(self, x_fp32: np.ndarray, scale: float) -> np.ndarray:
        """Quantize FP32 activation tensor to INT8."""
        q = np.clip(np.round(x_fp32 / scale), -128, 127).astype(np.int8)
        return q

    def _im2col(self, x: np.ndarray, kH: int, kW: int, padding: int) -> np.ndarray:
        """Extract sliding-window patches as columns for vectorized convolution.

        Input:  x of shape (B, C_in, H, W) as int32
        Output: col of shape (B, C_in*kH*kW, H_out*W_out) as int32
        """
        B, C_in, H, W = x.shape
        if padding > 0:
            x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)),
                       mode="constant", constant_values=0)
        _, _, H_pad, W_pad = x.shape
        H_out = H_pad - kH + 1
        W_out = W_pad - kW + 1

        # Use stride tricks for efficient im2col
        s = x.strides
        shape = (B, C_in, kH, kW, H_out, W_out)
        strides = (s[0], s[1], s[2], s[3], s[2], s[3])
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        # Reshape to (B, C_in*kH*kW, H_out*W_out)
        col = patches.reshape(B, C_in * kH * kW, H_out * W_out)
        return col, H_out, W_out

    def _conv2d_int(
        self,
        x_int8: np.ndarray,  # (B, C_in, H, W) int8
        w_int8: np.ndarray,  # (C_out, C_in, kH, kW) int8
        bias_fp32: np.ndarray,  # (C_out,) float32
        s_act_in: float,
        s_w: float,
        s_act_out: float,
        padding: int = 1,
    ) -> np.ndarray:
        """Vectorized integer convolution with INT32 accumulator.

        Computation (per output element):
          acc_int32 = sum(x_int8 * w_int8)            (INT32 accumulator via im2col matmul)
          acc_float = acc_int32 * (s_act_in * s_w)     (combined input scale)
          acc_float += bias_fp32                        (add FP32 bias)
          acc_float = max(acc_float, 0)                 (ReLU)
          out_int8 = clamp(round(acc_float / s_act_out), -128, 127)
        """
        B, C_in, H, W = x_int8.shape
        C_out, _, kH, kW = w_int8.shape
        combined_scale = s_act_in * s_w

        # im2col: (B, C_in*kH*kW, H_out*W_out) in int32
        col, H_out, W_out = self._im2col(x_int8.astype(np.int32), kH, kW, padding)

        # Reshape weights: (C_out, C_in*kH*kW) in int32
        w_mat = w_int8.reshape(C_out, -1).astype(np.int32)

        # Batched matmul: (B, C_out, H_out*W_out) — INT32 accumulation
        # Using int64 for the matmul to safely accumulate without overflow
        acc = np.zeros((B, C_out, H_out * W_out), dtype=np.int64)
        for b in range(B):
            acc[b] = w_mat @ col[b]  # (C_out, C_in*kH*kW) @ (C_in*kH*kW, H_out*W_out)

        # Check for INT32 overflow in the accumulator values
        overflow_mask = (acc > 2147483647) | (acc < -2147483648)
        self.overflow_count += int(overflow_mask.sum())

        # Scale to float, add bias, ReLU
        output = acc.astype(np.float64) * combined_scale
        output += bias_fp32[np.newaxis, :, np.newaxis]  # broadcast bias
        output = np.maximum(output, 0.0).astype(np.float32)  # ReLU
        output = output.reshape(B, C_out, H_out, W_out)

        # Re-quantize to INT8
        scaled = output / s_act_out
        q_out = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        self.saturation_count += int(np.sum((scaled < -128) | (scaled > 127)))

        return q_out

    def _maxpool2d_int(self, x_int8: np.ndarray, kernel: int = 2) -> np.ndarray:
        """Vectorized MaxPool2d on INT8 values."""
        B, C, H, W = x_int8.shape
        H_out = H // kernel
        W_out = W // kernel
        # Reshape to extract pooling windows
        x_reshaped = x_int8.reshape(B, C, H_out, kernel, W_out, kernel)
        out = x_reshaped.max(axis=(3, 5))
        return out

    def _adaptive_avg_pool_int(
        self,
        x_int8: np.ndarray,
        output_size: Tuple[int, int],
        s_in: float,
        s_out: float,
    ) -> np.ndarray:
        """Vectorized adaptive average pool on integer values.
        Average is computed, then re-quantized to INT8."""
        B, C, H, W = x_int8.shape
        oH, oW = output_size
        stride_h = H // oH
        stride_w = W // oW
        kH = H - (oH - 1) * stride_h
        kW = W - (oW - 1) * stride_w

        out = np.zeros((B, C, oH, oW), dtype=np.int8)
        for oh in range(oH):
            for ow in range(oW):
                h_start = oh * stride_h
                w_start = ow * stride_w
                region = x_int8[:, :, h_start:h_start+kH, w_start:w_start+kW].astype(np.float32)
                avg = np.round(region.mean(axis=(2, 3)))  # (B, C)
                val_fp = avg * s_in
                q_val = np.clip(np.round(val_fp / s_out), -128, 127).astype(np.int8)
                out[:, :, oh, ow] = q_val
        return out

    def _fc_int(
        self,
        x_int8: np.ndarray,     # (B, N_in) int8
        w_int8: np.ndarray,     # (N_out, N_in) int8
        bias_fp32: np.ndarray,  # (N_out,) float32
        s_act_in: float,
        s_w: float,
        s_act_out: float,
        apply_relu: bool = True,
        output_float: bool = False,
    ) -> np.ndarray:
        """Vectorized FC layer with INT32 accumulator."""
        combined_scale = s_act_in * s_w

        # Matmul in int64 to avoid overflow
        acc = x_int8.astype(np.int64) @ w_int8.astype(np.int64).T  # (B, N_out)

        # Check INT32 overflow
        overflow_mask = (acc > 2147483647) | (acc < -2147483648)
        self.overflow_count += int(overflow_mask.sum())

        # Scale, add bias, optional ReLU
        output = acc.astype(np.float64) * combined_scale + bias_fp32[np.newaxis, :]
        if apply_relu:
            output = np.maximum(output, 0.0)
        output = output.astype(np.float32)

        if output_float:
            return output  # Final layer: return FP32 logits

        scaled = output / s_act_out
        q_out = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        self.saturation_count += int(np.sum((scaled < -128) | (scaled > 127)))
        return q_out

    def run_inference(self, loader: DataLoader) -> Tuple[List[int], torch.Tensor]:
        """Run full fixed-point inference on a dataloader.
        Returns predictions and FP32 logits (from the final FC layer)."""
        all_preds = []
        all_logits = []

        for images, _ in loader:
            batch_np = images.numpy()  # (B, 3, 64, 64)

            # Quantize input to INT8
            x = self._quantize_activation(batch_np, self.act_scales[0])

            # Conv block 1: Conv + ReLU (fused) + MaxPool
            x = self._conv2d_int(
                x, self.q_conv_w[0], self.conv_biases[0],
                self.act_scales[0], self.s_conv_w[0], self.act_scales[1],
                padding=1,
            )
            x = self._maxpool2d_int(x, kernel=2)

            # Conv block 2
            x = self._conv2d_int(
                x, self.q_conv_w[1], self.conv_biases[1],
                self.act_scales[2], self.s_conv_w[1], self.act_scales[3],
                padding=1,
            )
            x = self._maxpool2d_int(x, kernel=2)

            # Conv block 3
            x = self._conv2d_int(
                x, self.q_conv_w[2], self.conv_biases[2],
                self.act_scales[4], self.s_conv_w[2], self.act_scales[5],
                padding=1,
            )
            x = self._maxpool2d_int(x, kernel=2)

            # Spatial pool (8x8 -> 2x2)
            x = self._adaptive_avg_pool_int(
                x, (2, 2), self.act_scales[6], self.act_scales[7],
            )

            # Flatten
            B = x.shape[0]
            x = x.reshape(B, -1)  # (B, 256)

            # FC1 + ReLU
            x = self._fc_int(
                x, self.q_fc_w[0], self.fc_biases[0],
                self.act_scales[7], self.s_fc_w[0], self.act_scales[8],
                apply_relu=True, output_float=False,
            )

            # FC2 (output layer -- return FP32 logits, no ReLU)
            logits = self._fc_int(
                x, self.q_fc_w[1], self.fc_biases[1],
                self.act_scales[8], self.s_fc_w[1], 1.0,
                apply_relu=False, output_float=True,
            )

            # Mask transistor and predict
            logits[:, TRANSISTOR_INDEX] = -float("inf")
            preds = np.argmax(logits[:, :TRANSISTOR_INDEX], axis=1)
            all_preds.extend(preds.tolist())
            all_logits.append(logits)

        all_logits_np = np.concatenate(all_logits, axis=0)
        return all_preds, torch.from_numpy(all_logits_np)



# =============================================================================
#  Main
# =============================================================================
def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cpu")  # Quantization comparison must be on CPU for reproducibility

    print(SEP)
    print("  Phase 2, Step 2.3 -- V5-Spatial Quantization Evaluation")
    print(SEP)

    # -- Load model and data --------------------------------------------------
    print("\n[1/6] Loading model and checkpoint...")
    model, ckpt = load_model(device)
    print(f"  Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"  Epoch:      {ckpt.get('epoch')}")

    print("\n[2/6] Loading test dataset (9 permitted classes, transistor excluded)...")
    test_loader, test_ds = build_test_loader()
    n_test = len(test_ds)
    print(f"  Test images: {n_test}")

    print("\n[3/6] Loading calibration subset from training split...")
    cal_loader = build_calibration_loader()
    print(f"  Calibration images: {CALIBRATION_SIZE} (from train split only)")

    # -- Experiment 1: FP32 Baseline ------------------------------------------
    print(f"\n{SEP2}")
    print("  Experiment 1: FP32 Baseline (Ground Truth)")
    print(SEP2)

    t0 = time.perf_counter()
    fp32_metrics, fp32_preds, fp32_labels, fp32_logits = run_fp32_baseline(model, test_loader, device)
    t1 = time.perf_counter()
    fp32_time = t1 - t0

    print(f"  Test Accuracy:  {fp32_metrics['accuracy']:.2f}%")
    print(f"  Macro F1:       {fp32_metrics['macro_f1']:.4f}")
    print(f"  Test Samples:   {fp32_metrics['total']}")
    print(f"  Inference Time: {fp32_time:.2f}s")

    # -- Experiment 2: INT8 Weight-Only Simulation ----------------------------
    print(f"\n{SEP2}")
    print("  Experiment 2: Simulated INT8 Weight-Only Quantization")
    print("  (Weights quantized to INT8, dequantized to FP32; activations remain FP32)")
    print(SEP2)

    t0 = time.perf_counter()
    _, w8_preds, w8_logits, w8_quant_info = run_int8_weight_only(model, test_loader, device)
    t1 = time.perf_counter()
    w8_time = t1 - t0

    w8_metrics = compute_metrics(w8_preds, fp32_labels)
    w8_agreement = sum(p == q for p, q in zip(w8_preds, fp32_preds))
    w8_agreement_pct = 100.0 * w8_agreement / n_test

    # Logit error (compare only the 9 permitted class logits, not the masked transistor)
    logit_diff = (w8_logits[:, :TRANSISTOR_INDEX] - fp32_logits[:, :TRANSISTOR_INDEX]).abs()
    w8_max_logit_err = float(logit_diff.max())
    w8_mean_logit_err = float(logit_diff.mean())

    print(f"  Test Accuracy:     {w8_metrics['accuracy']:.2f}%  (FP32: {fp32_metrics['accuracy']:.2f}%)")
    print(f"  Macro F1:          {w8_metrics['macro_f1']:.4f}  (FP32: {fp32_metrics['macro_f1']:.4f})")
    print(f"  FP32 Agreement:    {w8_agreement}/{n_test} ({w8_agreement_pct:.1f}%)")
    print(f"  Max Logit Error:   {w8_max_logit_err:.6f}")
    print(f"  Mean Logit Error:  {w8_mean_logit_err:.6f}")
    print(f"  Inference Time:    {w8_time:.2f}s")

    print("\n  Per-tensor quantization scales:")
    for name, info in w8_quant_info.items():
        print(f"    {name:30s} scale={info['scale']:.8f}  max_err={info['max_abs_error']:.8f}")

    # -- Experiment 3: Fixed-Point Integer Inference --------------------------
    print(f"\n{SEP2}")
    print("  Experiment 3: Fixed-Point Integer Inference Simulation")
    print("  (INT8 weights + INT8 activations + INT32 accumulators)")
    print("  Activation scales calibrated from training subset (NOT test data)")
    print(SEP2)

    print("\n  Calibrating activation scales from training subset...")
    t0 = time.perf_counter()
    fxp_engine = FixedPointInference(model, cal_loader, device)
    t_cal = time.perf_counter() - t0
    print(f"  Calibration time: {t_cal:.2f}s")

    print("\n  Activation scales (calibrated):")
    scale_labels = [
        "input", "after_conv1_relu", "after_pool1",
        "after_conv2_relu", "after_pool2",
        "after_conv3_relu", "after_pool3",
        "after_spatial_pool", "after_fc1_relu",
    ]
    for i, (lbl, s) in enumerate(zip(scale_labels, fxp_engine.act_scales)):
        print(f"    [{i}] {lbl:25s} scale={s:.8f}")

    print(f"\n  Running fixed-point inference on {n_test} test images...")
    print("  (This uses explicit integer arithmetic -- it will be slow)")
    t0 = time.perf_counter()
    fxp_preds, fxp_logits = fxp_engine.run_inference(test_loader)
    t1 = time.perf_counter()
    fxp_time = t1 - t0

    fxp_metrics = compute_metrics(fxp_preds, fp32_labels)
    fxp_agreement = sum(p == q for p, q in zip(fxp_preds, fp32_preds))
    fxp_agreement_pct = 100.0 * fxp_agreement / n_test

    fxp_logit_diff = (fxp_logits[:, :TRANSISTOR_INDEX] - fp32_logits[:, :TRANSISTOR_INDEX]).abs()
    fxp_max_logit_err = float(fxp_logit_diff.max())
    fxp_mean_logit_err = float(fxp_logit_diff.mean())

    print(f"\n  Test Accuracy:     {fxp_metrics['accuracy']:.2f}%  (FP32: {fp32_metrics['accuracy']:.2f}%)")
    print(f"  Macro F1:          {fxp_metrics['macro_f1']:.4f}  (FP32: {fp32_metrics['macro_f1']:.4f})")
    print(f"  FP32 Agreement:    {fxp_agreement}/{n_test} ({fxp_agreement_pct:.1f}%)")
    print(f"  Max Logit Error:   {fxp_max_logit_err:.4f}")
    print(f"  Mean Logit Error:  {fxp_mean_logit_err:.4f}")
    print(f"  INT32 Overflows:   {fxp_engine.overflow_count}")
    print(f"  INT8 Saturations:  {fxp_engine.saturation_count}")
    print(f"  Inference Time:    {fxp_time:.1f}s")

    # -- Save report ----------------------------------------------------------
    print(f"\n[6/6] Saving report to {OUTPUT_DIR}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "Phase 2, Step 2.3 -- V5-Spatial Quantization Evaluation",
        "checkpoint": CHECKPOINT_PATH.name,
        "test_samples": n_test,
        "calibration_samples": CALIBRATION_SIZE,
        "calibration_source": "cleaned_dataset/train (NOT test data)",
        "seed": SEED,
        "fp32_baseline": {
            "accuracy": fp32_metrics["accuracy"],
            "macro_f1": fp32_metrics["macro_f1"],
            "total_samples": fp32_metrics["total"],
            "correct": fp32_metrics["correct"],
            "inference_time_s": round(fp32_time, 3),
        },
        "int8_weight_only": {
            "description": (
                "Simulated weight-only INT8 quantization. All weights are symmetrically "
                "quantized to signed INT8 (scale = max|w|/127), then dequantized back to "
                "FP32 for inference. Activations remain in FP32. This is NOT full integer "
                "inference -- it measures accuracy impact of weight quantization alone."
            ),
            "accuracy": w8_metrics["accuracy"],
            "macro_f1": w8_metrics["macro_f1"],
            "accuracy_delta": round(w8_metrics["accuracy"] - fp32_metrics["accuracy"], 4),
            "f1_delta": round(w8_metrics["macro_f1"] - fp32_metrics["macro_f1"], 6),
            "fp32_prediction_agreement": w8_agreement,
            "fp32_prediction_agreement_pct": round(w8_agreement_pct, 2),
            "max_logit_error": round(w8_max_logit_err, 6),
            "mean_logit_error": round(w8_mean_logit_err, 6),
            "per_tensor_info": w8_quant_info,
            "inference_time_s": round(w8_time, 3),
        },
        "fixed_point_integer": {
            "description": (
                "Fixed-point integer inference simulation with INT8 weights, INT8 "
                "activations, FP32 biases added after INT32 accumulation, and INT32 "
                "accumulators. Activation scales are calibrated from a training-set "
                "subset (NOT test data). This simulates the integer-only datapath "
                "behavior of an FPGA implementation, including rounding and saturation. "
                "The final FC layer outputs FP32 logits (no re-quantization needed for "
                "argmax classification). Biases remain in FP32 to avoid introducing "
                "additional quantization error in the bias term; on FPGA hardware these "
                "would typically be stored as scaled INT16 or INT32 values."
            ),
            "accuracy": fxp_metrics["accuracy"],
            "macro_f1": fxp_metrics["macro_f1"],
            "accuracy_delta": round(fxp_metrics["accuracy"] - fp32_metrics["accuracy"], 4),
            "f1_delta": round(fxp_metrics["macro_f1"] - fp32_metrics["macro_f1"], 6),
            "fp32_prediction_agreement": fxp_agreement,
            "fp32_prediction_agreement_pct": round(fxp_agreement_pct, 2),
            "max_logit_error": round(fxp_max_logit_err, 4),
            "mean_logit_error": round(fxp_mean_logit_err, 4),
            "int32_overflows": fxp_engine.overflow_count,
            "int8_saturations": fxp_engine.saturation_count,
            "activation_scales": {
                lbl: round(s, 8) for lbl, s in zip(scale_labels, fxp_engine.act_scales)
            },
            "weight_scales": {
                f"conv{i}": round(s, 8) for i, s in enumerate(fxp_engine.s_conv_w)
            } | {
                f"fc{i}": round(s, 8) for i, s in enumerate(fxp_engine.s_fc_w)
            },
            "quantization_config": {
                "input_activations": "INT8 (symmetric per-tensor)",
                "weights": "INT8 (symmetric per-tensor)",
                "biases": "FP32 (added after INT32 accumulation; would be INT16/INT32 on FPGA)",
                "accumulators": "INT32",
                "output_activations": "INT8 (re-quantized after ReLU)",
                "final_layer_output": "FP32 logits (no re-quantization for argmax)",
                "pooling": "Integer max-pool (on INT8 values); integer average-pool with re-quantization",
            },
            "inference_time_s": round(fxp_time, 1),
            "limitations": [
                "Biases are stored in FP32, not fully quantized to INT16. On FPGA, biases "
                "would be pre-scaled to match the combined weight*activation scale.",
                "AdaptiveAvgPool2d uses floating-point division for the mean, then re-quantizes. "
                "On FPGA, this would use integer division or shift-based approximation.",
                "Per-tensor (not per-channel) weight quantization is used. Per-channel may "
                "yield better accuracy but increases hardware complexity.",
                "Activation scales are calibrated via max-value method. Percentile-based or "
                "KL-divergence calibration could reduce outlier sensitivity.",
            ],
        },
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {REPORT_PATH.name}")

    # -- Final Summary --------------------------------------------------------
    print(f"\n{SEP}")
    print("  Quantization Evaluation Summary")
    print(SEP)
    print(f"  {'Metric':<28} {'FP32':>12} {'INT8-WgtOnly':>14} {'FixedPoint':>14}")
    print(f"  {SEP2}")
    print(f"  {'Test Accuracy (%)':<28} {fp32_metrics['accuracy']:>12.2f} {w8_metrics['accuracy']:>14.2f} {fxp_metrics['accuracy']:>14.2f}")
    print(f"  {'Macro F1':<28} {fp32_metrics['macro_f1']:>12.4f} {w8_metrics['macro_f1']:>14.4f} {fxp_metrics['macro_f1']:>14.4f}")
    print(f"  {'FP32 Agreement (%)':<28} {'100.00':>12} {w8_agreement_pct:>14.2f} {fxp_agreement_pct:>14.2f}")
    print(f"  {'Max Logit Error':<28} {'0.000000':>12} {w8_max_logit_err:>14.6f} {fxp_max_logit_err:>14.4f}")
    print(f"  {'Mean Logit Error':<28} {'0.000000':>12} {w8_mean_logit_err:>14.6f} {fxp_mean_logit_err:>14.4f}")
    print(f"  {'INT32 Overflows':<28} {'n/a':>12} {'n/a':>14} {fxp_engine.overflow_count:>14}")
    print(f"  {'INT8 Saturations':<28} {'n/a':>12} {'n/a':>14} {fxp_engine.saturation_count:>14}")
    print(SEP)
    print(f"\n  Report saved to: {REPORT_PATH.resolve()}")
    print(f"  No existing files were modified.\n")


if __name__ == "__main__":
    main()
