"""
integer_v5_reference.py
-----------------------
Phase 2, Step 2.4.4 -- End-to-end integer-only Python reference for V5-Spatial
(ComponentCNNSpatial) electronic component classifier on PYNQ-Z2 FPGA.

Guarantees:
  - ZERO floating-point operations in the forward pass once input pixels are uint8.
  - ZERO PyTorch tensor operations in the integer pipeline (pure NumPy integer arrays).
  - Pure integer datapath:
      1. Camera pixel (uint8) -> Input LUT (256x8) -> Signed INT8 activations.
      2. Conv1: INT8 inputs * INT8 weights -> INT64/INT32 MAC + INT32 bias ->
                Integer ReLU -> Fixed-point Requantization (M, S) -> INT8 ->
                Integer MaxPool2d (2x2) -> INT8.
      3. Conv2: Same integer conv block structure.
      4. Conv3: Same integer conv block structure.
      5. Spatial Pool (AdaptiveAvgPool2d 8x8 -> 2x2):
                Integer 16-element sum (INT64) -> Integer division/rounding ->
                Fixed-point Requantization (M_pool, S_pool) -> INT8.
      6. FC1:   Flatten (256) -> INT8 inputs * INT8 weights -> INT64 MAC + INT32 bias ->
                Integer ReLU -> Fixed-point Requantization (M, S) -> INT8 (64).
      7. FC2:   INT8 inputs * INT8 weights -> INT64 MAC + INT32 bias ->
                Raw INT32/INT64 logits (10 classes).
      8. Argmax: Mask transistor (index 9) -> Argmax over classes 0..8.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import json
import numpy as np
from PIL import Image

# Default paths
PROJECT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = PROJECT_DIR / "fpga_export" / "v5_spatial"
INT_PARAMS_DIR = EXPORT_DIR / "integer_params"

# Permitted classes for the 9-class application (transistor index 9 masked)
PERMITTED_CLASSES = [
    "capacitor",
    "diode",
    "fuse",
    "inductor",
    "integrated_circuit",
    "led",
    "potentiometer",
    "push_button",
    "resistor",
]
TRANSISTOR_INDEX = 9
NUM_CLASSES = 10
INPUT_SIZE = 64


def integer_requantize_scalar(val: int, M: int, S: int, rounding: str = "half_up") -> int:
    """Scalar integer requantization for reference and unit testing.

    Formula: (val * M + rounding_bias) >> S
    Clamped to [-128, 127].

    Rounding options:
      - 'half_up': hardware standard `(prod + (1 << (S - 1))) >> S`
      - 'half_away_from_zero': symmetric rounding away from zero
      - 'round_even': round-to-nearest-even (matching IEEE 754 / np.round)
    """
    prod = int(val) * int(M)
    if S == 0:
        return int(np.clip(prod, -128, 127))

    half = 1 << (S - 1)
    if rounding == "half_up":
        # Standard hardware round-half-up
        res = (prod + half) >> S
    elif rounding == "half_away_from_zero":
        if prod >= 0:
            res = (prod + half) >> S
        else:
            res = -((-prod + half) >> S)
    elif rounding == "round_even":
        # Nearest-even rounding
        div = prod >> S
        rem = prod & ((1 << S) - 1)
        if rem > half:
            res = div + 1
        elif rem < half:
            res = div
        else:
            # Exactly on half: round to even
            res = div if (div % 2 == 0) else (div + 1)
    else:
        raise ValueError(f"Unknown rounding mode: {rounding}")

    return int(np.clip(res, -128, 127))


class IntegerV5Reference:
    """Pure integer implementation of V5-Spatial model."""

    def __init__(
        self,
        export_dir: Optional[Path] = None,
        use_16bit_requant: bool = False,
        pool_rounding: str = "round_even",
    ):
        self.export_dir = Path(export_dir) if export_dir else EXPORT_DIR
        self.int_params_dir = self.export_dir / "integer_params"
        self.use_16bit_requant = use_16bit_requant
        self.pool_rounding = pool_rounding

        # 1. Load Input LUT
        lut_path = self.int_params_dir / "input_lut_int8.npy"
        if not lut_path.exists():
            raise FileNotFoundError(f"Input LUT not found at: {lut_path}")
        self.input_lut = np.load(lut_path).astype(np.int8)
        assert self.input_lut.shape == (256,), f"Invalid LUT shape: {self.input_lut.shape}"

        # 2. Load Integer Biases (INT32)
        self.b_conv1 = np.load(self.int_params_dir / "conv1_bias_int32.npy").astype(np.int64)
        self.b_conv2 = np.load(self.int_params_dir / "conv2_bias_int32.npy").astype(np.int64)
        self.b_conv3 = np.load(self.int_params_dir / "conv3_bias_int32.npy").astype(np.int64)
        self.b_fc1 = np.load(self.int_params_dir / "fc1_bias_int32.npy").astype(np.int64)
        self.b_fc2 = np.load(self.int_params_dir / "fc2_bias_int32.npy").astype(np.int64)

        # 3. Load Multipliers and Shifts
        if self.use_16bit_requant:
            self.m = np.load(self.int_params_dir / "requant_multipliers_m16.npy").astype(np.int64)
            self.s = np.load(self.int_params_dir / "requant_shifts_s16.npy").astype(np.int32)
        else:
            self.m = np.load(self.int_params_dir / "requant_multipliers_m31.npy").astype(np.int64)
            self.s = np.load(self.int_params_dir / "requant_shifts_s31.npy").astype(np.int32)

        # Spatial Pool Requantization constant: ratio = s_pool3 / s_spatial_pool = 0.50742737 / 0.32026483 ~= 1.5843993
        # M_pool = 103835, S_pool = 16 (0 error across all 256 possible average values)
        self.m_pool = 103835
        self.s_pool = 16

        # 4. Load INT8 Weights (pre-quantized and exported to integer_params/)
        conv1_w_path = self.int_params_dir / "conv1_weight_int8.npy"
        if conv1_w_path.exists():
            self.w_conv1 = np.load(conv1_w_path).astype(np.int8)
            self.w_conv2 = np.load(self.int_params_dir / "conv2_weight_int8.npy").astype(np.int8)
            self.w_conv3 = np.load(self.int_params_dir / "conv3_weight_int8.npy").astype(np.int8)
            self.w_fc1 = np.load(self.int_params_dir / "fc1_weight_int8.npy").astype(np.int8)
            self.w_fc2 = np.load(self.int_params_dir / "fc2_weight_int8.npy").astype(np.int8)
        else:
            w_conv1_raw = np.load(self.export_dir / "features_0_weight.npy")
            w_conv2_raw = np.load(self.export_dir / "features_3_weight.npy")
            w_conv3_raw = np.load(self.export_dir / "features_6_weight.npy")
            w_fc1_raw = np.load(self.export_dir / "classifier_0_weight.npy")
            w_fc2_raw = np.load(self.export_dir / "classifier_2_weight.npy")

            self.w_conv1 = self._quantize_weight_tensor(w_conv1_raw)
            self.w_conv2 = self._quantize_weight_tensor(w_conv2_raw)
            self.w_conv3 = self._quantize_weight_tensor(w_conv3_raw)
            self.w_fc1 = self._quantize_weight_tensor(w_fc1_raw)
            self.w_fc2 = self._quantize_weight_tensor(w_fc2_raw)

        # Pre-flatten weights for matrix multiplication
        self.w_conv1_mat = self.w_conv1.reshape(16, -1).astype(np.int64)
        self.w_conv2_mat = self.w_conv2.reshape(32, -1).astype(np.int64)
        self.w_conv3_mat = self.w_conv3.reshape(64, -1).astype(np.int64)
        self.w_fc1_mat = self.w_fc1.astype(np.int64)
        self.w_fc2_mat = self.w_fc2.astype(np.int64)

    @staticmethod
    def _quantize_weight_tensor(w: np.ndarray) -> np.ndarray:
        """Quantize FP32 weight tensor to INT8 using symmetric per-tensor scale."""
        max_abs = float(np.abs(w).max())
        if max_abs == 0.0:
            scale = 1.0
        else:
            scale = max_abs / 127.0
        q = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
        return q

    def quantize_input_image(self, img_u8: np.ndarray) -> np.ndarray:
        """Map raw unsigned 8-bit RGB camera image to signed INT8 activations via LUT.

        Input:  img_u8 of shape (B, 3, 64, 64) with uint8 in [0, 255]
        Output: INT8 array of shape (B, 3, 64, 64) with values in [-127, 127]
        """
        assert img_u8.dtype == np.uint8, f"Expected uint8, got {img_u8.dtype}"
        return self.input_lut[img_u8]

    def _im2col_int(self, x: np.ndarray, kH: int = 3, kW: int = 3, padding: int = 1) -> Tuple[np.ndarray, int, int]:
        """Vectorized integer sliding window extraction using stride tricks.
        Zero-pads x with exact integer 0.
        """
        B, C_in, H, W = x.shape
        if padding > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)),
                           mode="constant", constant_values=0)
        else:
            x_pad = x

        _, _, H_pad, W_pad = x_pad.shape
        H_out = H_pad - kH + 1
        W_out = W_pad - kW + 1

        s = x_pad.strides
        shape = (B, C_in, kH, kW, H_out, W_out)
        strides = (s[0], s[1], s[2], s[3], s[2], s[3])
        patches = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides)
        col = patches.reshape(B, C_in * kH * kW, H_out * W_out)
        return col, H_out, W_out

    def _conv2d_block(
        self,
        x_int8: np.ndarray,      # (B, C_in, H, W) int8
        w_mat: np.ndarray,       # (C_out, C_in * 9) int64
        b_int32: np.ndarray,     # (C_out,) int64
        M: int,
        S: int,
        apply_pool: bool = True,
    ) -> np.ndarray:
        """Pure integer Conv2D + Bias Add + ReLU + Requantization + MaxPool2D block.

        Precision steps:
          1. MAC: int8 * int8 accumulated in signed int64
          2. Bias: add pre-scaled int32 bias directly to accumulator
          3. ReLU: int64 maximum with 0
          4. Requantization: (acc * M + (1 << (S - 1))) >> S, clipped to [-128, 127]
          5. MaxPool2d: 2x2 integer max on int8 values
        """
        B, C_in, H, W = x_int8.shape
        C_out = w_mat.shape[0]

        # 1. Im2Col
        col, H_out, W_out = self._im2col_int(x_int8.astype(np.int64), 3, 3, padding=1)

        # 2. Integer MAC: (B, C_out, H_out * W_out)
        acc = np.zeros((B, C_out, H_out * W_out), dtype=np.int64)
        for b in range(B):
            acc[b] = w_mat @ col[b]

        acc = acc.reshape(B, C_out, H_out, W_out)

        # 3. Integer Bias addition (pre-scaled to match MAC accumulator scale)
        acc += b_int32[np.newaxis, :, np.newaxis, np.newaxis]

        # 4. Integer ReLU (simplifies requantization datapath to non-negative values)
        acc = np.maximum(acc, 0)

        # 5. Fixed-Point Requantization (round-half-up integer arithmetic shift)
        if S > 0:
            rounding_bias = 1 << (S - 1)
            q_out = np.clip((acc * M + rounding_bias) >> S, -128, 127).astype(np.int8)
        else:
            q_out = np.clip(acc * M, -128, 127).astype(np.int8)

        # 6. Integer MaxPool2D (2x2, stride 2)
        if apply_pool:
            H_p = H_out // 2
            W_p = W_out // 2
            q_reshaped = q_out.reshape(B, C_out, H_p, 2, W_p, 2)
            out = q_reshaped.max(axis=(3, 5))
            return out
        return q_out

    def _spatial_adaptive_pool(self, x_int8: np.ndarray, rounding: Optional[str] = None) -> np.ndarray:
        """Pure integer AdaptiveAvgPool2D (8x8 -> 2x2).

        Averages 4x4 spatial regions (16 elements each).
        1. Accumulate 16 INT8 values into int64 sum (max sum: 16 * 127 = 2032).
        2. Integer division by 16 with round-to-nearest rounding:
           - 'round_even': Banker's round-to-even matching IEEE 754 / np.round
           - 'half_up': Hardware standard (sum + 8) >> 4
        3. Requantize to FC1 input scale using (M_pool, S_pool):
           q = ((avg * M_pool + (1 << (S_pool - 1))) >> S_pool), clipped to [-128, 127].
        """
        B, C, H, W = x_int8.shape
        assert H == 8 and W == 8, f"Expected 8x8 input to spatial pool, got {H}x{W}"

        mode = rounding if rounding is not None else self.pool_rounding

        # Non-overlapping 4x4 regions: (B, C, 2, 4, 2, 4)
        reshaped = x_int8.reshape(B, C, 2, 4, 2, 4).astype(np.int64)
        region_sums = reshaped.sum(axis=(3, 5))  # (B, C, 2, 2)

        # Rounding division by 16
        if mode == "round_even":
            div = region_sums // 16
            rem = region_sums % 16
            # Exact round-even at half-way boundary rem == 8, else (sum + 8) // 16
            int_avg = np.where(rem == 8, div + (div % 2), (region_sums + 8) // 16)
        elif mode == "half_up":
            # Hardware round-half-up (sum + 8) >> 4 for non-negative sums
            int_avg = (region_sums + 8) // 16
        else:
            raise ValueError(f"Unknown pool_rounding mode: {mode}")

        # Requantize to FC1 input scale: M=103835, S=16
        rounding_bias = 1 << (self.s_pool - 1)
        q_out = np.clip((int_avg * self.m_pool + rounding_bias) >> self.s_pool, -128, 127).astype(np.int8)
        return q_out

    def _fc1_block(self, x_int8: np.ndarray) -> np.ndarray:
        """Pure integer FC1: Matrix multiply + Bias + ReLU + Requantization."""
        B = x_int8.shape[0]
        x_flat = x_int8.reshape(B, 256).astype(np.int64)

        # Integer MAC
        acc = x_flat @ self.w_fc1_mat.T + self.b_fc1[np.newaxis, :]

        # Integer ReLU
        acc = np.maximum(acc, 0)

        # Requantization
        M = int(self.m[3])
        S = int(self.s[3])
        rounding_bias = 1 << (S - 1)
        q_out = np.clip((acc * M + rounding_bias) >> S, -128, 127).astype(np.int8)
        return q_out

    def _fc2_block(self, x_int8: np.ndarray) -> np.ndarray:
        """Pure integer FC2: Matrix multiply + Bias -> INT64/INT32 logits (10 classes)."""
        x_flat = x_int8.astype(np.int64)
        logits_int = x_flat @ self.w_fc2_mat.T + self.b_fc2[np.newaxis, :]
        return logits_int

    def forward_integer(self, img_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Complete integer forward pass.

        Input:  img_u8 (B, 3, 64, 64) in [0, 255] uint8
        Returns:
          - logits_int: (B, 10) signed integer logits
          - preds_9class: (B,) predicted class indices in [0, 8] (transistor masked)
        """
        # 1. Input LUT mapping
        x0 = self.quantize_input_image(img_u8)

        # 2. Conv Block 1: Conv + Bias + ReLU + Requant + Pool (64x64 -> 32x32)
        x1 = self._conv2d_block(x0, self.w_conv1_mat, self.b_conv1, int(self.m[0]), int(self.s[0]))

        # 3. Conv Block 2: Conv + Bias + ReLU + Requant + Pool (32x32 -> 16x16)
        x2 = self._conv2d_block(x1, self.w_conv2_mat, self.b_conv2, int(self.m[1]), int(self.s[1]))

        # 4. Conv Block 3: Conv + Bias + ReLU + Requant + Pool (16x16 -> 8x8)
        x3 = self._conv2d_block(x2, self.w_conv3_mat, self.b_conv3, int(self.m[2]), int(self.s[2]))

        # 5. Spatial Pool (AdaptiveAvgPool2d 8x8 -> 2x2)
        x4 = self._spatial_adaptive_pool(x3)

        # 6. FC1 + ReLU + Requant
        x5 = self._fc1_block(x4)

        # 7. FC2 Output Logits
        logits_int = self._fc2_block(x5)

        # 8. Transistor Masking and 9-Class Argmax
        # Since logits share a strictly positive scale factor, argmax on integer logits
        # is mathematically invariant. Slicing columns 0..8 masks transistor (index 9).
        preds_9class = np.argmax(logits_int[:, :TRANSISTOR_INDEX], axis=1)

        return logits_int, preds_9class

    def predict_image(self, img: Union[Image.Image, np.ndarray]) -> Tuple[int, str, np.ndarray]:
        """Predict a single PIL image or uint8 array.

        Returns (class_index, class_name, raw_logits).
        """
        if isinstance(img, Image.Image):
            # Bilinear resize to 64x64 and convert to RGB
            img_resized = img.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            arr = np.array(img_resized, dtype=np.uint8)
            u8 = np.transpose(arr, (2, 0, 1))[np.newaxis, :]
        elif isinstance(img, np.ndarray):
            if img.ndim == 3 and img.shape[2] == 3:
                u8 = np.transpose(img, (2, 0, 1))[np.newaxis, :]
            elif img.ndim == 4:
                u8 = img
            else:
                raise ValueError(f"Unsupported image shape: {img.shape}")
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        logits, preds = self.forward_integer(u8)
        pred_idx = int(preds[0])
        pred_name = PERMITTED_CLASSES[pred_idx]
        return pred_idx, pred_name, logits[0]
