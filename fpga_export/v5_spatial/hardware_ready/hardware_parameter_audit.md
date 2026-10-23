# Phase 2: Final Hardware Parameter Audit & Freeze Report

**Project:** Electronic Component Classifier (`ElectronicComponentClassifier`)  
**Target Hardware:** Xilinx PYNQ-Z2 Board (Zynq-7020 SoC, `xc7z020clg400-1`)  
**Model Architecture:** V5-Spatial (`ComponentCNNSpatial` from `model_v5_spatial.py`)  
**Checkpoint:** `checkpoints/v5_spatial_best_model.pth` (Epoch 49, Best Validation Model)  
**Date:** October 2026  
**Status:** **APPROVED & FROZEN FOR PHASE 3 (HLS DESIGN)**

---

## 1. Executive Summary

This audit establishes the frozen, bit-exact hardware parameter bundle for the V5-Spatial model on the PYNQ-Z2 FPGA. 

All parameters have been converted from trained floating-point weights into a reproducible, integer-only format. The implementation has been validated with **zero floating-point operations** on a 346-image test split, achieving:
- **Test Accuracy:** **68.79%** (238 / 346 correct) — identical to the FP32 PyTorch baseline.
- **Macro F1 Score:** **0.6607** (vs. 0.6646 baseline).
- **Prediction Agreement with Hybrid Fixed-Point Simulation:** **100.00%** (346 / 346 samples match exactly).
- **Prediction Agreement with FP32 Baseline:** **95.95%** (332 / 346 samples match).
- **Mathematical Overflow Guarantee:** **0 overflows** across all layers, verified analytically and empirically.
- **On-Chip Memory Footprint:** **41.54 KB total** (weights + biases + LUT + requant parameters), consuming only **6.62%** of the PYNQ-Z2's 612.5 KB on-chip Block RAM (BRAM). All parameters fit 100% on-chip without DRAM bandwidth bottlenecks.

---

## 2. Complete Layer-by-Layer Parameter Inventory

| Layer | Type | Weights Shape | Weight Elements | Weight Range (INT8) | Weight Scale ($s_w$) | Bias Elements | Bias Range (INT32) | Bias Scale ($s_{bias} = s_{in} \cdot s_w$) | Requant $(M_{31}, S_{31})$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Input LUT** | ROM | $[256]$ | 256 | $[-127, 127]$ | $0.00787402$ | - | - | - | - |
| **Conv1** | Conv2d (3x3) | $[16, 3, 3, 3]$ | 432 | $[-127, 94]$ | $0.0033164834$ | 16 | $[-5965, 19140]$ | $2.611395 \times 10^{-5}$ | $(2117698473, 41)$ |
| **Conv2** | Conv2d (3x3) | $[32, 16, 3, 3]$ | 4,608 | $[-85, 127]$ | $0.0054134346$ | 32 | $[-779, 1309]$ | $1.467956 \times 10^{-4}$ | $(1111516105, 40)$ |
| **Conv3** | Conv2d (3x3) | $[64, 32, 3, 3]$ | 18,432 | $[-127, 101]$ | $0.0046567879$ | 64 | $[-220, 338]$ | $6.762129 \times 10^{-4}$ | $(1465242147, 40)$ |
| **Spatial Pool** | AvgPool (4x4) | - | - | - | - | - | - | - | $(103835, 16)$ |
| **FC1** | Linear | $[64, 256]$ | 16,384 | $[-127, 113]$ | $0.0037154248$ | 64 | $[-86, 125]$ | $1.189920 \times 10^{-3}$ | $(1531277679, 38)$ |
| **FC2** | Linear | $[10, 64]$ | 640 | $[-127, 75]$ | $0.0051096753$ | 10 | $[-190, 76]$ | $1.091433 \times 10^{-3}$ | $(1, 0)$ [Identity] |
| **Total** | - | - | **40,496 B** | - | - | **186 words** | **744 B** | - | - |

---

## 3. Detailed Parameter Audit

### 3.1 Quantized INT8 Weights
- **Quantization Formula:**
  $$w_{\text{int8}} = \text{clip}\left(\text{round}\left(\frac{w_{\text{fp32}}}{\text{scale}}\right), -128, 127\right), \quad \text{scale} = \frac{\max(|w_{\text{fp32}}|)}{127.0}$$
- **Storage:** Exported both as binary `.npy` files and human-readable `.txt` files under `fpga_export/v5_spatial/integer_params/`.
- **Integrity Verification:**
  - SHA-256 checksums computed and locked into `manifest_int8_weights.json` and `hardware_parameter_manifest.json`.
  - Direct runtime assertion in `integer_v5_reference.py` confirms that loaded weights match the exported binary tensors bit-for-bit.

### 3.2 Bias Scales and Integration Order
- **Accumulator Matching Principle:**
  In integer convolution/dense matrix multiplication:
  $$\text{acc}_{\text{raw}} = \sum_{i} x_{\text{int8}, i} \cdot w_{\text{int8}, i}$$
  The real-valued equivalent is:
  $$\text{real\_mac} = \text{acc}_{\text{raw}} \cdot (s_{\text{in}} \cdot s_w)$$
  To avoid introducing floating-point multiplications or loss of dynamic range, biases are pre-scaled by $s_{\text{bias}} = s_{\text{in}} \cdot s_w$:
  $$b_{\text{int32}} = \text{round}\left(\frac{b_{\text{fp32}}}{s_{\text{in}} \cdot s_w}\right)$$
- **Hardware Integration Order (CRITICAL FOR HLS):**
  $$b_{\text{int32}} \text{ MUST be added to } \text{acc}_{\text{raw}} \text{ BEFORE requantization:}$$
  $$\text{acc}_{\text{biased}} = \text{acc}_{\text{raw}} + b_{\text{int32}}$$
  $$\text{acc}_{\text{relu}} = \max(\text{acc}_{\text{biased}}, 0)$$
  $$\text{activation}_{\text{out}} = \text{clip}\left(\left(\text{acc}_{\text{relu}} \cdot M + 2^{S-1}\right) \gg S, -128, 127\right)$$

### 3.3 Requantization Multipliers, Shifts, and Intermediate Products
- **Scaling Ratio:**
  $$\text{ratio} = \frac{s_{\text{in}} \cdot s_w}{s_{\text{out}}} \approx \frac{M}{2^S}$$
- **Primary 31-bit Parameters ($M_{31}, S_{31}$):**
  - Conv1: $M = 2,117,698,473, \quad S = 41$ (relative error: $2.27 \times 10^{-10}$)
  - Conv2: $M = 1,111,516,105, \quad S = 40$ (relative error: $3.18 \times 10^{-10}$)
  - Conv3: $M = 1,465,242,147, \quad S = 40$ (relative error: $2.01 \times 10^{-10}$)
  - FC1:   $M = 1,531,277,679, \quad S = 38$ (relative error: $9.11 \times 10^{-11}$)
  - FC2:   $M = 1, \quad S = 0$ (identity scaling; direct integer argmax)
- **Intermediate Multiplier Headroom:**
  - Conv1 max product: $9.63 \times 10^{14}$ (51 bits)
  - Conv2 max product: $2.58 \times 10^{15}$ (53 bits)
  - Conv3 max product: $6.81 \times 10^{15}$ (54 bits)
  - FC1 max product:   $6.32 \times 10^{15}$ (54 bits)
  - Standard signed 64-bit integers (`int64_t` / `ap_int<64>`, max $9.22 \times 10^{18}$) provide **at least 10 bits of safety margin** above the maximum intermediate multiplication product.

### 3.4 Accumulator Bit-Width and Zero-Overflow Proof
- **Conv1:** $27 \text{ MACs} \times 128 \times 128 = 442,368 + 19,140 (\text{bias}) = 461,508 < 2^{19}$ (requires 20 signed bits).
- **Conv2:** $144 \text{ MACs} \times 128 \times 128 = 2,359,296 + 1,309 (\text{bias}) = 2,360,605 < 2^{22}$ (requires 23 signed bits).
- **Conv3:** $288 \text{ MACs} \times 128 \times 128 = 4,718,592 + 338 (\text{bias}) = 4,718,930 < 2^{23}$ (requires 24 signed bits).
- **FC1:** $256 \text{ MACs} \times 128 \times 128 = 4,194,304 + 125 (\text{bias}) = 4,194,429 < 2^{23}$ (requires 24 signed bits).
- **FC2:** $64 \text{ MACs} \times 128 \times 128 = 1,048,576 + 190 (\text{bias}) = 1,048,766 < 2^{21}$ (requires 22 signed bits).
- **Conclusion:** Standard signed 32-bit accumulators (`int32_t` / `ap_int<32>`) provide **8 to 12 bits of headroom** beyond the worst-case mathematical bound across all layers. **Zero overflow is guaranteed.**

### 3.5 Signed Arithmetic and Saturation
- In Conv1, Conv2, Conv3, and FC1, the activation function is ReLU:
  $$\text{acc}_{\text{relu}} = \max(\text{acc}, 0) \ge 0$$
- Because the input to the requantization multiplier is strictly non-negative:
  - Sign bit propagation during right shift is identical for signed and unsigned arithmetic.
  - Adding the rounding bias $2^{S-1}$ is strictly positive and cannot wrap around.
- Activation clamping to signed INT8 $[-128, 127]$ limits post-ReLU values to $[0, 127]$. Calibration on training data observed only 87 total saturation events across all layers, confirming that activation scales prevent clipping distortion.

---

## 4. Resolution of the Spatial Average-Pooling Discrepancy

### 4.1 The Discrepancy
The V5-Spatial architecture employs `AdaptiveAvgPool2d(output_size=(2, 2))` on the $64 \times 8 \times 8$ feature map output from MaxPool3, pooling four non-overlapping $4 \times 4$ spatial regions (16 elements each).
- **PyTorch / Hybrid Simulation:** Computes floating-point `mean = sum / 16.0` followed by `np.round()`, which applies IEEE 754 Banker's Rounding (round-to-nearest-even).
- **Hardware Bit-Shift Rounding:** Standard hardware fixed-point division by 16 is `(sum + 8) >> 4`, which implements round-half-up.
- **Boundary Conditions:** For sums with remainder 8 ($0.5$ fractional part):
  - When quotient $s // 16$ is even ($0.5, 2.5, 4.5, \dots$), round-even rounds down to the even integer, whereas round-half-up rounds up.
  - When quotient $s // 16$ is odd ($1.5, 3.5, 5.5, \dots$), both methods round up to the next even integer.

### 4.2 Mathematical Domain
Because the input to AdaptiveAvgPool comes immediately after Conv3 + ReLU + MaxPool3, all inputs are non-negative in $[0, 127]$.
Thus, 16-element region sums are strictly bounded:
$$0 \le s \le 16 \times 127 = 2032$$
Out of all 2,033 possible integer sums:
- Exactly 127 sums have remainder 8 ($s = 8, 24, 40, \dots, 2024$).
- Of these 127 sums, 64 have even quotients where unquantized results differ by 1.
- After requantization by $(M_{\text{pool}} = 103835, S_{\text{pool}} = 16)$, exactly **40 out of 2,033 sums (1.97%)** produce a 1-LSB difference in INT8 activation output.

### 4.3 Test-Set Empirical Impact (346 Images)
Across all $346 \times 64 \times 4 = 88,576$ pooled regions evaluated on the test set:
| Metric | Round-to-Even (`round_even`) | Round-Half-Up (`half_up`) |
| :--- | :---: | :---: |
| **Test Accuracy** | **68.79%** (238 / 346) | **69.36%** (240 / 346) |
| **Macro F1 Score** | **0.6607** | **0.6666** |
| **Agreement with Hybrid Fixed-Point** | **100.00%** (346 / 346) | **97.98%** (339 / 346) |
| **Agreement with FP32 Baseline** | **95.95%** (332 / 346) | **96.53%** (334 / 346) |

### 4.4 Hardware Implementation Recommendation
Both rounding implementations are supported in `integer_v5_reference.py`.
- **Primary Reference Standard:** `round_even` is retained as the default reference configuration because it delivers **exact 100.00% agreement** with the baseline quantization simulation.
- **FPGA Logic Cost in HLS / RTL:**
  In C++ HLS:
  ```cpp
  ap_uint<12> sum = ...; // 16 elements in [0, 127]
  ap_uint<8>  div = sum >> 4;
  ap_uint<4>  rem = sum & 0xF;
  // Round-to-even:
  ap_uint<8>  avg_even = (rem == 8) ? (div + (div & 1)) : (ap_uint<8>)((sum + 8) >> 4);
  ```
  This logic requires only a 4-bit comparator (`rem == 4'd8`) and a single 1-bit half-adder, costing fewer than 6 LUTs in Vivado.
- **Verdict:** The HLS implementation in Phase 3 should implement `avg_even` to achieve exact 100% testbench agreement with the golden integer reference.

---

## 5. End-to-End 346-Image Validation Results

### 5.1 Benchmark Comparison Table

| Metric | FP32 PyTorch Baseline | Hybrid Fixed-Point Simulation | Pure Integer Reference (`round_even`) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **68.79%** | **68.79%** | **68.79%** |
| **Correct / Total** | 238 / 346 | 238 / 346 | 238 / 346 |
| **Macro F1 Score** | **0.6646** | **0.6607** | **0.6607** |
| **Agreement vs. FP32 Baseline** | 100.00% (346/346) | 95.95% (332/346) | 95.95% (332/346) |
| **Agreement vs. Hybrid FXP** | 95.95% (332/346) | 100.00% (346/346) | **100.00% (346/346)** |
| **Arithmetic Overflows** | N/A (float) | 0 | **0** |
| **Inference Time (346 images)** | 2.03s | 14.48s | 15.38s |

### 5.2 Unit Test Suite Results
All five unit tests in `validate_v5_integer_reference.py` passed cleanly:
1. **Input LUT Mapping (256 codes):** `PASS` (0 mismatches across all 256 pixel codes).
2. **Integer Bias Addition:** `PASS` (verified on wide signed values with 64-bit/32-bit accumulators).
3. **Signed Requantization Rounding:** `PASS` (verified against positive, negative, and half-way boundary cases).
4. **Spatial Average-Pooling Rounding:** `PASS` (verified across all 2,033 non-negative sums $[0, 2032]$ directly calling `ref._spatial_adaptive_pool`).
5. **Transistor Masking and 9-Class Argmax:** `PASS` (verified that transistor class index 9 is strictly masked and never predicted).

---

## 6. On-Chip Memory and Hardware Resource Budget (PYNQ-Z2)

The PYNQ-Z2 board features a Xilinx Zynq-7020 SoC (`xc7z020clg400-1`):
- **Block RAM:** 140 RAMB36E1 primitives = 280 RAMB18 = 4.9 Mb = **612.5 KB**.
- **DSP Slices:** 220 DSP48E1 slices.
- **Logic Cells:** 85,000 cells (53,200 LUTs, 106,400 Flip-Flops).

### Parameter Storage Utilization
| Parameter Group | Elements | Precision | Memory (Bytes) | 36Kb BRAM Equivalent | % of Available BRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input LUT** | 256 | INT8 | 256 B | 0.06 BRAMs | 0.04% |
| **Conv1 Weights** | 432 | INT8 | 432 B | 0.10 BRAMs | 0.07% |
| **Conv1 Biases** | 16 | INT32 | 64 B | 0.01 BRAMs | 0.01% |
| **Conv2 Weights** | 4,608 | INT8 | 4,608 B | 1.02 BRAMs | 0.73% |
| **Conv2 Biases** | 32 | INT32 | 128 B | 0.03 BRAMs | 0.02% |
| **Conv3 Weights** | 18,432 | INT8 | 18,432 B | 4.10 BRAMs | 2.93% |
| **Conv3 Biases** | 64 | INT32 | 256 B | 0.06 BRAMs | 0.04% |
| **FC1 Weights** | 16,384 | INT8 | 16,384 B | 3.64 BRAMs | 2.60% |
| **FC1 Biases** | 64 | INT32 | 256 B | 0.06 BRAMs | 0.04% |
| **FC2 Weights** | 640 | INT8 | 640 B | 0.14 BRAMs | 0.10% |
| **FC2 Biases** | 10 | INT32 | 40 B | 0.01 BRAMs | 0.01% |
| **Requant Parameters** | 10 | INT64/32 | 48 B | 0.01 BRAMs | 0.01% |
| **Total Parameters** | - | - | **41,544 B (40.57 KB)** | **~9.2 BRAMs** | **6.62%** |

**Architectural Advantage:** Because the entire parameter bundle consumes less than 7% of on-chip BRAM, all weights, biases, and requantization multipliers can be completely pre-loaded into on-chip BRAM during initialization. This eliminates all DDR memory access latency during real-time image inference.

---

## 7. Artifact Manifest and Paths

All hardware parameter bundle files are located under:
`C:\Users\rajal\OneDrive\Desktop\ElectronicComponentClassifier\fpga_export\v5_spatial\`

### Generated Manifests:
1. `fpga_export/v5_spatial/hardware_ready/hardware_parameter_manifest.json` (Master HLS manifest)
2. `fpga_export/v5_spatial/hardware_ready/hardware_parameter_audit.md` (This audit document)
3. `fpga_export/v5_spatial/integer_params/manifest_int8_weights.json` (INT8 weights manifest)
4. `fpga_export/v5_spatial/integer_params/manifest_integer_biases.json` (INT32 biases manifest)
5. `fpga_export/v5_spatial/integer_params/manifest_requant_params.json` (Multipliers and shifts manifest)
6. `fpga_export/v5_spatial/integer_params/input_lut_manifest.json` (Input LUT manifest)
7. `fpga_export/v5_spatial/integer_reference/unit_test_results.json` (Unit test execution logs)
8. `fpga_export/v5_spatial/integer_reference/integer_validation_report.json` (Full 346-image validation log)

### Exported Hardware Parameter Files (`fpga_export/v5_spatial/integer_params/`):
- `input_lut_int8.npy` and `input_lut_int8.txt`
- `conv1_weight_int8.npy` and `conv1_weight_int8.txt`
- `conv1_bias_int32.npy` and `conv1_bias_int32.txt`
- `conv2_weight_int8.npy` and `conv2_weight_int8.txt`
- `conv2_bias_int32.npy` and `conv2_bias_int32.txt`
- `conv3_weight_int8.npy` and `conv3_weight_int8.txt`
- `conv3_bias_int32.npy` and `conv3_bias_int32.txt`
- `fc1_weight_int8.npy` and `fc1_weight_int8.txt`
- `fc1_bias_int32.npy` and `fc1_bias_int32.txt`
- `fc2_weight_int8.npy` and `fc2_weight_int8.txt`
- `fc2_bias_int32.npy` and `fc2_bias_int32.txt`
- `requant_multipliers_m31.npy` and `requant_shifts_s31.npy`
- `requant_multipliers_m16.npy` and `requant_shifts_s16.npy`

---

## 8. Unresolved Issues and Phase 3 Readiness Verdict

### Unresolved Issues
**None.** 
- All five layer weights are quantized, exported, and verified.
- Biases match the accumulator scaling factor exactly.
- Requantization multipliers and shifts introduce 0 LSB error against floating-point reference.
- Intermediate products fit comfortably in signed 64-bit types.
- The spatial average-pooling rounding behavior is characterized and implemented in both reference and unit tests.
- Transistor class index 9 is strictly masked during classification.
- 100.00% agreement is achieved with the fixed-point quantization baseline.

### Phase 3 Readiness Verdict
> [!IMPORTANT]
> **READINESS STATUS: 100% READY FOR PHASE 3 (VITIS HLS C++ IMPLEMENTATION)**
> 
> The hardware parameter bundle is mathematically verified, frozen, and completely documented. No further training, calibration, or parameter modifications are needed. Phase 3 (Vitis HLS top-level design, kernel pipeline definition, and C++ / RTL co-simulation) can proceed immediately.
