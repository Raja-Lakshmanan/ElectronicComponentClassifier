# Mathematical and Implementation Audit: V5-Spatial Quantization & Fixed-Point Inference
**Project:** Electronic Component Classifier (`ElectronicComponentClassifier`)  
**Target Hardware:** PYNQ-Z2 (Xilinx Zynq-7020 SoC FPGA)  
**Audited Files:**  
- `evaluate_v5_quantization.py`
- `fpga_export/v5_spatial/quantization_eval/quantization_report.json`
- `fpga_export/v5_spatial/manifest.json`  
**Audit Date:** October 2026  
**Auditor:** Antigravity Pair Programmer  

---

## 1. Trace of the Fixed-Point Forward Pass (True Integer vs. Floating-Point)

An inspection of `FixedPointInference` in `evaluate_v5_quantization.py` reveals that the simulation is a **hybrid integer-multiply / floating-point-scaled execution**, rather than a purely synthesizable integer datapath.

### Layer-by-Layer Arithmetic Trace

| Layer | Operation | Source Lines | Arithmetic Precision Used in Python | Hardware Equivalent Needed for Synthesizable HLS |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | RGB Normalization to INT8 | Lines 408–411, 571 | **Floating-point**: `batch_np / scale`, then `np.round` and cast to `int8` | 256-entry 8-bit ROM/LUT or integer fixed-point transform |
| **Conv1** | Sliding window & Im2Col | Lines 413–434, 460 | **Integer indexing**: array padding and stride manipulation | 2-line circular BRAM shift buffers |
| **Conv1** | 2D Convolution MAC | Lines 463–469 | **TRUE INTEGER**: `w_mat.astype(int32) @ col.astype(int32)` accumulated into `int64`/`int32` | DSP48E1 $25 \times 18$ signed MAC |
| **Conv1** | Accumulator Scaling & Bias | Lines 476–477 | **FLOATING-POINT**: `acc.astype(float64) * scale + bias_fp32` | Integer bias add in accumulator + integer fixed-point multiplier $(M_0 \times 2^{-S})$ |
| **Conv1** | ReLU Activation | Line 478 | **FLOATING-POINT**: `np.maximum(output, 0.0).astype(float32)` | Bitwise sign-bit check on integer accumulator: `if (acc < 0) acc = 0;` |
| **Conv1** | Requantization to INT8 | Lines 482–484 | **FLOATING-POINT**: `output / s_act_out`, then `np.round` and `np.clip` | Arithmetic shift-right with rounding and saturation logic |
| **Pool1** | MaxPool2d (kernel=2, stride=2) | Lines 488–496, 579 | **TRUE INTEGER**: `np.max` on `int8` values | 2x2 integer comparator pipeline |
| **Conv2** | 2D Convolution MAC | Lines 463–469, 582 | **TRUE INTEGER**: `int32` dot product | DSP48E1 parallel MAC |
| **Conv2** | Scaling, Bias, ReLU, Requant | Lines 476–484 | **FLOATING-POINT**: Float multiplication, bias add, float ReLU, float divide | Fixed-point scale multiplier + integer shift |
| **Pool2** | MaxPool2d (kernel=2, stride=2) | Lines 488–496, 587 | **TRUE INTEGER**: `np.max` on `int8` values | 2x2 integer comparator |
| **Conv3** | 2D Convolution MAC | Lines 463–469, 590 | **TRUE INTEGER**: `int32` dot product | DSP48E1 parallel MAC |
| **Conv3** | Scaling, Bias, ReLU, Requant | Lines 476–484 | **FLOATING-POINT**: Float multiplication, bias add, float ReLU, float divide | Fixed-point scale multiplier + integer shift |
| **Pool3** | MaxPool2d (kernel=2, stride=2) | Lines 488–496, 595 | **TRUE INTEGER**: `np.max` on `int8` values | 2x2 integer comparator |
| **SpatialPool**| AdaptiveAvgPool2d (8x8 -> 2x2)| Lines 498–524, 598 | **FLOATING-POINT**: `region.mean()` in float32, then `val_fp = avg * s_in`, `val_fp / s_out` | Accumulate 16 INT8 values, arithmetic shift-right by 4 bits (`>> 4`) |
| **FC1** | Matrix Multiplication | Line 541, 607 | **TRUE INTEGER**: `x_int8.astype(int64) @ w_int8.astype(int64).T` | Parallel DSP MAC tree |
| **FC1** | Scaling, Bias, ReLU, Requant | Lines 548–558 | **FLOATING-POINT**: Float multiplication, bias add, float ReLU, float divide | Integer bias add + fixed-point multiplier + shift |
| **FC2** | Matrix Multiplication | Line 541, 614 | **TRUE INTEGER**: `int64` matrix product | Parallel DSP MAC tree |
| **FC2** | Scaling & Bias Addition | Line 548 | **FLOATING-POINT**: Float multiply and float bias add, returning FP32 logits | Scaled integer addition or final fixed-point logit output |
| **Predict** | Transistor Mask & Argmax | Lines 621–622 | **FLOATING-POINT**: `logits[:, 9] = -inf`, `np.argmax(logits[:, :9])` | Integer comparator over classes 0..8 (ignoring index 9) |

---

## 2. Input Conversion: Normalized $[-1.0, 1.0]$ RGB to INT8

### Python Implementation
- Transform pipeline (lines 97–103):
  $$\text{ToTensor}(P) \in [0.0, 1.0] \quad \implies \quad x = \frac{P/255.0 - 0.5}{0.5} = \frac{2P}{255} - 1.0 \in [-1.0, 1.0]$$
- Input activation scale calibration (lines 365–366, 405):
  $$\text{Max absolute value observed on training subset} = 1.0 \quad \implies \quad s_{\text{in}} = \frac{1.0}{127} \approx 0.0078740157$$
- Quantization formula (lines 408–411, 571):
  $$q(P) = \text{clip}\left(\text{round}\left(\frac{x}{s_{\text{in}}}\right), -128, 127\right) = \text{clip}\left(\text{round}(127 \times x), -128, 127\right)$$

### Mathematical Discrepancy with Standard FPGA Fixed-Point Preprocessing
Expanding $q(P)$ in terms of raw unsigned 8-bit camera pixel $P \in [0, 255]$:
$$q(P) = \text{round}\left(127 \times \left(\frac{2P - 255}{255}\right)\right) = \text{round}\left(\frac{254 P - 32385}{255}\right)$$
Many FPGA designers attempt to simplify this by directly casting the 8-bit camera pixel via unsigned-to-signed offset:
$$q_{\text{naive}}(P) = P - 128$$
- Evaluating $q(P)$ vs. $q_{\text{naive}}(P)$ across all 256 pixel values demonstrates that $q(P) - q_{\text{naive}}(P)$ differs by $\pm 1$ on **128 out of 256 pixel values (50.0%)**.
- **Impact for HLS:** If the HLS synthesizer uses naive $(P - 128)$ instead of the calibrated transform, input activations will incur a 1-LSB systematic offset.
- **HLS Solution:** Implement a small 256-byte Block ROM (lookup table) $q[P]$ in FPGA fabric, which executes in exactly 1 clock cycle with zero DSP usage and guarantees bit-exact agreement with the Python simulation.

---

## 3. Weight and Bias Quantization Analysis

### Weight Quantization
- Function: `quantize_tensor_int8` (lines 201–215).
- Scale definition: $s_w = \frac{\max(|W|)}{127.0}$.
- Zero handling: if $\max(|W|) == 0.0$, returns scale `1.0` and zero array. (Clean and safe).
- Clamping range: `[-128, 127]`.
- Verified scales from checkpoint:
  - Conv1 weight: `0.00331648` (max weight: `0.4212`)
  - Conv2 weight: `0.00541343` (max weight: `0.6875`)
  - Conv3 weight: `0.00465679` (max weight: `0.5914`)
  - FC1 weight: `0.00371542` (max weight: `0.4719`)
  - FC2 weight: `0.00510968` (max weight: `0.6489`)

### Bias Handling Issue (Discrepancy between Documentation and Code)
- In the script docstring (line 12), the text states:
  > *"INT16 biases, and INT32 accumulators"*
- However, in `FixedPointInference.__init__` (lines 310–320), biases are stored as raw float32 arrays:
  ```python
  self.conv_biases = [sd["features.0.bias"].cpu().numpy(), ...]
  ```
- And in `_conv2d_int` (line 476–477) and `_fc_int` (line 548):
  ```python
  output = acc.astype(np.float64) * combined_scale + bias_fp32[...]
  ```
  Biases are added in **floating-point** (`float64`/`float32`).
- To make biases integer-synthesizable in HLS, they must be pre-scaled into the integer accumulator domain:
  $$s_{\text{bias}} = s_{\text{act\_in}} \times s_w$$
  $$\text{bias}_{\text{int}} = \text{round}\left( \frac{\text{bias}_{\text{fp32}}}{s_{\text{bias}}} \right)$$
- We audited whether pre-scaled integer biases fit into INT16 across all layers:

| Layer | Accumulator Scale ($s_{\text{in}} \times s_w$) | Max Absolute FP32 Bias | Max Pre-scaled Integer Bias | Fits in INT16 ($\le 32,767$)? | Fits in INT32? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conv1** | $2.6114 \times 10^{-5}$ | 0.499828 | **19,140** | **YES** | YES |
| **Conv2** | $1.4680 \times 10^{-4}$ | 0.192137 | **1,309** | **YES** | YES |
| **Conv3** | $6.7621 \times 10^{-4}$ | 0.228764 | **338** | **YES** | YES |
| **FC1** | $1.1899 \times 10^{-3}$ | 0.148913 | **125** | **YES** (fits INT8) | YES |
| **FC2** | $1.0914 \times 10^{-3}$ | 0.207209 | **190** | **YES** | YES |

**Conclusion on Biases:** Signed INT16 is mathematically sufficient to represent the biases across all layers. However, the simulation did not actually quantize the biases to INT16; it executed floating-point bias addition.

---

## 4. Accumulator Scales, Bit-Widths, and Worst-Case Overflow Bounds

Let $N$ be the number of multiply-accumulate operations summed into a single accumulator:
$$\text{Max Accumulator Magnitude} = N \times 127 \times 127 + |\text{bias}_{\text{int}}|$$

### Exact Worst-Case Bounds for All Layers

| Layer | Input Vector Length ($N$) | Max MAC Product ($N \times 127^2$) | Max Integer Bias | Worst-Case Theoretical Bound | Minimum Signed Bits Required | INT32 Margin ($2.14 \times 10^9 / \text{Bound}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Conv1** | $3 \times 3 \times 3 = \mathbf{27}$ | 435,483 | 19,140 | **454,623** | **20 bits** | $\approx 4,724\times$ |
| **Conv2** | $16 \times 3 \times 3 = \mathbf{144}$ | 2,322,576 | 1,309 | **2,323,885** | **23 bits** | $\approx 924\times$ |
| **Conv3** | $32 \times 3 \times 3 = \mathbf{288}$ | 4,645,152 | 338 | **4,645,490** | **24 bits** | $\approx 462\times$ |
| **FC1** | $64 \times 2 \times 2 = \mathbf{256}$ | 4,129,024 | 125 | **4,129,149** | **23 bits** | $\approx 520\times$ |
| **FC2** | $64$ | 1,032,256 | 190 | **1,032,446** | **21 bits** | $\approx 2,080\times$ |

> [!NOTE]
> **Correction to Previous Report Formula:** In the earlier report, Conv3 worst-case bound was calculated using $64 \times 9 = 576$ inputs. That was an error (it used $C_{\text{out}} = 64$ instead of $C_{\text{in}} = 32$). The actual input depth for Conv3 is 32 channels, so $N = 32 \times 9 = 288$.

**Conclusion on Accumulator Width:**
- A standard 32-bit signed accumulator (`int32_t`) has a minimum safety headroom of **$462\times$** across the entire network.
- In fact, a 24-bit accumulator (`ap_int<24>` in Vitis HLS) would never overflow under any mathematical worst-case input.
- Zero INT32 overflows in the simulation is mathematically guaranteed.

---

## 5. Verification of Activations, Pooling, and Re-quantization

### 1. ReLU
- In code (lines 478, 550): `np.maximum(output, 0.0)`.
- Because ReLU is monotonic and scale factors are strictly positive, $\text{ReLU}(x \times S) = \text{ReLU}(x) \times S$.
- In HLS, integer ReLU can be implemented directly on the 32-bit accumulator prior to scaling: `if (acc < 0) acc = 0;`. This avoids negative numbers entering the requantization multiplier.

### 2. MaxPool2d
- In code (lines 488–496): Vectorized integer max over $2 \times 2$ window.
- Because MaxPool2d is a non-linear ordering operation on positive INT8 values, the integer maximum is bit-exact with floating-point maximum.
- As confirmed by the calibrated activation scales:
  - Conv1 scale = `0.02711690`, Pool1 scale = `0.02711690`
  - Conv2 scale = `0.14521016`, Pool2 scale = `0.14521016`
  - Conv3 scale = `0.50742737`, Pool3 scale = `0.50742737`
  MaxPool does not alter the activation dynamic range or scale.

### 3. AdaptiveAvgPool2d(2, 2)
- In code (lines 498–524):
  Feature map input size after Pool3 is $64 \times 8 \times 8$.
  Output target is $64 \times 2 \times 2$.
  Each output element is the average of a non-overlapping $4 \times 4$ window ($16$ elements).
- In the simulation, lines 520–522 compute:
  $$\text{avg} = \text{round}(\text{float32\_region}.\text{mean}())$$
  $$\text{val}_{\text{fp}} = \text{avg} \times s_{\text{in}}$$
  $$q = \text{clip}\left(\text{round}\left(\frac{\text{val}_{\text{fp}}}{s_{\text{out}}}\right), -128, 127\right)$$
- In HLS hardware:
  - The sum of 16 INT8 values has a worst-case range of $16 \times 127 = 2,032$ (fits in 12 bits).
  - Division by 16 is an exact 4-bit arithmetic shift right: $\text{sum} \gg 4$.
  - In the Python simulation, $s_{\text{in}} = 0.50742737$ and $s_{\text{out}} = 0.32026483$. The simulation rescaled by $\frac{0.50742737}{0.32026483} \approx 1.5844$.
  - For HLS, this rescaling multiplier must be incorporated into the fixed-point shift or eliminated by matching input/output scales.

---

## 6. Logit Generation, Argmax, and 9-Class Transistor Masking

- In code (lines 620–624):
  ```python
  logits[:, TRANSISTOR_INDEX] = -float("inf")
  preds = np.argmax(logits[:, :TRANSISTOR_INDEX], axis=1)
  ```
- Line 621 masks column index 9 to $-\infty$.
- Line 622 executes argmax strictly over columns `0..8` (`:TRANSISTOR_INDEX`).
- The dataset label mapping (`PERMITTED_CLASSES`, lines 88–94) maps the 9 target classes to integers `0..8`.
- The classification pipeline strictly prevents transistor (index 9) from ever being predicted, fully conforming to Phase 2 application requirements.

---

## 7. Audit of the 87 INT8 Saturations and Zero INT32 Overflows

### Overflows
- Counted in lines 472–473 and 544–545:
  ```python
  overflow_mask = (acc > 2147483647) | (acc < -2147483648)
  self.overflow_count += int(overflow_mask.sum())
  ```
- Result: **0 overflows**.
- Confirmed valid: Section 4 proved that the network's maximum possible accumulator value is $4,645,490 \ll 2,147,483,647$.

### Saturations
- Counted in lines 484 and 558:
  ```python
  self.saturation_count += int(np.sum((scaled < -128) | (scaled > 127)))
  ```
- Result: **87 saturations** across 346 test images.
- An instrumented execution was conducted to determine the exact layer breakdown of the 87 saturation events:

| Layer | Output Shape per Image | Total Elements Evaluated (346 Images) | Number of Saturations | Saturation Rate (%) | Cause |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Conv1 (Post-ReLU)** | $16 \times 64 \times 64$ | 22,675,456 | **65** | 0.000287% | Test image local edge activations exceeded training calibration max (3.66 vs 3.44) |
| **Conv2 (Post-ReLU)** | $32 \times 32 \times 32$ | 11,337,728 | **13** | 0.000115% | Test activations exceeded training max (20.05 vs 18.44) |
| **Conv3 (Post-ReLU)** | $64 \times 16 \times 16$ | 5,668,864 | **6** | 0.000106% | Test activations exceeded training max (87.15 vs 64.44) |
| **FC1 (Post-ReLU)** | $64$ | 22,144 | **3** | 0.013548% | Test activations exceeded training max (30.33 vs 27.13) |
| **FC2 (Logits)** | $10$ | 3,460 | **0** | 0.000000% | Final logits are unquantized FP32 |
| **Total** | - | **39,707,652** | **87** | **0.000219%** | Gracefully clipped to 127 |

The saturation count of 87 is genuine, extremely low ($< 0.0003\%$), and primarily confined to a few sharp local edge responses in Conv1.

---

## 8. Parameter Count and Memory Footprint Audit

### Parameter Count Breakdown
From `model_v5_spatial.py` and `manifest.json`:
1. Conv1 (`features.0`): $16 \times 3 \times 3 \times 3 = 432$ weights, $16$ biases
2. Conv2 (`features.3`): $32 \times 16 \times 3 \times 3 = 4,608$ weights, $32$ biases
3. Conv3 (`features.6`): $64 \times 32 \times 3 \times 3 = 18,432$ weights, $64$ biases
4. FC1 (`classifier.0`): $64 \times 256 = 16,384$ weights, $64$ biases
5. FC2 (`classifier.2`): $10 \times 64 = 640$ weights, $10$ biases

- **Total Weights:** $432 + 4,608 + 18,432 + 16,384 + 640 = \mathbf{40,496}$ elements.
- **Total Biases:** $16 + 32 + 64 + 64 + 10 = \mathbf{186}$ elements.
- **Total Trainable Parameters:** $40,496 + 186 = \mathbf{40,682}$ parameters.

### Storage Comparison Table

| Storage Configuration | Parameter Formula | Exact Byte Count | Size in KiB ($1024\text{ B}$) | Size in KB ($1000\text{ B}$) |
| :--- | :--- | :---: | :---: | :---: |
| **Original FP32 Model** | $40,682 \times 4\text{ bytes}$ | **162,728 bytes** | 158.91 KiB | 162.73 KB |
| **Pure INT8 (Weights + Biases in INT8)** | $40,682 \times 1\text{ byte}$ | **40,682 bytes** | 39.73 KiB | 40.68 KB |
| **INT8 Weights Only** | $40,496 \times 1\text{ byte}$ | **40,496 bytes** | 39.55 KiB | 40.50 KB |
| **INT8 Weights + INT16 Biases** | $40,496 \times 1 + 186 \times 2$ | **40,868 bytes** | 39.91 KiB | 40.87 KB |
| **INT8 Weights + INT32 Biases** | $40,496 \times 1 + 186 \times 4$ | **41,240 bytes** | 40.27 KiB | 41.24 KB |
| **Quantization Script Implied (INT8 W + FP32 B)** | $40,496 \times 1 + 186 \times 4$ | **41,240 bytes** | 40.27 KiB | 41.24 KB |

### Analysis of the "35.1 KB" Estimate in the Previous Report
The previous report claimed:
> *"The entire quantized model (weights + biases) occupies only 35.1 KB."*

- **Audit Finding:** **The 35.1 KB estimate is an error.**
  - 40,496 bytes equals $39.55\text{ KiB}$ ($40.5\text{ KB}$).
  - Even excluding the entire FC2 layer ($650$ params) and all biases, the weights of Conv1 + Conv2 + Conv3 + FC1 total $39,856\text{ bytes} = 38.92\text{ KiB}$.
  - The actual hardware memory footprint for the model on the PYNQ-Z2 will be **$40.9\text{ KB}$** (with INT16 biases) or **$41.2\text{ KB}$** (with INT32 biases).
  - This remains well within the PYNQ-Z2 Zynq-7020 on-chip BRAM budget ($4.9\text{ Mb} \approx 612\text{ KB}$), utilizing $< 7\%$ of total available BRAM.

---

## 9. Verification of Accuracy, Macro F1, Agreement, and Logit Errors

The evaluation metric calculations in `evaluate_v5_quantization.py` (lines 148–166, 686–700, 736–752) were independently audited:

1. **Test Sample Count:** 346 test images across 9 permitted classes (transistor excluded).
2. **Accuracy:**
   - FP32: $238 / 346 = \mathbf{68.7861\%} \implies 68.79\%$
   - INT8 Weight-Only: $240 / 346 = \mathbf{69.3642\%} \implies 69.36\%$
   - Fixed-Point Simulation: $238 / 346 = \mathbf{68.7861\%} \implies 68.79\%$
3. **Macro F1:**
   - Evaluated using `sklearn.metrics.f1_score(..., average="macro", zero_division=0)`.
   - FP32: $0.664634 \implies \mathbf{0.6646}$
   - INT8 Weight-Only: $0.669921 \implies \mathbf{0.6699}$
   - Fixed-Point: $0.660726 \implies \mathbf{0.6607}$
4. **Prediction Agreement:**
   - INT8 Weight-Only vs. FP32: $343 / 346 = \mathbf{99.13\%}$ agreement (only 3 predictions changed).
   - Fixed-Point vs. FP32: $332 / 346 = \mathbf{95.95\%}$ agreement (14 predictions changed).
5. **Logit Errors:**
   - Evaluated strictly across columns `0..8` (excluding masked transistor logit at index 9).
   - Weight-only: Max error $= 0.196405$, Mean error $= 0.043181$.
   - Fixed-point: Max error $= 1.3173$, Mean error $= 0.1408$.

All mathematical formulas for accuracy, F1, prediction agreement, and logit errors are implemented correctly and verified.

---

## 10. Summary of Correctness Issues, Unsupported Claims, and Differences with HLS

### Correctness Issues & Erroneous Claims in Earlier Materials
1. **Memory footprint under-reported:** Reported as "35.1 KB"; true footprint is **$40.9\text{ KB}$** (with INT16 biases) or **$41.2\text{ KB}$** (with INT32 biases).
2. **Conv3 bound formula error:** Earlier report calculated Conv3 accumulation bound with 64 input channels; Conv3 actually has 32 input channels. (Headroom is even larger: $462\times$ rather than $230\times$).
3. **Docstring inconsistency:** Docstring on line 12 claims "INT16 biases", but implementation uses floating-point addition for biases (`output += bias_fp32`).

### Differences Between Python Simulation and Synthesizable HLS
1. **Floating-point re-quantization:** Python uses floating-point multiplication and division (`output * combined_scale / s_act_out`). HLS must use integer fixed-point scaling $(A \times M_0) \gg S$.
2. **Floating-point bias addition:** Python adds FP32 biases after accumulator float scaling. HLS must add pre-scaled integer biases ($\text{bias}_{\text{int}}$) directly into the integer accumulator.
3. **Rounding behavior:** Python uses round-to-nearest-even (`np.round`). HLS typically uses round-half-up (`+ (1 << (S-1))`) before right shifting.
4. **Input transformation:** Python applies PyTorch's float-based normalization. HLS should use a 256-byte ROM lookup table to convert raw camera pixels $P \in [0, 255]$ into signed INT8.

---

## 11. Final Recommendation

### **Recommendation: Ready after specific fixes**

The V5-Spatial quantization scheme is structurally sound, mathematically verified, and retains 100% of the baseline accuracy (68.79%). However, before synthesizable C++ code is written in Vitis HLS, the following specific fixes must be completed:

1. **Export pre-quantized integer biases:** Generate and export `bias_int16.npy` or `bias_int32.npy` pre-scaled by $s_{\text{in}} \times s_w$ for all 5 layers.
2. **Generate fixed-point requantization multipliers:** Compute and export integer multiplier/shift pairs $(M_0, S)$ for each layer so HLS can perform integer-only scaling without floating-point division.
3. **Create the 256-entry input LUT:** Export a 256-element 8-bit lookup table array `input_lut[256]` for bit-exact mapping of camera pixels $P \in [0, 255]$ into signed INT8.
4. **Run a bit-exact integer-only verification script:** Verify inference in Python using strictly integer addition, multiplication, and bit-shifts (zero floating-point operations in the inner loop) to provide the ground-truth golden reference for C/RTL co-simulation.
