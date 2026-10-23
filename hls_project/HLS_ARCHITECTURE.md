# V5-Spatial HLS Architecture Specification

**Project:** Electronic Component Classifier (`ElectronicComponentClassifier`)  
**Document:** `hls_project/HLS_ARCHITECTURE.md`  
**Phase:** Phase 3 — High-Level Synthesis (HLS) C++ Implementation  
**Target Hardware:** Xilinx PYNQ-Z2 Evaluation Board  
**Target Device:** AMD/Xilinx Zynq-7000 SoC (`xc7z020clg400-1`)  
**Design Reference:** `integer_v5_reference.py` & `fpga_export/v5_spatial/hardware_ready/hardware_parameter_manifest.json`  
**Date:** October 2026  
**Status:** **SPECIFICATION FROZEN (PRE-SYNTHESIS)**

---

## 1. Architectural Overview and Target Environment

This specification defines the synthesizable C++ hardware architecture for the **V5-Spatial** convolutional neural network (`ComponentCNNSpatial`), planned for synthesis with AMD/Xilinx Vitis HLS targeting the programmable logic (PL) of the PYNQ-Z2 board.

### Target Device Hardware Constraints (`xc7z020clg400-1`)
- **Block RAM:** 140 RAMB36E1 primitives = 280 RAMB18 = 4.9 Mb = **612.5 KB**.
- **DSP Slices:** 220 DSP48E1 slices (dedicated $18 \times 25$ signed multipliers + accumulators).
- **Logic Cells:** 85,000 cells (~53,200 6-input LUTs, 106,400 Flip-Flops).
- **Target Clock:** 100 MHz ($T_{\text{clk}} = 10.0\text{ ns}$) nominal (configurable up to 142 MHz).
- **Memory Storage Strategy:** The total parameter bundle (weights, biases, LUT, requantization constants) requires only **41.54 KB** (~6.62% of on-chip BRAM). All parameters are permanently resident in on-chip Block RAM/ROM, eliminating DDR memory access stalls during real-time inference.

---

## 2. Input Tensor Layout and Pixel Quantization

```
Raw Camera Input (RGB, CHW)
   Channel 0 (R): 64 x 64 pixels [uint8: 0..255]
   Channel 1 (G): 64 x 64 pixels [uint8: 0..255]
   Channel 2 (B): 64 x 64 pixels [uint8: 0..255]
                      │
                      ▼
   Single-cycle Input LUT ROM (256 x 8-bit)
                      │
                      ▼
   Quantized Activation Tensor (CHW): 3 x 64 x 64 [int8: -127..127]
```

### Specifications:
- **Input Dimensions:** 3 channels, 64 rows, 64 columns ($3 \times 64 \times 64 = 12,288\text{ bytes}$).
- **Data Layout:** Planar Channel-First (CHW order):
  $$\text{Memory Offset} = c \cdot (64 \times 64) + h \cdot 64 + w, \quad c \in [0, 2], \, h \in [0, 63], \, w \in [0, 63]$$
  Channel ordering is strictly **R (0), G (1), B (2)**.
- **Pixel Conversion Mechanism:**
  Raw camera inputs are unsigned 8-bit integers $P \in [0, 255]$. The hardware converts each pixel using a 256-entry signed 8-bit ROM (`input_lut_int8`):
  $$x_{\text{int8}} = \text{LUT}[P], \quad \text{where } \text{LUT}[P] = \text{clip}\left(\text{round}\left(\frac{2P/255.0 - 1.0}{0.00787402}\right), -128, 127\right)$$
- **Hardware LUT Implementation:** A single 256-entry dual-port ROM (`ap_int<8> lut[256]`), requiring 0 DSPs and less than 1 RAMB18 primitive.

---

## 3. Detailed Layer-by-Layer Compute Pipeline

### 3.1 Conv Block 1 (`features.0` to `features.2`)
- **Input:** $3 \times 64 \times 64$, signed INT8 activations (`ap_int<8>`).
- **Layer Dimensions:**
  - Convolution: 16 output filters, 3 input channels, $3 \times 3$ kernel, stride 1, padding 1.
  - Weights: `[16, 3, 3, 3]`, 432 elements, signed INT8 (`ap_int<8>`), scale $s_w = 0.003316483394367488$.
  - Biases: `[16]`, signed INT32 (`ap_int<32>`), scale $s_{\text{bias}} = 2.611395015383921 \times 10^{-5}$, range $[-5965, 19140]$.
- **Arithmetic Pipeline:**
  1. **MAC:** 27 multiply-accumulate operations per output pixel ($3 \times 3 \times 3$):
     $$\text{acc}_{\text{raw}} = \sum_{c=0}^{2} \sum_{ky=0}^{2} \sum_{kx=0}^{2} x(c, y+ky-1, x+kx-1) \cdot w(m, c, ky, kx)$$
     $\text{acc}_{\text{raw}}$ is accumulated in a signed 32-bit register (`ap_int<32>`).
  2. **Bias Addition:** Add pre-scaled INT32 bias directly to MAC:
     $$\text{acc}_{\text{biased}} = \text{acc}_{\text{raw}} + b_{\text{int32}}[m]$$
  3. **ReLU:** Integer half-wave rectification:
     $$\text{acc}_{\text{relu}} = (\text{acc}_{\text{biased}} > 0) \ ? \ \text{acc}_{\text{biased}} : 0$$
  4. **Fixed-Point Requantization:**
     - Scaling ratio: $0.0009630177703541885$
     - Hardware multiplier: $M_{31} = 2,117,698,473$ (`ap_int<32>`)
     - Right shift: $S_{31} = 41$
     - Intermediate product: $\text{prod} = (\text{ap\_int<64>})\text{acc}_{\text{relu}} \cdot M_{31}$ (requires 51 bits; fits in `ap_int<64>`)
     - Rounding and Shift:
       $$q_{\text{conv1}} = (\text{prod} + (1\text{LL} \ll 40)) \gg 41$$
     - Saturation: $\text{clip}(q_{\text{conv1}}, 0, 127)$ -> `ap_int<8>`.
  5. **MaxPool2d:** Non-overlapping $2 \times 2$ window, stride 2:
     $$x_{\text{out}}(m, y', x') = \max_{dy, dx \in \{0, 1\}} q_{\text{conv1}}(m, 2y'+dy, 2x'+dx)$$
- **Output:** $16 \times 32 \times 32$ signed INT8 activations (`ap_int<8>`), output scale $0.0271169$.

---

### 3.2 Conv Block 2 (`features.3` to `features.5`)
- **Input:** $16 \times 32 \times 32$, signed INT8 activations (`ap_int<8>`).
- **Layer Dimensions:**
  - Convolution: 32 output filters, 16 input channels, $3 \times 3$ kernel, stride 1, padding 1.
  - Weights: `[32, 16, 3, 3]`, 4,608 elements, signed INT8 (`ap_int<8>`), scale $s_w = 0.005413434636874462$.
  - Biases: `[32]`, signed INT32 (`ap_int<32>`), scale $s_{\text{bias}} = 1.4679555171705646 \times 10^{-4}$, range $[-779, 1309]$.
- **Arithmetic Pipeline:**
  1. **MAC:** 144 multiplications per output pixel ($16 \times 3 \times 3$), accumulated in `ap_int<32>`.
  2. **Bias Addition:** Add $b_{\text{int32}}[m]$ to accumulator before ReLU.
  3. **ReLU:** $\text{acc}_{\text{relu}} = \max(\text{acc}_{\text{biased}}, 0)$.
  4. **Fixed-Point Requantization:**
     - Scaling ratio: $0.0010109180081108724$
     - Hardware multiplier: $M_{31} = 1,111,516,105$
     - Right shift: $S_{31} = 40$
     - Intermediate product: $\text{prod} = (\text{ap\_int<64>})\text{acc}_{\text{relu}} \cdot M_{31}$ (requires 53 bits; fits in `ap_int<64>`)
     - Rounding and Shift:
       $$q_{\text{conv2}} = (\text{prod} + (1\text{LL} \ll 39)) \gg 40$$
     - Saturation: $\text{clip}(q_{\text{conv2}}, 0, 127)$ -> `ap_int<8>`.
  5. **MaxPool2d:** Non-overlapping $2 \times 2$ window, stride 2.
- **Output:** $32 \times 16 \times 16$ signed INT8 activations (`ap_int<8>`), output scale $0.14521016$.

---

### 3.3 Conv Block 3 (`features.6` to `features.7`)
- **Input:** $32 \times 16 \times 16$, signed INT8 activations (`ap_int<8>`).
- **Layer Dimensions:**
  - Convolution: 64 output filters, 32 input channels, $3 \times 3$ kernel, stride 1, padding 1.
  - Weights: `[64, 32, 3, 3]`, 18,432 elements, signed INT8 (`ap_int<8>`), scale $s_w = 0.004656787932388426$.
  - Biases: `[64]`, signed INT32 (`ap_int<32>`), scale $s_{\text{bias}} = 6.762128795052959 \times 10^{-4}$, range $[-220, 338]$.
- **Arithmetic Pipeline:**
  1. **MAC:** 288 multiplications per output pixel ($32 \times 3 \times 3$), accumulated in `ap_int<32>`.
  2. **Bias Addition:** Add $b_{\text{int32}}[m]$ to accumulator before ReLU.
  3. **ReLU:** $\text{acc}_{\text{relu}} = \max(\text{acc}_{\text{biased}}, 0)$.
  4. **Fixed-Point Requantization:**
     - Scaling ratio: $0.0013326299697791085$
     - Hardware multiplier: $M_{31} = 1,465,242,147$
     - Right shift: $S_{31} = 40$
     - Intermediate product: $\text{prod} = (\text{ap\_int<64>})\text{acc}_{\text{relu}} \cdot M_{31}$ (requires 54 bits; fits in `ap_int<64>`)
     - Rounding and Shift:
       $$q_{\text{conv3}} = (\text{prod} + (1\text{LL} \ll 39)) \gg 40$$
     - Saturation: $\text{clip}(q_{\text{conv3}}, 0, 127)$ -> `ap_int<8>`.
  5. **MaxPool2d:** Non-overlapping $2 \times 2$ window, stride 2.
- **Output:** $64 \times 8 \times 8$ signed INT8 activations (`ap_int<8>`), output scale $0.50742737$.

---

### 3.4 Spatial Adaptive Average Pooling (`features.8`)
- **Input:** $64 \times 8 \times 8$, signed INT8 activations (`ap_int<8>`). Because it follows Conv3 + ReLU + MaxPool3, inputs are strictly non-negative: $x \in [0, 127]$.
- **Target Output:** $64 \times 2 \times 2$ signed INT8 activations (`ap_int<8>`).
- **Spatial Subdivisions:**
  Each channel ($8 \times 8$) is partitioned into 4 non-overlapping quadrants of $4 \times 4$ pixels (16 elements):
  - Top-Left: $y \in [0..3], \, x \in [0..3]$
  - Top-Right: $y \in [0..3], \, x \in [4..7]$
  - Bottom-Left: $y \in [4..7], \, x \in [0..3]$
  - Bottom-Right: $y \in [4..7], \, x \in [4..7]$

#### Exact Hardware Integer Implementation:
1. **16-Element Accumulation:**
   Sum 16 unsigned values:
   $$\text{sum} = \sum_{i=0}^{3} \sum_{j=0}^{3} x_{\text{in}}(c, 4Y+i, 4X+j), \quad Y, X \in \{0, 1\}$$
   Because $x \in [0, 127]$, the maximum possible sum is $16 \times 127 = 2032$. This requires a 12-bit unsigned accumulator (`ap_uint<12> sum`).
2. **Division by 16 with Round-to-Nearest-Even (Banker's Rounding):**
   Division by 16 is equivalent to right shifting by 4 bits. Remainder is the lowest 4 bits:
   ```cpp
   ap_uint<12> sum = ...; // range [0, 2032]
   ap_uint<8>  div = sum >> 4;       // bits 11..4
   ap_uint<4>  rem = sum & 0xF;      // bits 3..0
   
   // Exact Banker's Rounding (round-to-nearest-even matching PyTorch mean()):
   ap_uint<8> avg;
   if (rem > 8) {
       avg = div + 1;
   } else if (rem < 8) {
       avg = div;
   } else {
       // Exactly on half boundary (rem == 8): round to nearest even
       avg = (div[0] == 1) ? (div + 1) : div;
   }
   ```
   *Hardware Cost:* Only a 4-bit equality test (`rem == 8`) and a 1-bit incrementer, costing less than 6 LUTs in Vivado.
3. **Requantization to FC1 Scale ($0.50742737 \to 0.32026483$):**
   - Scaling ratio: $\text{ratio} = 1.584399285$
   - Hardware multiplier: $M_{\text{pool}} = 103,835$ (`ap_uint<18>`)
   - Right shift: $S_{\text{pool}} = 16$
   - Computation:
     ```cpp
     ap_uint<32> pool_prod = avg * (ap_uint<18>)103835;
     ap_int<8>   q_pool    = (pool_prod + 32768) >> 16;
     ```
- **Output:** $64 \times 2 \times 2$ signed INT8 activations (`ap_int<8>`), output scale $0.32026483$.

---

### 3.5 Flattening Order
The $64 \times 2 \times 2$ feature map is flattened into a 1D vector of 256 elements.
To maintain 100% mathematical equivalence with PyTorch and Python reference `x.view(x.size(0), -1)`:
$$\text{flat\_index} = c \cdot 4 + Y \cdot 2 + X, \quad c \in [0, 63], \, Y \in [0, 1], \, X \in [0, 1]$$
- Index 0: Channel 0, $Y=0, X=0$
- Index 1: Channel 0, $Y=0, X=1$
- Index 2: Channel 0, $Y=1, X=0$
- Index 3: Channel 0, $Y=1, X=1$
- Index 4: Channel 1, $Y=0, X=0$
- ...
- Index 255: Channel 63, $Y=1, X=1$

---

### 3.6 FC1 Layer (`classifier.0`)
- **Input:** 256 signed INT8 activations (`ap_int<8>`).
- **Layer Dimensions:**
  - Linear fully connected: 256 inputs, 64 output neurons.
  - Weights: `[64, 256]`, 16,384 elements, signed INT8 (`ap_int<8>`), scale $s_w = 0.0037154247910957637$.
  - Biases: `[64]`, signed INT32 (`ap_int<32>`), scale $s_{\text{bias}} = 1.189919794017772 \times 10^{-3}$, range $[-86, 125]$.
- **Arithmetic Pipeline:**
  1. **MAC:** 256 multiply-accumulate operations per neuron, accumulated in `ap_int<32>`.
  2. **Bias Addition:** Add pre-scaled INT32 bias $b_{\text{int32}}[k]$ to accumulator before ReLU.
  3. **ReLU:** $\text{acc}_{\text{relu}} = \max(\text{acc}_{\text{biased}}, 0)$.
  4. **Fixed-Point Requantization:**
     - Scaling ratio: $0.005570755744482241$
     - Hardware multiplier: $M_{31} = 1,531,277,679$
     - Right shift: $S_{31} = 38$
     - Intermediate product: $\text{prod} = (\text{ap\_int<64>})\text{acc}_{\text{relu}} \cdot M_{31}$ (requires 54 bits; fits in `ap_int<64>`)
     - Rounding and Shift:
       $$q_{\text{fc1}} = (\text{prod} + (1\text{LL} \ll 37)) \gg 38$$
     - Saturation: $\text{clip}(q_{\text{fc1}}, 0, 127)$ -> `ap_int<8>`.
- **Output:** 64 signed INT8 activations (`ap_int<8>`), output scale $0.21360116$.

---

### 3.7 FC2 Output Layer (`classifier.2`) & Classification
- **Input:** 64 signed INT8 activations (`ap_int<8>`).
- **Layer Dimensions:**
  - Linear fully connected: 64 inputs, 10 output logits.
  - Weights: `[10, 64]`, 640 elements, signed INT8 (`ap_int<8>`), scale $s_w = 0.005109675287261722$.
  - Biases: `[10]`, signed INT32 (`ap_int<32>`), scale $s_{\text{bias}} = 1.0914325685824372 \times 10^{-3}$, range $[-190, 76]$.
- **Arithmetic Pipeline:**
  1. **MAC:** 64 multiply-accumulate operations per logit, accumulated in `ap_int<32>`.
  2. **Bias Addition:** Add $b_{\text{int32}}[k]$ to compute the final integer logit:
     $$\text{logit}_{\text{int}}[k] = \sum_{j=0}^{63} x_{\text{fc1}}[j] \cdot w_{\text{fc2}}[k, j] + b_{\text{int32}}[k], \quad k \in [0, 9]$$
  3. **Requantization:** Identity ($M=1, S=0$). Because all logits share the identical positive scale factor $s_{\text{logits}} = s_{\text{fc1\_out}} \cdot s_{\text{fc2\_w}} = 0.00109143$, argmax classification over integer logits is mathematically identical to argmax over floating-point logits. No DSP or multiplier cycles are wasted.

#### Nine-Class Decision Policy:
- The network computes 10 integer logits: $\text{logit}_{\text{int}}[0 \dots 9]$.
- **Transistor Masking:** Index 9 corresponds to `transistor`. It is strictly masked from the decision logic.
- **Argmax Logic:**
  ```cpp
  ap_uint<4> best_class = 0;
  ap_int<32> max_val    = logits[0];
  
  for (int c = 1; c < 9; c++) { // Loop ONLY over indices 0..8 (classes 0..8)
      if (logits[c] > max_val) {
          max_val    = logits[c];
          best_class = c;
      }
  }
  ```
- Output: Winning class index in $[0, 8]$ and maximum integer logit score.

---

## 4. Parameter Storage and Hardware Arithmetic Specifications

### 4.1 Memory Allocation & BRAM Layout
| Parameter Name | Tensor Shape | Precision | Words | Storage (Bytes) | Storage Primitive |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `input_lut` | $[256]$ | `ap_int<8>` | 256 | 256 B | ROM (LUT / 1 BRAM18) |
| `conv1_weight` | $[16, 3, 3, 3]$ | `ap_int<8>` | 432 | 432 B | BRAM ROM |
| `conv1_bias` | $[16]$ | `ap_int<32>` | 16 | 64 B | Distributed ROM (LUTs) |
| `conv2_weight` | $[32, 16, 3, 3]$ | `ap_int<8>` | 4,608 | 4,608 B | BRAM ROM |
| `conv2_bias` | $[32]$ | `ap_int<32>` | 32 | 128 B | Distributed ROM (LUTs) |
| `conv3_weight` | $[64, 32, 3, 3]$ | `ap_int<8>` | 18,432 | 18,432 B | BRAM ROM |
| `conv3_bias` | $[64]$ | `ap_int<32>` | 64 | 256 B | Distributed ROM (LUTs) |
| `fc1_weight` | $[64, 256]$ | `ap_int<8>` | 16,384 | 16,384 B | BRAM ROM |
| `fc1_bias` | $[64]$ | `ap_int<32>` | 64 | 256 B | Distributed ROM (LUTs) |
| `fc2_weight` | $[10, 64]$ | `ap_int<8>` | 640 | 640 B | BRAM ROM |
| `fc2_bias` | $[10]$ | `ap_int<32>` | 10 | 40 B | Distributed ROM (LUTs) |
| `requant_constants` | - | `ap_uint<32>`, `ap_uint<8>` | - | 48 B | Embedded constants in code |
| **Total** | - | - | **40,946 words** | **41,544 B (~40.57 KB)** | **~9.2 RAMB36 equivalent** |

### 4.2 Arithmetic Bit-Widths and Overflow Protection
1. **Accumulators:**
   All MAC operations and bias additions use signed 32-bit registers (`ap_int<32>`).
   - Maximum theoretical sum: Conv3 worst-case $= 4,718,930$ (requires 24 signed bits).
   - Headroom: Signed 32-bit integer provides $2,147,483,647$, leaving **at least 8 bits of safety margin**. Overflows are strictly impossible.
2. **Requantization Multipliers:**
   - Pre-shift products use signed 64-bit registers (`ap_int<64>`).
   - Maximum theoretical product: Conv3 worst-case $= 4.72 \times 10^6 \times 1.47 \times 10^9 = 6.91 \times 10^{15}$ (requires 54 bits).
   - Headroom: Signed 64-bit integer holds $9.22 \times 10^{18}$, leaving **10 bits of safety margin**.
3. **Rounding Convention:**
   - Round-half-up addition of $(1\text{LL} \ll (S - 1))$ is applied prior to right-shift by $S$ bits.
   - Because inputs are non-negative post-ReLU, arithmetic right-shift behavior is identical to logical right-shift.
4. **Saturation:**
   - Results are clamped to $[0, 127]$ for post-ReLU layers and $[-128, 127]$ for general activations using:
     ```cpp
     inline ap_int<8> saturate_int8(ap_int<32> val) {
         if (val > 127) return 127;
         if (val < -128) return -128;
         return (ap_int<8>)val;
     }
     ```

---

## 5. Input and Output Interface Proposal for PYNQ-Z2

> [!NOTE]
> **PROPOSAL STATUS: TENTATIVE / TO BE CONFIRMED**  
> The interface pragmas described below represent the standard, high-performance integration pattern for PYNQ-Z2 (Zynq-7000). This proposal must be confirmed against the installed Vitis HLS tool version and PYNQ DMA driver conventions during Step 3.3.

### 5.1 Proposed Interface Architecture
```
                   +---------------------------------------------+
                   |           V5-Spatial HLS Core IP            |
                   |                                             |
  PS (ARM Core)    | +-----------------------------------------+ |
  AXI4-Lite Control| |        s_axi_control (Slave)            | |
  ─────────────────┼─► Reg 0x00: AP_START / AP_DONE / AP_IDLE  | |
                   | | Reg 0x10: Predicted Class (0..8)        | |
                   | | Reg 0x18..0x3C: 10 Raw Integer Logits   | |
                   | +-----------------------------------------+ |
                   |                                             |
  PS (AXI DMA)     | +-----------------------------------------+ |
  AXI4-Stream Data | |        axis_in (Slave AXI-Stream)       | |
  ─────────────────┼─► 3 x 64 x 64 (12,288 bytes) uint8 pixels | |
                   | +-----------------------------------------+ |
                   +---------------------------------------------+
```

### 5.2 Proposed Vitis HLS Pragmas
```cpp
void v5_spatial_top(
    hls::stream<ap_uint<8> > &axis_in,   // Streaming input from PYNQ DMA
    ap_uint<4>               &pred_class,// Output predicted class (0..8)
    ap_int<32>               logits[10]  // Output raw integer logits
) {
    #pragma HLS INTERFACE axis port=axis_in
    #pragma HLS INTERFACE s_axilite port=pred_class bundle=control
    #pragma HLS INTERFACE s_axilite port=logits bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    
    // Core pipeline execution ...
}
```

### 5.3 Interface Rationale
1. **AXI4-Stream for Image Pixels:** PYNQ DMA can stream all 12,288 bytes in a single hardware burst transfer directly into the core, achieving near-wire-speed input loading.
2. **AXI4-Lite for Control & Results:** Simple memory-mapped registers allow Python on the PYNQ ARM core (`overlay.v5_spatial.read()`) to initiate inference, poll for completion, and read the predicted class index and integer logits directly without setting up an output DMA buffer.

---

## 6. Layer-by-Layer Verification Test Strategy

To guarantee **100% bit-exact equivalence** before full system synthesis, verification must proceed bottom-up:

```
[Layer 1: Input LUT]
   │ Compare C-sim vs Python LUT array for all 256 pixel codes [0..255].
   ▼
[Layer 2: Conv Block 1]
   │ Feed synthetic test patterns & real image; compare Conv1+ReLU+MaxPool output (16x32x32).
   ▼
[Layer 3: Conv Block 2]
   │ Compare Conv2+ReLU+MaxPool output (32x16x16) against Python reference intermediate tensor.
   ▼
[Layer 4: Conv Block 3]
   │ Compare Conv3+ReLU+MaxPool output (64x8x8) against Python reference intermediate tensor.
   ▼
[Layer 5: Spatial Adaptive Pool]
   │ Test all 2,033 non-negative sums [0..2032]; verify round-to-even behavior against Python.
   ▼
[Layer 6: FC1 Layer]
   │ Verify flattened vector (256) and 64 post-ReLU activations.
   ▼
[Layer 7: FC2 Layer & Argmax]
   │ Verify 10 integer logits, transistor exclusion at index 9, and winning class index.
   ▼
[Full-Network 346-Image Regression]
   │ Run full test suite; verify exact 68.79% accuracy and 100% agreement with integer reference.
```

### Test Harness Implementation Plan:
1. **Test Vector Export:** Create a utility script (`export_hls_test_vectors.py`) to dump sample images, intermediate layer activations, and expected logits into `hls_project/data/`.
2. **C-Simulation Testbench:** Implement `hls_project/testbench/tb_classifier.cpp` to load input data, invoke `v5_spatial_top()`, and assert that every intermediate layer matches the Python reference with **0 LSB error**.

---

## 7. Architecture Summary & Unresolved Questions

### Architecture Summary Table
| Stage | Input Shape | Output Shape | Parameters | Primary Arithmetic | Intermediate Width |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input LUT** | $3 \times 64 \times 64$ (`uint8`) | $3 \times 64 \times 64$ (`int8`) | 256 B | Direct ROM Table Lookup | 8 bits |
| **Conv1 Block** | $3 \times 64 \times 64$ | $16 \times 32 \times 32$ | 432 B w / 64 B b | $3 \times 3$ Conv + Bias + ReLU + Requant + 2x2 MaxPool | 32b MAC / 64b Mult |
| **Conv2 Block** | $16 \times 32 \times 32$ | $32 \times 16 \times 16$ | 4,608 B w / 128 B b | $3 \times 3$ Conv + Bias + ReLU + Requant + 2x2 MaxPool | 32b MAC / 64b Mult |
| **Conv3 Block** | $32 \times 16 \times 16$ | $64 \times 8 \times 8$ | 18,432 B w / 256 B b | $3 \times 3$ Conv + Bias + ReLU + Requant + 2x2 MaxPool | 32b MAC / 64b Mult |
| **Spatial Pool** | $64 \times 8 \times 8$ | $64 \times 2 \times 2$ | 2 B ($M, S$) | $4 \times 4$ Region Sum + Round-Even Div16 + Requant | 12b Sum / 32b Mult |
| **Flatten** | $64 \times 2 \times 2$ | $256$ | - | PyTorch CHW Flatten Order | 8 bits |
| **FC1 Block** | $256$ | $64$ | 16,384 B w / 256 B b | Dense MAC + Bias + ReLU + Requant | 32b MAC / 64b Mult |
| **FC2 Layer** | $64$ | $10$ | 640 B w / 40 B b | Dense MAC + Bias -> Raw INT32 Logits | 32b MAC |
| **Classifier** | $10$ logits | 1 class index | - | Slice indices $0 \dots 8$ -> Argmax (mask transistor) | 32b Compare |

### Unresolved Questions & External Dependencies
1. **Installed Vitis HLS Tool Version:**
   The exact version of Vitis HLS on the friend's laptop (e.g., 2022.1, 2022.2, 2023.1, or 2024.1) will dictate whether `ap_axiu<8, 0, 0, 0>` or standard `hls::stream<ap_uint<8> >` is preferred for AXI-Stream interfaces, and whether `s_axilite` bundle syntax requires specific compiler flags.
2. **Line Buffer Depth vs. Resource Tradeoff:**
   For 2D convolutions, we can either:
   - *Option A (Streaming Line Buffers):* Maintain 2 line buffers per layer to stream pixels line-by-line (minimal BRAM, lower latency).
   - *Option B (Full Ping-Pong Buffers):* Buffer entire activation feature maps in on-chip BRAM (simple control logic, consumes ~50 KB BRAM, perfectly viable given 612.5 KB total BRAM).
3. **Target Clock Period:**
   Whether 100 MHz (10 ns) or 125 MHz (8 ns) is the primary target for timing closure on the `xc7z020clg400-1` speed grade -1 part.

---

## 8. Confirmed Integrated C++ Top-Level Software Interface (Step 3.9)

### 8.1 Confirmed Top-Level Function Prototype
Implemented in [`hls_project/src/component_classifier.h`](component_classifier.h) and [`component_classifier.cpp`](component_classifier.cpp):
```cpp
int component_classifier(
    const uint8_t image_rgb[3][64][64],  // Planar CHW camera image (R=0, G=1, B=2)
    int32_t logits_out[10]               // All 10 raw signed INT32 class logits
);
```

### 8.2 End-to-End Pipelined Data Flow
```
Raw Planar Camera Frame (uint8 [3][64][64])
       │
       ▼ [Stage 1: quantize_input_image_lut]
Signed Input Activations (int8 [3][64][64], scale = 0.00787402)
       │
       ▼ [Stage 2: conv1] Conv2d(3->16, 3x3) + Bias + ReLU + Requant(M=2117698473, S=41) + MaxPool2d(2)
Conv1 Activations (int8 [16][32][32], scale = 0.02711690)
       │
       ▼ [Stage 3: conv2] Conv2d(16->32, 3x3) + Bias + ReLU + Requant(M=1111516105, S=40) + MaxPool2d(2)
Conv2 Activations (int8 [32][16][16], scale = 0.14521016)
       │
       ▼ [Stage 4: conv3] Conv2d(32->64, 3x3) + Bias + ReLU + Requant(M=1465242147, S=40) + MaxPool2d(2)
Conv3 Activations (int8 [64][8][8], scale = 0.50742737)
       │
       ▼ [Stage 5: spatial_pool_with_requant] 4x4 AvgPool (Banker's round-even) + Requant(M=103835, S=16)
Pooled Activations (int8 [64][2][2], scale = 0.32026483)
       │
       ▼ [Stage 6: Flatten] c * 4 + py * 2 + px
Flattened Feature Vector (int8 [256], scale = 0.32026483)
       │
       ▼ [Stage 7: fc1] Dense(256->64) + Bias + ReLU + Requant(M=1531277679, S=38)
FC1 Hidden Activations (int8 [64], scale = 0.21360116)
       │
       ▼ [Stage 8: fc2] Dense(64->10) + Bias -> Identity Requantization (M=1, S=0)
Raw Output Logits (signed int32 [10], scale = 0.00109143)
       │
       ▼ [Stage 9: predict_9class] argmax over classes 0..8 (mask transistor index 9)
Winning Class Index (int in [0..8])
```

### 8.3 Layer-to-Layer Memory Footprint
- Total intermediate buffer footprint across all stages is **41,504 bytes (~40.5 KB)**.
- All intermediate buffers are strictly overwritten on every inference pass, guaranteeing zero residual state or buffer reuse hazards.
- Host-side C++ verification achieved **100% bit-exact parity (0 mismatches)** against `integer_v5_reference.py` across real and synthetic test vectors.

