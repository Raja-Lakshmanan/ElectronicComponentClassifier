# V5-Spatial HLS Project for PYNQ-Z2 FPGA

**Project:** Electronic Component Classifier (`ElectronicComponentClassifier`)  
**Phase:** Phase 3 — High-Level Synthesis (HLS) C++ Implementation  
**Target Hardware:** Xilinx PYNQ-Z2 Board  
**Target FPGA:** AMD/Xilinx Zynq-7000 SoC (`xc7z020clg400-1`)  

---

## 1. Overview and Target Hardware Specifications

This directory contains the synthesizable C++ implementation and verification environment for the **V5-Spatial** convolutional neural network, targeting the programmable logic (PL) of the **PYNQ-Z2** evaluation board.

### Target FPGA Resource Budget (`xc7z020clg400-1`)
| Resource Type | Available on Chip | V5-Spatial Parameter Requirement | Available Headroom |
| :--- | :---: | :---: | :---: |
| **Block RAM (BRAM_18K / BRAM_36K)** | 280 (18K) / 140 (36K)<br>*(4.9 Mb = 612.5 KB)* | ~9.2 36K BRAMs<br>*(41.54 KB)* | **> 93% BRAM headroom** (for line buffers & activation ping-pong buffers) |
| **DSP48E1 Slices** | 220 slices | Parameterized MAC tiling | Available for parallel convolution multipliers |
| **Lookup Tables (LUTs)** | 53,200 | Control logic, muxes, adders | High headroom |
| **Flip-Flops (FFs)** | 106,400 | Pipeline registers | High headroom |

Because the entire quantized parameter bundle (weights, biases, LUT, requantization constants) occupies only **41.54 KB**, all parameters are stored entirely in on-chip Block RAM (BRAM). This achieves maximum memory bandwidth and eliminates external DDR memory access bottlenecks during real-time image inference.

---

## 2. V5-Spatial Architecture Specification

The V5-Spatial architecture (`ComponentCNNSpatial` from `model_v5_spatial.py`) is an integer-only CNN specifically tailored for FPGA spatial inference:

```
[Input Camera Image: 3 x 64 x 64 uint8]
                 │
                 ▼
[Input LUT: 256 x 8-bit ROM -> Signed INT8 [-127, 127]]
                 │
                 ▼
[Conv Block 1: 16 filters, 3x3, pad 1, stride 1]
  ├── INT8 * INT8 MAC -> INT32 Accumulator
  ├── INT32 Bias Addition (pre-scaled)
  ├── Integer ReLU (max(acc, 0))
  ├── Fixed-point Requantization (M=2117698473, S=41) -> INT8
  └── MaxPool2d: 2x2, stride 2 -> [16 x 32 x 32]
                 │
                 ▼
[Conv Block 2: 32 filters, 3x3, pad 1, stride 1]
  ├── INT8 * INT8 MAC -> INT32 Accumulator
  ├── INT32 Bias Addition (pre-scaled)
  ├── Integer ReLU (max(acc, 0))
  ├── Fixed-point Requantization (M=1111516105, S=40) -> INT8
  └── MaxPool2d: 2x2, stride 2 -> [32 x 16 x 16]
                 │
                 ▼
[Conv Block 3: 64 filters, 3x3, pad 1, stride 1]
  ├── INT8 * INT8 MAC -> INT32 Accumulator
  ├── INT32 Bias Addition (pre-scaled)
  ├── Integer ReLU (max(acc, 0))
  ├── Fixed-point Requantization (M=1465242147, S=40) -> INT8
  └── MaxPool2d: 2x2, stride 2 -> [64 x 8 x 8]
                 │
                 ▼
[Spatial Adaptive Average Pool: 8x8 -> 2x2]
  ├── 16-element integer sum per 4x4 region (sum in [0, 2032])
  ├── Division by 16 with round-to-nearest
  └── Requantization (M_pool=103835, S_pool=16) -> [64 x 2 x 2] INT8
                 │
                 ▼
[Flatten: 64 x 2 x 2 = 256 INT8 activations]
                 │
                 ▼
[FC1 Layer: 256 -> 64]
  ├── INT8 * INT8 MAC -> INT32 Accumulator
  ├── INT32 Bias Addition (pre-scaled)
  ├── Integer ReLU (max(acc, 0))
  └── Fixed-point Requantization (M=1531277679, S=38) -> [64] INT8
                 │
                 ▼
[FC2 Output Layer: 64 -> 10]
  ├── INT8 * INT8 MAC -> INT32 Accumulator
  └── INT32 Bias Addition -> [10] Raw INT32 Logits
                 │
                 ▼
[Application Classification Logic]
  ├── Transistor Masking: Exclude class index 9 (slice classes 0..8)
  └── Integer Argmax: Determine winning class among 9 permitted classes
```

---

## 3. Nine-Class Output Policy

The neural network outputs 10 raw integer logits corresponding to the original training classes:
- Index 0: `capacitor`
- Index 1: `diode`
- Index 2: `fuse`
- Index 3: `inductor`
- Index 4: `integrated_circuit`
- Index 5: `led`
- Index 6: `potentiometer`
- Index 7: `push_button`
- Index 8: `resistor`
- Index 9: `transistor` (Excluded / Masked)

### Classification Rules:
1. **Transistor Masking:** During inference, class index 9 (`transistor`) is masked and never selected.
2. **Scale Invariance:** Because integer logits share a positive common scale factor, argmax over integer logits is mathematically identical to argmax over floating-point logits. No divider or floating-point softmax is required on the FPGA.
3. **Argmax Evaluation:** The winning prediction is determined solely over indices $0 \dots 8$.

---

## 4. Subdirectory Structure and Purpose

The `hls_project` directory is structured as follows:

```
hls_project/
├── README.md           # This architecture and design documentation
├── src/                # Synthesizable C++ source files and headers
├── testbench/          # C++ testbench files for C-sim and co-simulation
├── data/               # Golden test vectors, parameters, and test images
└── reports/            # Synthesis, latency, and resource utilization reports
```

### Purpose of Each Subdirectory:

- **`src/` (Synthesizable HLS Source Code):**
  - Contains top-level kernel functions, pipeline directives (`#pragma HLS PIPELINE`, `#pragma HLS UNROLL`), layer execution kernels (Conv2D, MaxPool, SpatialPool, FC), parameter ROM declarations, and interface definitions (AXI4-Lite control, AXI4-Stream data transfer).
  - Uses Xilinx arbitrary precision types (`ap_int<8>`, `ap_int<32>`, `ap_int<64>`, `ap_uint<8>`).

- **`testbench/` (C++ Simulation Testbench):**
  - Contains the C++ testbench (`tb_classifier.cpp`) used for Vitis HLS C-simulation (`csim_design`) and C/RTL co-simulation (`cosim_design`).
  - Verifies bit-exact output equivalence against the Python golden reference [integer_v5_reference.py](file:///c:/Users/rajal/OneDrive/Desktop/ElectronicComponentClassifier/integer_v5_reference.py).

- **`data/` (Test Vectors & Parameter Headers):**
  - Stores formatted C++ header files (`weights.h`, `biases.h`, `lut.h`, `requant.h`) generated from `fpga_export/v5_spatial/integer_params/`.
  - Stores sample input images and corresponding golden intermediate activations/logits for automated testbench validation.

- **`reports/` (Synthesis & Co-simulation Reports):**
  - Destination for Vitis HLS synthesis reports (`csynth.rpt`), resource utilization summaries (LUT, FF, DSP, BRAM), timing closure reports, and co-simulation verification logs.

---

## 5. Development Strategy

1. **Source Generation:** Synthesizable C++ source code, testbenches, and header files are prepared and validated in this repository.
2. **Synthesis & Implementation:** Project files and synthesizable sources will be compiled, simulated, and synthesized in Vitis HLS / Vivado on the synthesis workstation.
3. **Golden Reference:** All HLS kernels are required to achieve 100% bit-exact prediction agreement with the Python integer reference [integer_v5_reference.py](file:///c:/Users/rajal/OneDrive/Desktop/ElectronicComponentClassifier/integer_v5_reference.py).
