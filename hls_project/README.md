# V5-Spatial HLS Project — PYNQ-Z2

**Target:** PYNQ-Z2 / Zynq-7020 (xc7z020clg400-1)  
**Implementation:** Integer C++ inference pipeline intended for AMD/Xilinx Vitis HLS and Vivado.

> **Status:** The repository contains host-side C++ testbenches and project reports describing reported MSVC regression passes. Vitis HLS C simulation, synthesis, RTL co-simulation, resource/timing measurement, Vivado implementation, bitstream generation, and physical board validation must still be confirmed on the target toolchain. No resource estimate in this document is a synthesis result.

## Pipeline

Input is a planar RGB byte tensor ([3][64][64], 12,288 bytes). The design uses an input quantization lookup table, three convolution + ReLU + requantization + max-pooling stages, spatial pooling with pooling requantization, a 256-to-64 fully-connected layer, and a 64-to-10 logits layer.

The ten logits follow this order:

| Index | Class | Deployment selection |
|---:|---|---|
| 0 | capacitor | Included |
| 1 | diode | Included |
| 2 | fuse | Included |
| 3 | inductor | Included |
| 4 | integrated_circuit | Included |
| 5 | led | Included |
| 6 | potentiometer | Included |
| 7 | push_button | Included |
| 8 | resistor | Included |
| 9 | transistor | Diagnostic logit only; excluded from argmax |

The output logits are raw INT32 values. The deployment prediction uses argmax over indices 0–8; class 9 is retained for diagnostics.

## Repository files

- src/ — top-level interface, layer kernels, and inference wrapper.
- testbench/ — host/HLS testbenches and golden-vector generation scripts.
- run_hls.tcl — Tcl build script. Check its working directory and tool-version compatibility before running.
- HLS_ARCHITECTURE.md — dataflow and arithmetic notes.
- PRE_SYNTHESIS_AUDIT.md — static compatibility audit.
- STEP_3_11_REPORT.md, STEP_3_12_REPORT.md, STEP_3_13_REPORT.md — implementation reports and host-test summaries.

The quantized parameters and manifests are under the root fpga_export/v5_spatial/ directory.

## Host-side tests

The step reports state that eight MSVC x64 regression suites passed with no mismatches against the Python integer reference on the test vectors they cover. This is a reported host-side result, not a hardware result; reproduce it before relying on it for release sign-off.

The C++ test generator scripts reside in testbench/. The checked-in Tcl script adds testbench/test_v5_spatial_top.cpp as the HLS testbench. Run it from the hls_project/ working directory unless you have verified and adjusted its relative paths.

## Hardware validation workflow

On a compatible workstation:

1. Confirm the Vitis HLS version and Tcl command compatibility.
2. Run C simulation and fix compile or functional issues.
3. Run C synthesis and inspect measured BRAM, DSP, LUT, FF, latency, and timing.
4. Run C/RTL co-simulation.
5. Export the IP and integrate it in Vivado with the Zynq Processing System.
6. Generate the bitstream and .hwh file.
7. Load the overlay on the physical PYNQ-Z2, run inference, and compare logits with the integer reference.
8. Measure board accuracy and latency on the held-out test set.

No .bit, .hwh, synthesis report, or physical-board result should be described as completed until it has actually been generated and verified.

## Resource estimates

The source documentation has included parameter-size and BRAM estimates, but activation buffering and HLS scheduling can materially affect utilization. Treat resource fit and 100 MHz timing as unknown until measured synthesis and implementation reports are available.

## Accuracy

The root README documents the current software/reference evaluation as 238/346 (68.7861%) on the non-transistor test subset. Do not describe this as FPGA-measured accuracy.
