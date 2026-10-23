# Electronic Component Classifier — V5-Spatial FPGA

A quantized convolutional neural network (CNN) for classifying electronic components, with an integer C++ inference pipeline being prepared for AMD/Xilinx Vitis HLS and the PYNQ-Z2 board.

> **Hardware status:** The C++ implementation has reported passing host-side regression tests. Vitis HLS C simulation, synthesis, RTL co-simulation, Vivado implementation, bitstream generation, and physical PYNQ-Z2 validation are **not yet confirmed**. Resource numbers in the design notes are estimates until synthesis reports are available.

## Project goals

- Train and evaluate a CNN for electronic-component image classification.
- Export an integer-friendly model using INT8 weights/activations and INT32 biases/logits.
- Build a portable C++ reference-matching inference pipeline.
- Prepare an HLS IP targeting the Zynq-7020 device on the PYNQ-Z2.

## Classes and deployment policy

The trained model produces 10 raw logits in this order:

| Index | Class |
|---:|---|
| 0 | capacitor |
| 1 | diode |
| 2 | fuse |
| 3 | inductor |
| 4 | integrated_circuit |
| 5 | led |
| 6 | potentiometer |
| 7 | push_button |
| 8 | resistor |
| 9 | transistor |

The current deployment policy selects the prediction only from classes **0–8**. The transistor logit (index 9) remains available for diagnostics but is excluded from argmax selection.

## V5-Spatial inference pipeline

1. Planar RGB input image, shape `[3, 64, 64]`, `uint8` (12,288 bytes).
2. A 256-entry input lookup table maps pixel values to signed INT8.
3. Three convolution stages with integer accumulators, ReLU, requantization, and max pooling.
4. Spatial adaptive pooling from `[64, 8, 8]` to `[64, 2, 2]`, followed by pooling requantization.
5. Flatten to 256 values, then FC1 (256 → 64) and FC2 (64 → 10).
6. Return 10 raw INT32 logits and the permitted nine-class prediction.

The exact quantization constants and parameter manifests are stored under `fpga_export/v5_spatial/`. The HLS implementation and testbenches are under `hls_project/`.

## Repository layout

- `model_v5_spatial.py` — V5-Spatial model definition.
- `train_v5_spatial.py` — training script.
- `integer_v5_reference.py` — Python integer inference/reference.
- `validate_v5_integer_reference.py` — integer-reference validation script.
- `export_v5_*.py` — model parameter and quantization export tools.
- `fpga_export/v5_spatial/` — exported parameters, manifests, and validation reports.
- `hls_project/src/` — portable C++ inference core and top-level wrapper.
- `hls_project/testbench/` — host/HLS testbenches and golden-vector generators.
- `hls_project/run_hls.tcl` — Vitis HLS project/build script.
- `hls_project/HLS_ARCHITECTURE.md` — architectural notes.
- `hls_project/PRE_SYNTHESIS_AUDIT.md` — static pre-synthesis review.
- `hls_project/STEP_3_11_REPORT.md`, `STEP_3_12_REPORT.md`, `STEP_3_13_REPORT.md` — implementation and test reports.

## Reported evaluation results

The current host/reference evaluation reports **68.7861% accuracy (238/346)** on 346 non-transistor test images. Treat this as a software/reference result, not an FPGA-measured result. The project notes also report eight MSVC host regression suites passing with zero mismatches against the Python integer reference on their test vectors. These results have not been independently reproduced by GitHub CI in this repository.

## Host-side test status

The HLS testbenches were reported to compile and pass on Windows/MSVC. To reproduce the tests, use a Windows environment with the Visual Studio C++ Build Tools installed and follow the source/testbench instructions in `hls_project/README.md` and `hls_project/STEP_3_13_REPORT.md`. Test generator scripts are provided under `hls_project/testbench/`.

No generic cross-platform command is asserted here because the checked-in test workflow is Windows/MSVC-specific and should be verified against your local project files.

## Vitis HLS and PYNQ-Z2 next steps

On a workstation with a compatible AMD/Xilinx Vitis HLS and Vivado installation:

1. Review `hls_project/run_hls.tcl` and confirm that all paths resolve from the `hls_project` directory.
2. Run HLS C simulation and inspect any compiler/interface errors.
3. Run C synthesis and inspect **measured** BRAM, DSP, LUT, FF, latency, and timing reports.
4. Run C/RTL co-simulation.
5. Export and integrate the IP in Vivado with the Zynq Processing System.
6. Generate the bitstream and hardware handoff file (`.hwh`).
7. Test on the physical PYNQ-Z2 using PYNQ Python and compare logits against the integer reference.
8. Evaluate the complete held-out test set on hardware and record measured accuracy and latency.

Do not treat the current HLS TCL script or PYNQ Python examples as hardware-verified until each stage has actually completed.

## Requirements

The Python training and evaluation scripts use Python and PyTorch; exact package versions should be taken from the source environment. HLS/bitstream stages require compatible AMD/Xilinx Vitis HLS and Vivado tools, plus a PYNQ-Z2 board for physical validation.

## Dataset and checkpoints

Large datasets and training checkpoints are not required to run the HLS inference core. Follow the original dataset's license and distribution terms before redistributing images. For FPGA work, retain the integer parameter manifests, embedded C++ parameters, and any golden test vectors required by the testbenches.

## License

No project license is declared by this README. Add a license only after deciding the terms under which you want others to use and redistribute the source and associated model assets.
