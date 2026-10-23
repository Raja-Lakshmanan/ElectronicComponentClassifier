# ==============================================================================
# run_hls.tcl — Vitis HLS Automated Build Script for V5-Spatial Classifier
# Target: PYNQ-Z2 (AMD/Xilinx Zynq-7000 SoC xc7z020clg400-1)
# ==============================================================================

# Create project
open_project -reset proj_v5_spatial
set_top v5_spatial_top

# Add synthesizable C++ source files
add_files src/v5_spatial_top.cpp
add_files src/component_classifier.cpp
add_files src/conv1.cpp
add_files src/conv2.cpp
add_files src/conv3.cpp
add_files src/spatial_pool.cpp
add_files src/fc1.cpp
add_files src/fc2.cpp

# Add C-simulation testbench
add_files -tb testbench/test_v5_spatial_top.cpp

# Open solution targeting PYNQ-Z2
open_solution -reset "solution1" -flow_target vivado
set_part {xc7z020clg400-1}
create_clock -period 10.0 -name default
set_clock_uncertainty 2.7

# Step 1: C Simulation
puts "=============================================================================="
puts ">>> STEP 1: RUNNING C SIMULATION (csim_design) <<<"
puts "=============================================================================="
csim_design -clean

# Step 2: C Synthesis
puts "=============================================================================="
puts ">>> STEP 2: RUNNING C SYNTHESIS (csynth_design) <<<"
puts "=============================================================================="
csynth_design

# Step 3: C/RTL Co-simulation
puts "=============================================================================="
puts ">>> STEP 3: RUNNING CO-SIMULATION (cosim_design) <<<"
puts "=============================================================================="
cosim_design -rtl verilog -trace_level all

# Step 4: Export RTL IP Package for Vivado Block Design
puts "=============================================================================="
puts ">>> STEP 4: EXPORTING IP PACKAGE (export_design) <<<"
puts "=============================================================================="
export_design -format ip_catalog -rtl verilog -description "V5-Spatial 9-Class Electronic Component Classifier" -vendor "user" -version "1.0"

puts "=============================================================================="
puts ">>> VITIS HLS BUILD COMPLETE <<<"
puts "=============================================================================="
exit
