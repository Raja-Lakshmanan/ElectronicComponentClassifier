"""
generate_fc1_golden.py
----------------------
Phase 3, Step 3.7 -- Golden Data Generator and C++ Code Generator for FC1 Layer.

Generates:
  1. hls_project/src/fc1.cpp (with real exported INT8 weights and INT32 biases)
  2. hls_project/testbench/test_fc1.cpp (with embedded golden test vectors)

Uses integer_v5_reference.py to compute exact bit-for-bit intermediate activations:
  - Real test image (Capacitors_aluminum_electrolytic_00007.jpg) propagated through
    input quantization, Conv1, Conv2, Conv3, spatial pooling, and flattening to
    yield real FC1 input [256] and golden FC1 output [64].
  - Synthetic boundary cases:
      * All-zero input [256]
      * Maximum non-negative input (+127 for all 256 elements)
      * Mixed positive and zero activations (deterministic pattern)
      * Sparse impulse vectors (indices 0, 128, 255)
      * Bias, ReLU, requantization, and saturation edge cases
"""

from pathlib import Path
from PIL import Image
import sys
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from integer_v5_reference import IntegerV5Reference, INPUT_SIZE


def format_1d_array(arr_1d: np.ndarray, indent: str = "    ", per_line: int = 16) -> str:
    lines = []
    for i in range(0, len(arr_1d), per_line):
        chunk = arr_1d[i:i + per_line]
        line = ", ".join(f"{int(v):4d}" for v in chunk)
        lines.append(f"{indent}{line},")
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("  Generating FC1 C++ Implementation & Golden Test Vectors")
    print("=" * 80)

    ref = IntegerV5Reference()
    params_dir = PROJECT_DIR / "fpga_export/v5_spatial/integer_params"

    w_fc1 = np.load(params_dir / "fc1_weight_int8.npy").astype(np.int8)  # (64, 256)
    b_fc1 = np.load(params_dir / "fc1_bias_int32.npy").astype(np.int32)  # (64,)
    m_fc1 = int(ref.m[3])  # 1531277679
    s_fc1 = int(ref.s[3])  # 38

    print(f"Loaded FC1 Weights: shape {w_fc1.shape}, dtype {w_fc1.dtype}, min {w_fc1.min()}, max {w_fc1.max()}")
    print(f"Loaded FC1 Biases:  shape {b_fc1.shape}, dtype {b_fc1.dtype}, min {b_fc1.min()}, max {b_fc1.max()}")
    print(f"Requantization params: M={m_fc1}, S={s_fc1}")

    # =========================================================================
    # 1. Generate hls_project/src/fc1.cpp
    # =========================================================================
    fc1_cpp_path = PROJECT_DIR / "hls_project/src/fc1.cpp"
    print(f"Generating C++ source: {fc1_cpp_path}")

    w_lines = []
    for k in range(64):
        w_lines.append(f"    {{ // Neuron {k}")
        row_str = format_1d_array(w_fc1[k], indent="        ", per_line=16)
        w_lines.append(row_str)
        w_lines.append("    },")
    weights_str = "\n".join(w_lines)

    biases_str = format_1d_array(b_fc1, indent="    ", per_line=8)

    fc1_cpp_content = f"""#include "fc1.h"

// ============================================================================
// Exported Weights: [64][256] signed INT8 (16,384 elements)
// Source: fpga_export/v5_spatial/integer_params/fc1_weight_int8.npy
// Quantization Scale: 0.0037154247910957637
// ============================================================================
const int8_t FC1_WEIGHTS[FC1_OUT_FEATURES][FC1_IN_FEATURES] = {{
{weights_str}
}};

// ============================================================================
// Exported Biases: [64] signed INT32 (64 elements)
// Source: fpga_export/v5_spatial/integer_params/fc1_bias_int32.npy
// Pre-scaled by (s_in * s_w) = 0.0011899198890980703
// ============================================================================
const int32_t FC1_BIASES[FC1_OUT_FEATURES] = {{
{biases_str}
}};

// ============================================================================
// Fixed-Point Requantization Scalar Function
// Formula: ((acc * M + (1LL << (S - 1))) >> S), clamped to [-128, 127]
// Post-ReLU acc >= 0, so output is in [0, 127]
// ============================================================================
int8_t fc1_requantize(int64_t acc, int64_t M, int S) {{
    if (acc <= 0) return 0;
    if (S <= 0) {{
        int64_t prod = acc * M;
        if (prod > 127) return 127;
        return static_cast<int8_t>(prod);
    }}
    int64_t rounding_bias = 1LL << (S - 1);
    int64_t scaled = (acc * M + rounding_bias) >> S;
    if (scaled > 127) return 127;
    if (scaled < -128) return -128;
    return static_cast<int8_t>(scaled);
}}

// ============================================================================
// FC1 Layer Forward Pass (Frozen on-chip weights and biases)
// ============================================================================
void fc1(
    const int8_t input[FC1_IN_FEATURES],
    int8_t output[FC1_OUT_FEATURES]
) {{
    for (int k = 0; k < FC1_OUT_FEATURES; ++k) {{
        int32_t acc = FC1_BIASES[k];
        for (int j = 0; j < FC1_IN_FEATURES; ++j) {{
            acc += static_cast<int32_t>(input[j]) * static_cast<int32_t>(FC1_WEIGHTS[k][j]);
        }}
        // Integer ReLU
        int32_t acc_relu = (acc > 0) ? acc : 0;
        // Fixed-point requantization
        output[k] = fc1_requantize(acc_relu, FC1_M, FC1_S);
    }}
}}

// ============================================================================
// Parameterized FC1 Layer Forward Pass (For custom weights/biases/params testing)
// ============================================================================
void fc1_custom(
    const int8_t input[FC1_IN_FEATURES],
    int8_t output[FC1_OUT_FEATURES],
    const int8_t weights[FC1_OUT_FEATURES][FC1_IN_FEATURES],
    const int32_t biases[FC1_OUT_FEATURES],
    int64_t multiplier,
    int shift
) {{
    for (int k = 0; k < FC1_OUT_FEATURES; ++k) {{
        int32_t acc = biases[k];
        for (int j = 0; j < FC1_IN_FEATURES; ++j) {{
            acc += static_cast<int32_t>(input[j]) * static_cast<int32_t>(weights[k][j]);
        }}
        int32_t acc_relu = (acc > 0) ? acc : 0;
        output[k] = fc1_requantize(acc_relu, multiplier, shift);
    }}
}}
"""

    with open(fc1_cpp_path, "w", encoding="utf-8") as f:
        f.write(fc1_cpp_content)
    print(f"Successfully generated {fc1_cpp_path} ({len(fc1_cpp_content)} bytes)")

    # =========================================================================
    # 2. Compute Test Vectors using integer_v5_reference.py
    # =========================================================================
    # Suite 1: Real Camera Image Propagation
    img_path = PROJECT_DIR / "cleaned_dataset/test/capacitor/Capacitors_aluminum_electrolytic_00007.jpg"
    print(f"\nLoading real test image: {img_path.name}")
    pil_img = Image.open(img_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    img_u8 = np.transpose(np.array(pil_img, dtype=np.uint8), (2, 0, 1))[np.newaxis, :]

    x0 = ref.quantize_input_image(img_u8)
    x1 = ref._conv2d_block(x0, ref.w_conv1_mat, ref.b_conv1, int(ref.m[0]), int(ref.s[0]))
    x2 = ref._conv2d_block(x1, ref.w_conv2_mat, ref.b_conv2, int(ref.m[1]), int(ref.s[1]))
    x3 = ref._conv2d_block(x2, ref.w_conv3_mat, ref.b_conv3, int(ref.m[2]), int(ref.s[2]))
    x4 = ref._spatial_adaptive_pool(x3)  # (1, 64, 2, 2) requantized

    real_in = x4.reshape(256).astype(np.int8)  # (256,)
    golden_real_out = ref._fc1_block(x4)[0].astype(np.int8)  # (64,)

    print(f"Real FC1 input:  min {real_in.min()}, max {real_in.max()}, nonzeros {(real_in != 0).sum()}/256")
    print(f"Real FC1 output: min {golden_real_out.min()}, max {golden_real_out.max()}, nonzeros {(golden_real_out != 0).sum()}/64")

    # Suite 2: All-Zero Input Vector
    zero_in = np.zeros(256, dtype=np.int8)
    golden_zero_out = ref._fc1_block(zero_in.reshape(1, 256))[0].astype(np.int8)
    print(f"Zero FC1 output: min {golden_zero_out.min()}, max {golden_zero_out.max()}, nonzeros {(golden_zero_out != 0).sum()}/64")

    # Suite 3: Positive-Extreme Input Vector (+127)
    max_in = np.full(256, 127, dtype=np.int8)
    golden_max_out = ref._fc1_block(max_in.reshape(1, 256))[0].astype(np.int8)
    print(f"Max +127 FC1 output: min {golden_max_out.min()}, max {golden_max_out.max()}, nonzeros {(golden_max_out != 0).sum()}/64")

    # Suite 4: Mixed Positive and Zero Values
    # Deterministic ramp pattern in [0, 127]
    mixed_in = np.array([(j * 17) % 128 if (j % 3 != 0) else 0 for j in range(256)], dtype=np.int8)
    golden_mixed_out = ref._fc1_block(mixed_in.reshape(1, 256))[0].astype(np.int8)
    print(f"Mixed FC1 input: min {mixed_in.min()}, max {mixed_in.max()}, nonzeros {(mixed_in != 0).sum()}/256")
    print(f"Mixed FC1 output: min {golden_mixed_out.min()}, max {golden_mixed_out.max()}, nonzeros {(golden_mixed_out != 0).sum()}/64")

    # Suite 5: Sparse Impulse Vectors
    # 5a: Impulse at index 0 (first position, value 100)
    imp0_in = np.zeros(256, dtype=np.int8)
    imp0_in[0] = 100
    golden_imp0_out = ref._fc1_block(imp0_in.reshape(1, 256))[0].astype(np.int8)

    # 5b: Impulse at index 128 (middle position, value 100)
    imp128_in = np.zeros(256, dtype=np.int8)
    imp128_in[128] = 100
    golden_imp128_out = ref._fc1_block(imp128_in.reshape(1, 256))[0].astype(np.int8)

    # 5c: Impulse at index 255 (last position, value 100)
    imp255_in = np.zeros(256, dtype=np.int8)
    imp255_in[255] = 100
    golden_imp255_out = ref._fc1_block(imp255_in.reshape(1, 256))[0].astype(np.int8)

    print(f"Impulse 0 output nonzeros:   {(golden_imp0_out != 0).sum()}/64")
    print(f"Impulse 128 output nonzeros: {(golden_imp128_out != 0).sum()}/64")
    print(f"Impulse 255 output nonzeros: {(golden_imp255_out != 0).sum()}/64")

    # =========================================================================
    # 3. Generate test_fc1.cpp
    # =========================================================================
    testbench_path = PROJECT_DIR / "hls_project/testbench/test_fc1.cpp"
    print(f"\nWriting testbench to: {testbench_path}")

    real_in_str = format_1d_array(real_in)
    golden_real_out_str = format_1d_array(golden_real_out)
    golden_zero_out_str = format_1d_array(golden_zero_out)
    golden_max_out_str = format_1d_array(golden_max_out)
    mixed_in_str = format_1d_array(mixed_in)
    golden_mixed_out_str = format_1d_array(golden_mixed_out)
    golden_imp0_out_str = format_1d_array(golden_imp0_out)
    golden_imp128_out_str = format_1d_array(golden_imp128_out)
    golden_imp255_out_str = format_1d_array(golden_imp255_out)

    testbench_content = f"""/**
 * test_fc1.cpp
 * ------------
 * Standalone C++ verification testbench for FC1 Layer (classifier.0).
 *
 * Compares C++ implementation against golden tensors generated by
 * integer_v5_reference.py on:
 *   1. Real camera test image passed through input LUT, Conv1, Conv2, Conv3,
 *      spatial pooling, and flattening -> FC1 [256] -> [64]
 *   2. All-zero input vector [256] (bias-only behavior and ReLU cut-off)
 *   3. Positive-extreme input vector (+127 for all 256 inputs, saturation test)
 *   4. Mixed positive and zero values (deterministic activation pattern)
 *   5. Sparse impulse vectors (indices 0, 128, 255) to verify column weighting
 *   6. Scalar fixed-point requantization boundary unit tests
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../src/fc1.h"

// ============================================================================
// EMBEDDED GOLDEN DATA ARRAYS (Generated from integer_v5_reference.py)
// ============================================================================

// 1. Real Image Input [256]
static const int8_t GOLDEN_REAL_IN[256] = {{
{real_in_str}
}};

// 1. Real Image Golden Output [64]
static const int8_t GOLDEN_REAL_OUT[64] = {{
{golden_real_out_str}
}};

// 2. All-Zero Golden Output [64]
static const int8_t GOLDEN_ZERO_OUT[64] = {{
{golden_zero_out_str}
}};

// 3. Positive-Extreme Golden Output [64]
static const int8_t GOLDEN_MAX_OUT[64] = {{
{golden_max_out_str}
}};

// 4. Mixed Input [256]
static const int8_t GOLDEN_MIXED_IN[256] = {{
{mixed_in_str}
}};

// 4. Mixed Golden Output [64]
static const int8_t GOLDEN_MIXED_OUT[64] = {{
{golden_mixed_out_str}
}};

// 5a. Impulse 0 Golden Output [64]
static const int8_t GOLDEN_IMP0_OUT[64] = {{
{golden_imp0_out_str}
}};

// 5b. Impulse 128 Golden Output [64]
static const int8_t GOLDEN_IMP128_OUT[64] = {{
{golden_imp128_out_str}
}};

// 5c. Impulse 255 Golden Output [64]
static const int8_t GOLDEN_IMP255_OUT[64] = {{
{golden_imp255_out_str}
}};

// ============================================================================
// Verification Helper Function
// ============================================================================
int compare_vector_64(
    const char* suite_name,
    const int8_t cpp_out[FC1_OUT_FEATURES],
    const int8_t golden_out[FC1_OUT_FEATURES]
) {{
    int mismatches = 0;
    int max_abs_diff = 0;

    for (int k = 0; k < FC1_OUT_FEATURES; ++k) {{
        int8_t cpp_val = cpp_out[k];
        int8_t gold_val = golden_out[k];
        int diff = std::abs(static_cast<int>(cpp_val) - static_cast<int>(gold_val));

        if (diff > 0) {{
            mismatches++;
            if (diff > max_abs_diff) max_abs_diff = diff;
            if (mismatches <= 10) {{
                std::cout << "  [MISMATCH] [" << suite_name << "] k=" << k
                          << ": C++ = " << static_cast<int>(cpp_val)
                          << ", Golden = " << static_cast<int>(gold_val)
                          << " (diff=" << diff << ")\\n";
            }}
        }}
    }}

    std::cout << "  " << std::left << std::setw(38) << suite_name
              << " | Total: 64 | Mismatches: " << mismatches
              << " | MaxDiff: " << max_abs_diff
              << " | Result: " << (mismatches == 0 ? "PASS" : "FAIL")
              << "\\n";

    return mismatches;
}}

int main() {{
    std::cout << "================================================================================" << std::endl;
    std::cout << "  V5-Spatial HLS C++ Verification: FC1 Layer (Step 3.7)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    int total_failures = 0;

    static int8_t test_input[FC1_IN_FEATURES];
    static int8_t test_output[FC1_OUT_FEATURES];

    // ========================================================================
    // TEST SUITE 1: Real Camera Image Propagation Test
    // ========================================================================
    std::cout << "\\n[Test Suite 1/6] Real Camera Image Forward Pass..." << std::endl;
    std::memcpy(test_input, GOLDEN_REAL_IN, sizeof(test_input));
    fc1(test_input, test_output);
    total_failures += compare_vector_64("Suite 1: Real Test Image", test_output, GOLDEN_REAL_OUT);

    // ========================================================================
    // TEST SUITE 2: All-Zero Input Vector Test
    // ========================================================================
    std::cout << "\\n[Test Suite 2/6] All-Zero Input Vector (Bias & ReLU cut-off)..." << std::endl;
    std::memset(test_input, 0, sizeof(test_input));
    fc1(test_input, test_output);
    total_failures += compare_vector_64("Suite 2: All-Zero Input", test_output, GOLDEN_ZERO_OUT);

    // ========================================================================
    // TEST SUITE 3: Positive-Extreme Input Vector (+127)
    // ========================================================================
    std::cout << "\\n[Test Suite 3/6] Positive-Extreme Input Vector (+127 saturation)..." << std::endl;
    std::memset(test_input, 127, sizeof(test_input));
    fc1(test_input, test_output);
    total_failures += compare_vector_64("Suite 3: Max +127 Input", test_output, GOLDEN_MAX_OUT);

    // ========================================================================
    // TEST SUITE 4: Mixed Positive and Zero Values
    // ========================================================================
    std::cout << "\\n[Test Suite 4/6] Mixed Positive and Zero Activations..." << std::endl;
    std::memcpy(test_input, GOLDEN_MIXED_IN, sizeof(test_input));
    fc1(test_input, test_output);
    total_failures += compare_vector_64("Suite 4: Mixed Activations", test_output, GOLDEN_MIXED_OUT);

    // ========================================================================
    // TEST SUITE 5: Sparse Impulse Vectors (Indices 0, 128, 255)
    // ========================================================================
    std::cout << "\\n[Test Suite 5/6] Sparse Impulse Vectors (Column Indexing)..." << std::endl;

    // 5a: Impulse at index 0
    std::memset(test_input, 0, sizeof(test_input));
    test_input[0] = 100;
    fc1(test_input, test_output);
    total_failures += compare_vector_64("Suite 5a: Impulse at Index 0", test_output, GOLDEN_IMP0_OUT);

    // 5b: Impulse at index 128
    std::memset(test_input, 0, sizeof(test_input));
    test_input[128] = 100;
    fc1(test_input, test_output);
    total_failures += compare_vector_64("Suite 5b: Impulse at Index 128", test_output, GOLDEN_IMP128_OUT);

    // 5c: Impulse at index 255
    std::memset(test_input, 0, sizeof(test_input));
    test_input[255] = 100;
    fc1(test_input, test_output);
    total_failures += compare_vector_64("Suite 5c: Impulse at Index 255", test_output, GOLDEN_IMP255_OUT);

    // ========================================================================
    // TEST SUITE 6: Scalar Requantization Boundary Tests
    // ========================================================================
    std::cout << "\\n[Test Suite 6/6] Scalar Requantization Boundary Tests..." << std::endl;
    int requant_mismatches = 0;

    // Test cases: (acc, expected_q)
    // Formula: q = clip(((acc * 1531277679 + (1LL << 37)) >> 38), -128, 127)
    // ratio = 0.005570755744...
    // acc <= 0 -> 0 (ReLU)
    // acc = 90 -> 90 * 0.00557 = 0.501 -> 1
    // acc = 179 -> 179 * 0.00557 = 0.997 -> 1
    // acc = 180 -> 180 * 0.00557 = 1.002 -> 1
    // acc = 270 -> 270 * 0.00557 = 1.504 -> 2
    // acc = 22797 -> 22797 * 0.00557 = 126.996 -> 127
    // acc = 22800 -> 22800 * 0.00557 = 127.013 -> clipped to 127
    // acc = 4129149 -> theoretical max accumulation -> clipped to 127
    struct RequantTestCase {{
        int64_t acc;
        int8_t expected;
        const char* desc;
    }};

    RequantTestCase requant_cases[] = {{
        {{-1000,   0, "Negative accumulator (ReLU cut-off)"}},
        {{   -1,   0, "Negative boundary (ReLU cut-off)"}},
        {{    0,   0, "Zero accumulator"}},
        {{    1,   0, "Small positive (rounds to 0)"}},
        {{   89,   0, "Just below half (89/179.5 -> rounds to 0)"}},
        {{   90,   1, "Just above half (90*0.00557 = 0.501 -> rounds to 1)"}},
        {{  179,   1, "Just below 1.0 (rounds to 1)"}},
        {{  180,   1, "Just above 1.0 (rounds to 1)"}},
        {{  269,   1, "Just below 1.5 (rounds to 1)"}},
        {{  270,   2, "Just above 1.5 (rounds to 2)"}},
        {{22797, 127, "Just below saturation (rounds to 127)"}},
        {{22800, 127, "Saturation threshold (clipped to 127)"}},
        {{100000, 127, "Well above saturation (clipped to 127)"}},
        {{4129149LL, 127, "Theoretical max accumulation (clipped to 127)"}},
    }};

    int num_requant_cases = sizeof(requant_cases) / sizeof(requant_cases[0]);
    for (int i = 0; i < num_requant_cases; ++i) {{
        int8_t res = fc1_requantize(requant_cases[i].acc, FC1_M, FC1_S);
        if (res != requant_cases[i].expected) {{
            requant_mismatches++;
            std::cout << "  [MISMATCH] Case " << i << " (" << requant_cases[i].desc << "): "
                      << "acc=" << requant_cases[i].acc << " -> got " << static_cast<int>(res)
                      << ", expected " << static_cast<int>(requant_cases[i].expected) << "\\n";
        }}
    }}

    std::cout << "  " << std::left << std::setw(38) << "Suite 6: Scalar Requantization Tests"
              << " | Total: " << num_requant_cases
              << " | Mismatches: " << requant_mismatches
              << " | Result: " << (requant_mismatches == 0 ? "PASS" : "FAIL")
              << "\\n";

    total_failures += requant_mismatches;

    // ========================================================================
    // Final Summary
    // ========================================================================
    std::cout << "\\n================================================================================" << std::endl;
    if (total_failures == 0) {{
        std::cout << "  >>> ALL FC1 C++ VERIFICATION TESTS PASSED (0 MISMATCHES) <<<" << std::endl;
    }} else {{
        std::cout << "  >>> VERIFICATION FAILED: " << total_failures << " TOTAL MISMATCHES <<<" << std::endl;
    }}
    std::cout << "================================================================================" << std::endl;

    return (total_failures == 0) ? 0 : 1;
}}
"""

    with open(testbench_path, "w", encoding="utf-8") as f:
        f.write(testbench_content)

    print(f"Successfully generated {testbench_path} ({len(testbench_content)} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
