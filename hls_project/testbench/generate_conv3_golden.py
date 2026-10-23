"""
generate_conv3_golden.py
------------------------
Phase 3, Step 3.5 -- Golden Data Generator and C++ Code Generator for Conv3 Layer.

Generates:
  1. hls_project/src/conv3.cpp (with real exported weights and biases)
  2. hls_project/testbench/test_conv3.cpp (with embedded golden test vectors)

Uses integer_v5_reference.py to compute exact bit-for-bit intermediate activations:
  - Real test image (Capacitors_aluminum_electrolytic_00007.jpg) passed through
    input quantization, Conv1, and Conv2 to yield real Conv3 input [32][16][16]
  - Conv3 golden output [64][8][8] (4,096 elements)
  - Synthetic boundary cases (zeros, positive-extreme +127, negative-extreme -128, impulse)
"""

from pathlib import Path
from PIL import Image
import sys
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from integer_v5_reference import IntegerV5Reference

def main():
    print("=" * 80)
    print("  Generating Conv3 C++ Implementation & Golden Test Vectors")
    print("=" * 80)

    ref = IntegerV5Reference()
    w1 = ref.w_conv1_mat
    b1 = ref.b_conv1
    m1 = int(ref.m[0])
    s1 = int(ref.s[0])

    w2_mat = ref.w_conv2_mat
    b2 = ref.b_conv2
    m2 = int(ref.m[1])
    s2 = int(ref.s[1])

    w3 = ref.w_conv3  # [64, 32, 3, 3] int8
    w3_mat = ref.w_conv3_mat
    b3 = np.load(PROJECT_DIR / "fpga_export/v5_spatial/integer_params/conv3_bias_int32.npy")  # [64] int32
    m3 = int(ref.m[2])
    s3 = int(ref.s[2])

    print(f"Loaded Conv3 Weights: shape {w3.shape}, min {w3.min()}, max {w3.max()}")
    print(f"Loaded Conv3 Biases:  shape {b3.shape}, min {b3.min()}, max {b3.max()}")
    print(f"Requantization params: M={m3}, S={s3}")

    # =========================================================================
    # 1. Format Weights and Biases for hls_project/src/conv3.cpp
    # =========================================================================
    w_lines = []
    for m in range(64):
        w_lines.append(f"    {{ // Filter {m}")
        for c in range(32):
            w_lines.append(f"        {{ // In-channel {c}")
            for ky in range(3):
                row_str = ", ".join(f"{int(w3[m, c, ky, kx]):4d}" for kx in range(3))
                w_lines.append(f"            {{{row_str}}},")
            w_lines.append("        },")
        w_lines.append("    },")
    weights_str = "\n".join(w_lines)

    biases_str = ", ".join(f"{int(val)}" for val in b3)

    conv3_cpp_content = f"""#include "conv3.h"

// ============================================================================
// Exported Weights: [64][32][3][3] signed INT8 (18,432 elements)
// Source: fpga_export/v5_spatial/integer_params/conv3_weight_int8.npy
// Quantization Scale: 0.004656787932388426
// ============================================================================
const int8_t CONV3_WEIGHTS[CONV3_OUT_CH][CONV3_IN_CH][CONV3_KERNEL_SIZE][CONV3_KERNEL_SIZE] = {{
{weights_str}
}};

// ============================================================================
// Exported Biases: [64] signed INT32 (64 elements)
// Source: fpga_export/v5_spatial/integer_params/conv3_bias_int32.npy
// Pre-scaled by (s_in * s_w) = 0.0006762128795052959
// ============================================================================
const int32_t CONV3_BIASES[CONV3_OUT_CH] = {{
    {biases_str}
}};

// ============================================================================
// Fixed-Point Requantization Scalar Function
// Formula: ((acc * M + (1LL << (S - 1))) >> S), clamped to [-128, 127]
// ============================================================================
int8_t conv3_requantize(int64_t acc, int64_t M, int S) {{
    if (S <= 0) {{
        int64_t prod = acc * M;
        if (prod > 127) return 127;
        if (prod < -128) return -128;
        return static_cast<int8_t>(prod);
    }}
    int64_t rounding_bias = 1LL << (S - 1);
    int64_t prod = acc * M;
    int64_t rounded = (prod + rounding_bias) >> S;
    if (rounded > 127) return 127;
    if (rounded < -128) return -128;
    return static_cast<int8_t>(rounded);
}}

// ============================================================================
// Parameterized Conv3 Execution Engine
// ============================================================================
void conv3_custom(
    const int8_t input[CONV3_IN_CH][CONV3_IN_H][CONV3_IN_W],
    int8_t output[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W],
    const int8_t weights[CONV3_OUT_CH][CONV3_IN_CH][CONV3_KERNEL_SIZE][CONV3_KERNEL_SIZE],
    const int32_t biases[CONV3_OUT_CH],
    int64_t multiplier,
    int shift
) {{
    // Intermediate pre-pool buffer: 64 channels, 16 rows, 16 cols (16 KB)
    int8_t conv_out[CONV3_OUT_CH][CONV3_CONV_H][CONV3_CONV_W];

    // 1. Convolution (3x3, pad 1, stride 1) + Bias Addition + ReLU + Requantization
    for (int m = 0; m < CONV3_OUT_CH; ++m) {{
        const int32_t b_val = biases[m];

        for (int y = 0; y < CONV3_CONV_H; ++y) {{
            for (int x = 0; x < CONV3_CONV_W; ++x) {{
                int64_t acc = 0;

                for (int c = 0; c < CONV3_IN_CH; ++c) {{
                    for (int ky = 0; ky < CONV3_KERNEL_SIZE; ++ky) {{
                        int in_y = y + ky - CONV3_PAD;
                        for (int kx = 0; kx < CONV3_KERNEL_SIZE; ++kx) {{
                            int in_x = x + kx - CONV3_PAD;

                            int8_t in_val = 0;
                            if (in_y >= 0 && in_y < CONV3_IN_H && in_x >= 0 && in_x < CONV3_IN_W) {{
                                in_val = input[c][in_y][in_x];
                            }}
                            acc += static_cast<int32_t>(in_val) * static_cast<int32_t>(weights[m][c][ky][kx]);
                        }}
                    }}
                }}

                // Add pre-scaled INT32 bias
                acc += b_val;

                // Integer ReLU
                if (acc < 0) {{
                    acc = 0;
                }}

                // Fixed-point requantization with round-half-up and saturation
                conv_out[m][y][x] = conv3_requantize(acc, multiplier, shift);
            }}
        }}
    }}

    // 2. 2x2 Max Pooling with stride 2 -> [64][8][8]
    for (int m = 0; m < CONV3_OUT_CH; ++m) {{
        for (int oy = 0; oy < CONV3_OUT_H; ++oy) {{
            for (int ox = 0; ox < CONV3_OUT_W; ++ox) {{
                int iy = oy * 2;
                int ix = ox * 2;

                int8_t v00 = conv_out[m][iy][ix];
                int8_t v01 = conv_out[m][iy][ix + 1];
                int8_t v10 = conv_out[m][iy + 1][ix];
                int8_t v11 = conv_out[m][iy + 1][ix + 1];

                int8_t max_val = v00;
                if (v01 > max_val) max_val = v01;
                if (v10 > max_val) max_val = v10;
                if (v11 > max_val) max_val = v11;

                output[m][oy][ox] = max_val;
            }}
        }}
    }}
}}

// ============================================================================
// Standard Conv3 Forward Pass (Uses frozen on-chip parameters)
// ============================================================================
void conv3(
    const int8_t input[CONV3_IN_CH][CONV3_IN_H][CONV3_IN_W],
    int8_t output[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W]
) {{
    conv3_custom(input, output, CONV3_WEIGHTS, CONV3_BIASES, CONV3_M, CONV3_S);
}}
"""

    conv3_cpp_path = PROJECT_DIR / "hls_project/src/conv3.cpp"
    with open(conv3_cpp_path, "w", encoding="utf-8") as f:
        f.write(conv3_cpp_content)
    print(f"Generated: {conv3_cpp_path.name}")

    # =========================================================================
    # 2. Compute Golden Reference Tensors
    # =========================================================================
    # Real test image passed through input quantization, Conv1, and Conv2
    p_img = PROJECT_DIR / "cleaned_dataset/test/capacitor/Capacitors_aluminum_electrolytic_00007.jpg"
    img = Image.open(p_img).convert("RGB").resize((64, 64), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    u8 = np.transpose(arr, (2, 0, 1))[np.newaxis, :]
    x0_real = ref.quantize_input_image(u8)
    x1_real = ref._conv2d_block(x0_real, w1, b1, m1, s1)
    x2_real = ref._conv2d_block(x1_real, w2_mat, b2, m2, s2)
    y_real = ref._conv2d_block(x2_real, w3_mat, b3, m3, s3)[0]
    print(f"Real sample Conv3: in shape {x2_real[0].shape} (range [{x2_real.min()}, {x2_real.max()}]) -> out shape {y_real.shape} (range [{y_real.min()}, {y_real.max()}])")

    # Boundary 1: All Zeros input [32, 16, 16]
    x2_zero = np.zeros((1, 32, 16, 16), dtype=np.int8)
    y_zero = ref._conv2d_block(x2_zero, w3_mat, b3, m3, s3)[0]
    print(f"Zero input Conv3: out shape {y_zero.shape} (range [{y_zero.min()}, {y_zero.max()}])")

    # Boundary 2: Positive extreme (+127) [32, 16, 16]
    x2_pos = np.full((1, 32, 16, 16), 127, dtype=np.int8)
    y_pos = ref._conv2d_block(x2_pos, w3_mat, b3, m3, s3)[0]
    print(f"Positive extreme Conv3: out shape {y_pos.shape} (range [{y_pos.min()}, {y_pos.max()}])")

    # Boundary 3: Negative extreme (-128) [32, 16, 16]
    x2_neg = np.full((1, 32, 16, 16), -128, dtype=np.int8)
    y_neg = ref._conv2d_block(x2_neg, w3_mat, b3, m3, s3)[0]
    print(f"Negative extreme Conv3: out shape {y_neg.shape} (range [{y_neg.min()}, {y_neg.max()}])")

    # Boundary 4: Impulse corner and center [32, 16, 16]
    x2_imp = np.zeros((1, 32, 16, 16), dtype=np.int8)
    x2_imp[0, 0, 0, 0] = 100
    x2_imp[0, 10, 8, 8] = 100
    y_imp = ref._conv2d_block(x2_imp, w3_mat, b3, m3, s3)[0]
    print(f"Impulse input Conv3: out shape {y_imp.shape} (range [{y_imp.min()}, {y_imp.max()}])")

    # =========================================================================
    # 3. Format Golden Data into C++ Arrays
    # =========================================================================
    def array_to_c_flat(arr, name, dtype="int8_t", per_line=16):
        flat = arr.flatten()
        lines = [f"static const {dtype} {name}[{flat.size}] = {{"]
        for i in range(0, flat.size, per_line):
            chunk = flat[i:i+per_line]
            chunk_str = ", ".join(f"{int(v):4d}" for v in chunk)
            lines.append(f"    {chunk_str},")
        lines.append("};")
        return "\n".join(lines)

    c_real_in = array_to_c_flat(x2_real[0], "GOLDEN_REAL_IN")
    c_real_out = array_to_c_flat(y_real, "GOLDEN_REAL_OUT")
    c_zero_out = array_to_c_flat(y_zero, "GOLDEN_ZERO_OUT")
    c_pos_out = array_to_c_flat(y_pos, "GOLDEN_POS_OUT")
    c_neg_out = array_to_c_flat(y_neg, "GOLDEN_NEG_OUT")
    c_imp_in = array_to_c_flat(x2_imp[0], "GOLDEN_IMP_IN")
    c_imp_out = array_to_c_flat(y_imp, "GOLDEN_IMP_OUT")

    test_conv3_cpp_content = f"""/**
 * test_conv3.cpp
 * --------------
 * Standalone C++ verification testbench for Conv3 layer.
 *
 * Compares C++ implementation against golden tensors generated by
 * integer_v5_reference.py on:
 *   1. Real camera test image passed through input quantization, Conv1, and Conv2
 *   2. Zero input boundary test (bias addition, ReLU clipping, constant channel response)
 *   3. Positive dynamic range test (+127 input, padding border attenuation)
 *   4. Negative input boundary test (-128 input, ReLU cut-off)
 *   5. Corner & center impulse test (padding border semantics, max pool decimation)
 *   6. Scalar fixed-point requantization boundary unit tests
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../src/conv3.h"

// ============================================================================
// EMBEDDED GOLDEN DATA ARRAYS (Generated from integer_v5_reference.py)
// ============================================================================
{c_real_in}

{c_real_out}

{c_zero_out}

{c_pos_out}

{c_neg_out}

{c_imp_in}

{c_imp_out}

// ============================================================================
// TEST HARNESS HELPERS
// ============================================================================
struct TestResult {{
    const char* name;
    int total_elements;
    int mismatches;
    int max_abs_error;
    bool passed;
}};

void compare_tensors(
    const int8_t test_out[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W],
    const int8_t* golden_flat,
    int total_elements,
    int& mismatches,
    int& max_abs_error
) {{
    mismatches = 0;
    max_abs_error = 0;
    const int8_t* test_flat = reinterpret_cast<const int8_t*>(test_out);

    for (int i = 0; i < total_elements; ++i) {{
        int diff = std::abs(static_cast<int>(test_flat[i]) - static_cast<int>(golden_flat[i]));
        if (diff > 0) {{
            mismatches++;
            if (diff > max_abs_error) {{
                max_abs_error = diff;
            }}
        }}
    }}
}}

// ============================================================================
// TEST SUITES
// ============================================================================

// Test 1: Real Image Intermediate Activations (post-Conv2)
TestResult run_real_image_test() {{
    const auto* in_tensor = reinterpret_cast<const int8_t(*)[CONV3_IN_H][CONV3_IN_W]>(GOLDEN_REAL_IN);
    int8_t out_tensor[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W];

    conv3(in_tensor, out_tensor);

    TestResult res;
    res.name = "Real Test Image (Post-Conv2 Intermediate)";
    res.total_elements = CONV3_OUT_CH * CONV3_OUT_H * CONV3_OUT_W;
    compare_tensors(out_tensor, GOLDEN_REAL_OUT, res.total_elements, res.mismatches, res.max_abs_error);
    res.passed = (res.mismatches == 0 && res.max_abs_error == 0);
    return res;
}}

// Test 2: All Zeros Boundary Test
TestResult run_zero_input_test() {{
    int8_t zero_in[CONV3_IN_CH][CONV3_IN_H][CONV3_IN_W] = {{0}};
    int8_t out_tensor[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W];

    conv3(zero_in, out_tensor);

    TestResult res;
    res.name = "Zero Input Boundary (Bias & ReLU test)";
    res.total_elements = CONV3_OUT_CH * CONV3_OUT_H * CONV3_OUT_W;
    compare_tensors(out_tensor, GOLDEN_ZERO_OUT, res.total_elements, res.mismatches, res.max_abs_error);
    res.passed = (res.mismatches == 0 && res.max_abs_error == 0);
    return res;
}}

// Test 3: Positive Dynamic Range Test (+127 everywhere)
TestResult run_positive_saturation_test() {{
    int8_t pos_in[CONV3_IN_CH][CONV3_IN_H][CONV3_IN_W];
    std::memset(pos_in, 127, sizeof(pos_in));
    int8_t out_tensor[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W];

    conv3(pos_in, out_tensor);

    TestResult res;
    res.name = "Positive Range (+127 all elements)";
    res.total_elements = CONV3_OUT_CH * CONV3_OUT_H * CONV3_OUT_W;
    compare_tensors(out_tensor, GOLDEN_POS_OUT, res.total_elements, res.mismatches, res.max_abs_error);
    res.passed = (res.mismatches == 0 && res.max_abs_error == 0);
    return res;
}}

// Test 4: Negative Input Test (-128 everywhere)
TestResult run_negative_relu_test() {{
    int8_t neg_in[CONV3_IN_CH][CONV3_IN_H][CONV3_IN_W];
    std::memset(neg_in, -128, sizeof(neg_in));
    int8_t out_tensor[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W];

    conv3(neg_in, out_tensor);

    TestResult res;
    res.name = "Negative Input (-128 all elements)";
    res.total_elements = CONV3_OUT_CH * CONV3_OUT_H * CONV3_OUT_W;
    compare_tensors(out_tensor, GOLDEN_NEG_OUT, res.total_elements, res.mismatches, res.max_abs_error);
    res.passed = (res.mismatches == 0 && res.max_abs_error == 0);
    return res;
}}

// Test 5: Impulse Corner & Center Test
TestResult run_impulse_padding_test() {{
    const auto* in_tensor = reinterpret_cast<const int8_t(*)[CONV3_IN_H][CONV3_IN_W]>(GOLDEN_IMP_IN);
    int8_t out_tensor[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W];

    conv3(in_tensor, out_tensor);

    TestResult res;
    res.name = "Impulse (Padding border & center test)";
    res.total_elements = CONV3_OUT_CH * CONV3_OUT_H * CONV3_OUT_W;
    compare_tensors(out_tensor, GOLDEN_IMP_OUT, res.total_elements, res.mismatches, res.max_abs_error);
    res.passed = (res.mismatches == 0 && res.max_abs_error == 0);
    return res;
}}

// Test 6: Scalar Requantization Boundary Tests
TestResult run_scalar_requant_tests() {{
    TestResult res;
    res.name = "Scalar Requantization & Saturation Limits";
    res.total_elements = 6;
    res.mismatches = 0;
    res.max_abs_error = 0;

    // Test cases: (acc, M, S, expected)
    struct TestCase {{
        int64_t acc;
        int64_t M;
        int S;
        int8_t expected;
    }} cases[] = {{
        {{0, CONV3_M, CONV3_S, 0}},                   // Zero accumulator -> 0
        {{338, CONV3_M, CONV3_S, 0}},                 // Max Conv3 bias (338) -> 0
        {{-220, CONV3_M, CONV3_S, 0}},                // Min Conv3 bias (-220) -> 0 (clipped post-ReLU)
        {{1000000000LL, CONV3_M, CONV3_S, 127}},     // Large positive accumulator -> saturated to 127
        {{-1000000000LL, CONV3_M, CONV3_S, -128}},   // Large negative accumulator -> saturated to -128
        {{32, 1, 5, 1}}                               // (32 * 1 + 16) >> 5 = 48 >> 5 = 1
    }};

    for (const auto& tc : cases) {{
        int8_t actual = conv3_requantize(tc.acc, tc.M, tc.S);
        if (actual != tc.expected) {{
            res.mismatches++;
            int diff = std::abs(static_cast<int>(actual) - static_cast<int>(tc.expected));
            if (diff > res.max_abs_error) res.max_abs_error = diff;
        }}
    }}

    res.passed = (res.mismatches == 0);
    return res;
}}

// ============================================================================
// MAIN RUNNER
// ============================================================================
int main() {{
    std::cout << "================================================================================\\n";
    std::cout << "  Conv3 Layer C++ Verification Testbench\\n";
    std::cout << "  Comparing Portable C++ Implementation vs Python Integer Reference\\n";
    std::cout << "================================================================================\\n\\n";

    TestResult results[] = {{
        run_real_image_test(),
        run_zero_input_test(),
        run_positive_saturation_test(),
        run_negative_relu_test(),
        run_impulse_padding_test(),
        run_scalar_requant_tests()
    }};

    std::cout << std::left 
              << std::setw(45) << "Test Suite"
              << std::setw(12) << "Elements"
              << std::setw(14) << "Mismatches"
              << std::setw(14) << "Max Abs Err"
              << "Status\\n";
    std::cout << "--------------------------------------------------------------------------------\\n";

    bool all_passed = true;
    for (const auto& r : results) {{
        std::cout << std::left 
                  << std::setw(45) << r.name
                  << std::setw(12) << r.total_elements
                  << std::setw(14) << r.mismatches
                  << std::setw(14) << r.max_abs_error
                  << (r.passed ? "[PASS]" : "[FAIL]") << "\\n";
        if (!r.passed) {{
            all_passed = false;
        }}
    }}

    std::cout << "================================================================================\\n";
    if (all_passed) {{
        std::cout << "  VERDICT: ALL 6 TEST SUITES PASSED (0 Mismatches across all tensors)\\n";
        std::cout << "  Conv3 portable C++ implementation is BIT-EXACT with Python integer reference.\\n";
    }} else {{
        std::cout << "  VERDICT: SOME TESTS FAILED. See mismatches above.\\n";
    }}
    std::cout << "================================================================================\\n";

    return all_passed ? 0 : 1;
}}
"""

    test_conv3_cpp_path = PROJECT_DIR / "hls_project/testbench/test_conv3.cpp"
    with open(test_conv3_cpp_path, "w", encoding="utf-8") as f:
        f.write(test_conv3_cpp_content)
    print(f"Generated: {test_conv3_cpp_path.name}")
    print("Done generating Conv3 files!")

if __name__ == "__main__":
    main()
