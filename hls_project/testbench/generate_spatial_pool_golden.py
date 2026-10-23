"""
generate_spatial_pool_golden.py
--------------------------------
Phase 3, Step 3.6 -- Golden Data Generator and C++ Testbench Generator for
Spatial Adaptive Average Pooling Layer.

Generates:
  1. hls_project/testbench/test_spatial_pool.cpp (with embedded golden test vectors)

Uses integer_v5_reference.py to compute exact bit-for-bit intermediate activations:
  - Real test image (Capacitors_aluminum_electrolytic_00007.jpg) propagated through
    input quantization, Conv1, Conv2, and Conv3 to yield real Conv3 output [64][8][8].
  - Pure Spatial Pooling golden output [64][2][2] (256 elements, Banker's round-to-even)
  - Requantized Spatial Pooling golden output [64][2][2] (matching ref._spatial_adaptive_pool)
  - Synthetic boundary cases:
      * All-zero input [64][8][8] -> [64][2][2]
      * Half-way boundary test patterns (sums around rem == 8)
      * Even vs Odd quotient tie-breaker suite (s=8, 24, 40, 56, 72, 88, 104, 120, ...)
      * Maximum non-negative INT8 values (+127 -> sum 2032 -> avg 127)
      * Deterministic channel indexing & 4-quadrant spatial localization pattern
  - Exhaustive 2,033-sum verification table (s = 0..2032)
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

def make_region_with_sum(target_sum: int) -> np.ndarray:
    """Create a 4x4 INT8 region whose sum is exactly target_sum in [0, 2032]."""
    assert 0 <= target_sum <= 2032, f"Target sum {target_sum} out of range [0, 2032]"
    q = target_sum // 16
    r = target_sum % 16
    region = np.full(16, q, dtype=np.int8)
    if r > 0:
        region[:r] = q + 1
    assert int(region.astype(int).sum()) == target_sum
    return region.reshape(4, 4)

def round_even_div16(s: int) -> int:
    """Exact round-to-nearest-even division by 16."""
    div = s >> 4
    rem = s & 0xF
    if rem > 8:
        return div + 1
    elif rem < 8:
        return div
    else:
        return (div + 1) if (div & 1) else div

def main():
    print("=" * 80)
    print("  Generating Spatial Adaptive Average Pooling Golden Test Vectors")
    print("=" * 80)

    ref = IntegerV5Reference()

    # =========================================================================
    # 1. Real Image Forward Propagation
    # =========================================================================
    img_path = PROJECT_DIR / "cleaned_dataset/test/capacitor/Capacitors_aluminum_electrolytic_00007.jpg"
    print(f"Loading real test image: {img_path.name}")
    pil_img = Image.open(img_path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    img_u8 = np.transpose(np.array(pil_img, dtype=np.uint8), (2, 0, 1))[np.newaxis, :]

    # Propagate through Input LUT -> Conv1 -> Conv2 -> Conv3
    x0 = ref.quantize_input_image(img_u8)
    x1 = ref._conv2d_block(x0, ref.w_conv1_mat, ref.b_conv1, int(ref.m[0]), int(ref.s[0]))
    x2 = ref._conv2d_block(x1, ref.w_conv2_mat, ref.b_conv2, int(ref.m[1]), int(ref.s[1]))
    x3 = ref._conv2d_block(x2, ref.w_conv3_mat, ref.b_conv3, int(ref.m[2]), int(ref.s[2]))

    print(f"Propagated Real Conv3 Output x3: shape {x3.shape}, min {x3.min()}, max {x3.max()}")
    assert x3.shape == (1, 64, 8, 8), f"Expected (1, 64, 8, 8), got {x3.shape}"
    assert x3.min() >= 0, f"Expected non-negative activations, got min {x3.min()}"

    real_in = x3[0].copy()  # (64, 8, 8)

    # Compute golden pure average pooling (Banker's round-to-even)
    reshaped_real = real_in.reshape(64, 2, 4, 2, 4).astype(np.int64)
    sums_real = reshaped_real.sum(axis=(2, 4))  # (64, 2, 2)
    div_real = sums_real // 16
    rem_real = sums_real % 16
    golden_real_avg = np.where(rem_real == 8, div_real + (div_real % 2), (sums_real + 8) // 16).astype(np.int8)

    # Compute golden requantized spatial pool (from reference)
    golden_real_requant = ref._spatial_adaptive_pool(x3, rounding="round_even")[0]

    print(f"Real pure avg output:    min {golden_real_avg.min()}, max {golden_real_avg.max()}")
    print(f"Real requantized output: min {golden_real_requant.min()}, max {golden_real_requant.max()}")

    # =========================================================================
    # 2. All-Zero Input Test
    # =========================================================================
    zero_in = np.zeros((64, 8, 8), dtype=np.int8)
    golden_zero_avg = np.zeros((64, 2, 2), dtype=np.int8)
    golden_zero_requant = np.zeros((64, 2, 2), dtype=np.int8)

    # =========================================================================
    # 3. Half-Way Boundary & Neighbor Tests (Sums around rem == 8)
    # =========================================================================
    # 64 channels * 4 quadrants = 256 positions.
    # We test sums s from 0 up to 255 with diverse remainders, specifically including
    # rem == 7, 8, 9 across various quotients.
    boundary_in = np.zeros((64, 8, 8), dtype=np.int8)
    golden_boundary_avg = np.zeros((64, 2, 2), dtype=np.int8)
    golden_boundary_requant = np.zeros((64, 2, 2), dtype=np.int8)

    # Generate test sums around half-way boundaries
    test_sums = []
    for div in range(16):
        test_sums.extend([
            16 * div + 7,   # rem 7 (round down)
            16 * div + 8,   # rem 8 (tie to nearest even)
            16 * div + 9,   # rem 9 (round up)
            16 * div + 0,   # rem 0 (exact)
        ])
    # test_sums has 64 values; replicate/cycle across 256 positions
    pos_idx = 0
    for c in range(64):
        for py in range(2):
            for px in range(2):
                s = test_sums[pos_idx % len(test_sums)]
                boundary_in[c, py*4:(py+1)*4, px*4:(px+1)*4] = make_region_with_sum(s)
                exp_avg = round_even_div16(s)
                golden_boundary_avg[c, py, px] = exp_avg
                # Requantize
                q = int(np.clip(((exp_avg * 103835 + 32768) >> 16), -128, 127))
                golden_boundary_requant[c, py, px] = q
                pos_idx += 1

    # =========================================================================
    # 4. Even vs Odd Quotient Tie-Breaker Suite
    # =========================================================================
    # Specifically test rem == 8 for all even div (0, 2, 4, ...) and odd div (1, 3, 5, ...)
    tie_in = np.zeros((64, 8, 8), dtype=np.int8)
    golden_tie_avg = np.zeros((64, 2, 2), dtype=np.int8)
    golden_tie_requant = np.zeros((64, 2, 2), dtype=np.int8)

    pos_idx = 0
    for c in range(64):
        for py in range(2):
            for px in range(2):
                div = (pos_idx % 32)
                s = div * 16 + 8  # rem == 8 always
                tie_in[c, py*4:(py+1)*4, px*4:(px+1)*4] = make_region_with_sum(s)
                exp_avg = div if (div % 2 == 0) else (div + 1)
                golden_tie_avg[c, py, px] = exp_avg
                q = int(np.clip(((exp_avg * 103835 + 32768) >> 16), -128, 127))
                golden_tie_requant[c, py, px] = q
                pos_idx += 1

    # =========================================================================
    # 5. Maximum Non-Negative INT8 Input (+127)
    # =========================================================================
    max_in = np.full((64, 8, 8), 127, dtype=np.int8)
    golden_max_avg = np.full((64, 2, 2), 127, dtype=np.int8)
    # 127 * 103835 + 32768 = 13187045 >> 16 = 201 -> clipped to 127
    golden_max_requant = np.full((64, 2, 2), 127, dtype=np.int8)

    # =========================================================================
    # 6. Deterministic Channel Indexing and Spatial Quadrant Pattern
    # =========================================================================
    det_in = np.zeros((64, 8, 8), dtype=np.int8)
    golden_det_avg = np.zeros((64, 2, 2), dtype=np.int8)
    golden_det_requant = np.zeros((64, 2, 2), dtype=np.int8)

    for c in range(64):
        for py in range(2):
            for px in range(2):
                # Unique distinct value for each (c, py, px) in [0..120]
                val = ((c * 4 + py * 2 + px) % 120)
                # Assign constant val to all 16 elements in the quadrant
                det_in[c, py*4:(py+1)*4, px*4:(px+1)*4] = val
                golden_det_avg[c, py, px] = val
                q = int(np.clip(((val * 103835 + 32768) >> 16), -128, 127))
                golden_det_requant[c, py, px] = q

    # =========================================================================
    # 7. Exhaustive Verification Table of All 2033 Sums (0 through 2032)
    # =========================================================================
    exhaustive_expected = []
    tie_cases_count = 0
    even_ties = 0
    odd_ties = 0

    for s in range(2033):
        div = s // 16
        rem = s % 16
        exp_even = int(np.round(s / 16.0))
        ref_even = div + (div % 2) if rem == 8 else (s + 8) // 16
        assert exp_even == ref_even, f"Discrepancy at sum {s}: np.round={exp_even}, ref={ref_even}"
        exhaustive_expected.append(exp_even)
        if rem == 8:
            tie_cases_count += 1
            if div % 2 == 0:
                even_ties += 1
            else:
                odd_ties += 1

    print(f"Exhaustive test table prepared: 2033 sums [0, 2032].")
    print(f"Half-way tie cases (rem == 8): total {tie_cases_count} (even div: {even_ties}, odd div: {odd_ties})")

    # =========================================================================
    # Generate test_spatial_pool.cpp
    # =========================================================================
    testbench_path = PROJECT_DIR / "hls_project/testbench/test_spatial_pool.cpp"
    print(f"Writing standalone testbench to: {testbench_path}")

    real_in_str = format_1d_array(real_in.flatten())
    golden_real_avg_str = format_1d_array(golden_real_avg.flatten())
    golden_real_req_str = format_1d_array(golden_real_requant.flatten())

    boundary_in_str = format_1d_array(boundary_in.flatten())
    golden_boundary_avg_str = format_1d_array(golden_boundary_avg.flatten())
    golden_boundary_req_str = format_1d_array(golden_boundary_requant.flatten())

    tie_in_str = format_1d_array(tie_in.flatten())
    golden_tie_avg_str = format_1d_array(golden_tie_avg.flatten())
    golden_tie_req_str = format_1d_array(golden_tie_requant.flatten())

    det_in_str = format_1d_array(det_in.flatten())
    golden_det_avg_str = format_1d_array(golden_det_avg.flatten())
    golden_det_req_str = format_1d_array(golden_det_requant.flatten())

    exhaustive_str = format_1d_array(np.array(exhaustive_expected, dtype=np.int8))

    testbench_content = f"""/**
 * test_spatial_pool.cpp
 * ---------------------
 * Standalone C++ verification testbench for Spatial Adaptive Average Pooling.
 *
 * Compares C++ implementation against golden tensors generated by
 * integer_v5_reference.py on:
 *   1. Real camera test image passed through input LUT, Conv1, Conv2, Conv3
 *      [64][8][8] -> [64][2][2] (256 elements)
 *   2. Zero input boundary test (all-zero [64][8][8] -> all-zero [64][2][2])
 *   3. Constant inputs covering sums around half-way rounding boundaries (rem == 7, 8, 9)
 *   4. Even vs Odd quotient tie-breaker suite (s=8, 24, 40, 56, 72, 88, 104, 120, ...)
 *   5. Maximum non-negative INT8 input values (+127 -> sum 2032 -> avg 127)
 *   6. Deterministic channel indexing & 4-quadrant spatial localization pattern
 *   7. Exhaustive verification of all 2,033 non-negative sums [0, 2032]
 *
 * Verifies both:
 *   - Pure Spatial Average Pooling (spatial_pool): division by 16 with round-even
 *   - Spatial Pooling with Requantization (spatial_pool_with_requant): matching ref._spatial_adaptive_pool
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../src/spatial_pool.h"

// ============================================================================
// EMBEDDED GOLDEN DATA ARRAYS (Generated from integer_v5_reference.py)
// ============================================================================

// 1. Real Image Input [64][8][8] (4096 elements)
static const int8_t GOLDEN_REAL_IN[4096] = {{
{real_in_str}
}};

// 1a. Real Image Golden Pure Average Output [64][2][2] (256 elements)
static const int8_t GOLDEN_REAL_AVG[256] = {{
{golden_real_avg_str}
}};

// 1b. Real Image Golden Requantized Output [64][2][2] (256 elements)
static const int8_t GOLDEN_REAL_REQUANT[256] = {{
{golden_real_req_str}
}};

// 3. Half-Way Boundary Test Input [64][8][8] (4096 elements)
static const int8_t GOLDEN_BOUNDARY_IN[4096] = {{
{boundary_in_str}
}};

static const int8_t GOLDEN_BOUNDARY_AVG[256] = {{
{golden_boundary_avg_str}
}};

static const int8_t GOLDEN_BOUNDARY_REQUANT[256] = {{
{golden_boundary_req_str}
}};

// 4. Even vs Odd Quotient Tie-Breaker Test Input [64][8][8] (4096 elements)
static const int8_t GOLDEN_TIE_IN[4096] = {{
{tie_in_str}
}};

static const int8_t GOLDEN_TIE_AVG[256] = {{
{golden_tie_avg_str}
}};

static const int8_t GOLDEN_TIE_REQUANT[256] = {{
{golden_tie_req_str}
}};

// 6. Deterministic Indexing Pattern Input [64][8][8] (4096 elements)
static const int8_t GOLDEN_DET_IN[4096] = {{
{det_in_str}
}};

static const int8_t GOLDEN_DET_AVG[256] = {{
{golden_det_avg_str}
}};

static const int8_t GOLDEN_DET_REQUANT[256] = {{
{golden_det_req_str}
}};

// 7. Exhaustive Expected Table for all 2033 sums [0, 2032]
static const int8_t EXHAUSTIVE_EXPECTED[2033] = {{
{exhaustive_str}
}};

// ============================================================================
// Verification Helper Function
// ============================================================================
int compare_tensor_256(
    const char* suite_name,
    const int8_t cpp_out[SPATIAL_POOL_OUT_CH][SPATIAL_POOL_OUT_H][SPATIAL_POOL_OUT_W],
    const int8_t golden_out[256]
) {{
    int mismatches = 0;
    int max_abs_diff = 0;

    for (int c = 0; c < SPATIAL_POOL_OUT_CH; ++c) {{
        for (int py = 0; py < SPATIAL_POOL_OUT_H; ++py) {{
            for (int px = 0; px < SPATIAL_POOL_OUT_W; ++px) {{
                int flat_idx = c * 4 + py * 2 + px;
                int8_t cpp_val = cpp_out[c][py][px];
                int8_t gold_val = golden_out[flat_idx];
                int diff = std::abs(static_cast<int>(cpp_val) - static_cast<int>(gold_val));

                if (diff > 0) {{
                    mismatches++;
                    if (diff > max_abs_diff) max_abs_diff = diff;
                    if (mismatches <= 10) {{
                        std::cout << "  [MISMATCH] [" << suite_name << "] c=" << c
                                  << ", pos=(" << py << "," << px << ")"
                                  << " (flat " << flat_idx << "):"
                                  << " C++ = " << static_cast<int>(cpp_val)
                                  << ", Golden = " << static_cast<int>(gold_val)
                                  << " (diff=" << diff << ")\\n";
                    }}
                }}
            }}
        }}
    }}

    std::cout << "  " << std::left << std::setw(42) << suite_name
              << " | Total: 256 | Mismatches: " << mismatches
              << " | MaxDiff: " << max_abs_diff
              << " | Result: " << (mismatches == 0 ? "PASS" : "FAIL")
              << "\\n";

    return mismatches;
}}

int main() {{
    std::cout << "================================================================================" << std::endl;
    std::cout << "  V5-Spatial HLS C++ Verification: Spatial Adaptive Average Pooling (Step 3.6)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    int total_failures = 0;

    // Allocate test buffers
    static int8_t test_input[SPATIAL_POOL_IN_CH][SPATIAL_POOL_IN_H][SPATIAL_POOL_IN_W];
    static int8_t test_output[SPATIAL_POOL_OUT_CH][SPATIAL_POOL_OUT_H][SPATIAL_POOL_OUT_W];

    // ========================================================================
    // TEST SUITE 1: Real Camera Image Propagation Test
    // ========================================================================
    std::cout << "\\n[Test Suite 1/7] Real Image (LUT -> Conv1 -> Conv2 -> Conv3 -> Pool)..." << std::endl;
    std::memcpy(test_input, GOLDEN_REAL_IN, sizeof(test_input));

    // 1a. Pure Average Pooling
    spatial_pool(test_input, test_output);
    total_failures += compare_tensor_256("Suite 1a: Real Image (Pure Avg)", test_output, GOLDEN_REAL_AVG);

    // 1b. Requantized Pooling
    spatial_pool_with_requant(test_input, test_output);
    total_failures += compare_tensor_256("Suite 1b: Real Image (Requantized)", test_output, GOLDEN_REAL_REQUANT);

    // ========================================================================
    // TEST SUITE 2: All-Zero Input Test
    // ========================================================================
    std::cout << "\\n[Test Suite 2/7] All-Zero Input Boundary Test..." << std::endl;
    std::memset(test_input, 0, sizeof(test_input));
    static const int8_t ZERO_GOLDEN[256] = {{0}};

    spatial_pool(test_input, test_output);
    total_failures += compare_tensor_256("Suite 2a: All-Zero (Pure Avg)", test_output, ZERO_GOLDEN);

    spatial_pool_with_requant(test_input, test_output);
    total_failures += compare_tensor_256("Suite 2b: All-Zero (Requantized)", test_output, ZERO_GOLDEN);

    // ========================================================================
    // TEST SUITE 3: Half-Way Boundary & Neighbor Tests (rem == 7, 8, 9)
    // ========================================================================
    std::cout << "\\n[Test Suite 3/7] Half-Way Boundary & Neighbor Tests..." << std::endl;
    std::memcpy(test_input, GOLDEN_BOUNDARY_IN, sizeof(test_input));

    spatial_pool(test_input, test_output);
    total_failures += compare_tensor_256("Suite 3a: Boundary (Pure Avg)", test_output, GOLDEN_BOUNDARY_AVG);

    spatial_pool_with_requant(test_input, test_output);
    total_failures += compare_tensor_256("Suite 3b: Boundary (Requantized)", test_output, GOLDEN_BOUNDARY_REQUANT);

    // ========================================================================
    // TEST SUITE 4: Even vs Odd Quotient Tie-Breaker Suite (rem == 8)
    // ========================================================================
    std::cout << "\\n[Test Suite 4/7] Even vs Odd Quotient Tie-Breaker Tests..." << std::endl;
    std::memcpy(test_input, GOLDEN_TIE_IN, sizeof(test_input));

    spatial_pool(test_input, test_output);
    total_failures += compare_tensor_256("Suite 4a: Tie-Breakers (Pure Avg)", test_output, GOLDEN_TIE_AVG);

    spatial_pool_with_requant(test_input, test_output);
    total_failures += compare_tensor_256("Suite 4b: Tie-Breakers (Requantized)", test_output, GOLDEN_TIE_REQUANT);

    // ========================================================================
    // TEST SUITE 5: Maximum Non-Negative INT8 Input (+127)
    // ========================================================================
    std::cout << "\\n[Test Suite 5/7] Maximum Non-Negative INT8 Input (+127)..." << std::endl;
    std::memset(test_input, 127, sizeof(test_input));
    static int8_t MAX_AVG_GOLDEN[256];
    std::memset(MAX_AVG_GOLDEN, 127, sizeof(MAX_AVG_GOLDEN));

    spatial_pool(test_input, test_output);
    total_failures += compare_tensor_256("Suite 5a: Max +127 (Pure Avg)", test_output, MAX_AVG_GOLDEN);

    spatial_pool_with_requant(test_input, test_output);
    total_failures += compare_tensor_256("Suite 5b: Max +127 (Requantized)", test_output, MAX_AVG_GOLDEN);

    // ========================================================================
    // TEST SUITE 6: Deterministic Channel & Quadrant Indexing Pattern
    // ========================================================================
    std::cout << "\\n[Test Suite 6/7] Deterministic Channel & Quadrant Indexing Pattern..." << std::endl;
    std::memcpy(test_input, GOLDEN_DET_IN, sizeof(test_input));

    spatial_pool(test_input, test_output);
    total_failures += compare_tensor_256("Suite 6a: Indexing (Pure Avg)", test_output, GOLDEN_DET_AVG);

    spatial_pool_with_requant(test_input, test_output);
    total_failures += compare_tensor_256("Suite 6b: Indexing (Requantized)", test_output, GOLDEN_DET_REQUANT);

    // ========================================================================
    // TEST SUITE 7: Exhaustive Verification of All 2033 Sums [0, 2032]
    // ========================================================================
    std::cout << "\\n[Test Suite 7/7] Exhaustive Verification of Rounding Helper over 2,033 Sums..." << std::endl;
    int exhaustive_mismatches = 0;
    int halfup_differences = 0;

    for (int s = 0; s < 2033; ++s) {{
        int8_t cpp_res = spatial_pool_div16_round_even(s);
        int8_t exp_res = EXHAUSTIVE_EXPECTED[s];
        int8_t halfup_res = static_cast<int8_t>((s + 8) / 16);

        if (cpp_res != exp_res) {{
            exhaustive_mismatches++;
            if (exhaustive_mismatches <= 10) {{
                std::cout << "  [MISMATCH] Exhaustive sum=" << s
                          << ": C++=" << static_cast<int>(cpp_res)
                          << ", Expected=" << static_cast<int>(exp_res) << "\\n";
            }}
        }}

        if (cpp_res != halfup_res) {{
            halfup_differences++;
        }}
    }}

    std::cout << "  " << std::left << std::setw(42) << "Suite 7: All 2,033 Non-Negative Sums"
              << " | Total: 2033 | Mismatches: " << exhaustive_mismatches
              << " | Round-Half-Up Diffs: " << halfup_differences
              << " | Result: " << (exhaustive_mismatches == 0 ? "PASS" : "FAIL")
              << "\\n";

    total_failures += exhaustive_mismatches;

    // ========================================================================
    // Final Summary
    // ========================================================================
    std::cout << "\\n================================================================================" << std::endl;
    if (total_failures == 0) {{
        std::cout << "  >>> ALL SPATIAL POOLING C++ VERIFICATION TESTS PASSED (0 MISMATCHES) <<<" << std::endl;
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
