"""
generate_fc2_golden.py
----------------------
Phase 3, Step 3.8 -- Golden Data Generator and C++ Code Generator for FC2 Layer
and 9-Class Argmax Decision Policy.

Generates:
  1. hls_project/src/fc2.cpp (with real exported INT8 weights and INT32 biases)
  2. hls_project/testbench/test_fc2.cpp (with embedded golden test vectors)

Uses integer_v5_reference.py to compute exact bit-for-bit intermediate activations:
  - Real test image (Capacitors_aluminum_electrolytic_00007.jpg) propagated through
    input quantization, Conv1, Conv2, Conv3, spatial pooling, and FC1 to yield
    real FC2 input [64], golden FC2 logits [10], and predicted 9-class index.
  - Synthetic test suites:
      * All-zero input [64] (bias-only logits)
      * Maximum positive input (+127 for all 64 elements)
      * Negative synthetic input (-50 for all 64 elements)
      * Deterministic mixed-value input vector
      * Class 9 largest logit test (must be excluded from 9-class prediction)
      * Tie-breaking test cases among eligible classes 0..8
"""

from pathlib import Path
from PIL import Image
import sys
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from integer_v5_reference import IntegerV5Reference, INPUT_SIZE, PERMITTED_CLASSES, TRANSISTOR_INDEX


def format_1d_array(arr_1d: np.ndarray, indent: str = "    ", per_line: int = 16) -> str:
    lines = []
    for i in range(0, len(arr_1d), per_line):
        chunk = arr_1d[i:i + per_line]
        line = ", ".join(f"{int(v):6d}" for v in chunk)
        lines.append(f"{indent}{line},")
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("  Generating FC2 C++ Implementation & Golden Test Vectors")
    print("=" * 80)

    ref = IntegerV5Reference()
    params_dir = PROJECT_DIR / "fpga_export/v5_spatial/integer_params"

    w_fc2 = np.load(params_dir / "fc2_weight_int8.npy").astype(np.int8)  # (10, 64)
    b_fc2 = np.load(params_dir / "fc2_bias_int32.npy").astype(np.int32)  # (10,)

    print(f"Loaded FC2 Weights: shape {w_fc2.shape}, dtype {w_fc2.dtype}, min {w_fc2.min()}, max {w_fc2.max()}")
    print(f"Loaded FC2 Biases:  shape {b_fc2.shape}, dtype {b_fc2.dtype}, min {b_fc2.min()}, max {b_fc2.max()}")
    print(f"FC2 Biases values: {b_fc2.tolist()}")

    # =========================================================================
    # 1. Generate hls_project/src/fc2.cpp
    # =========================================================================
    fc2_cpp_path = PROJECT_DIR / "hls_project/src/fc2.cpp"
    print(f"Generating C++ source: {fc2_cpp_path}")

    w_lines = []
    for k in range(10):
        w_lines.append(f"    {{ // Class {k}")
        row_str = format_1d_array(w_fc2[k], indent="        ", per_line=16)
        w_lines.append(row_str)
        w_lines.append("    },")
    weights_str = "\n".join(w_lines)

    biases_str = format_1d_array(b_fc2, indent="    ", per_line=10)

    fc2_cpp_content = f"""#include "fc2.h"

// ============================================================================
// Exported Weights: [10][64] signed INT8 (640 elements)
// Source: fpga_export/v5_spatial/integer_params/fc2_weight_int8.npy
// Quantization Scale: 0.005109675287261722
// ============================================================================
const int8_t FC2_WEIGHTS[FC2_OUT_CLASSES][FC2_IN_FEATURES] = {{
{weights_str}
}};

// ============================================================================
// Exported Biases: [10] signed INT32 (10 elements)
// Source: fpga_export/v5_spatial/integer_params/fc2_bias_int32.npy
// Pre-scaled by (s_in * s_w) = 0.0010914325685824372
// ============================================================================
const int32_t FC2_BIASES[FC2_OUT_CLASSES] = {{
{biases_str}
}};

// ============================================================================
// FC2 Layer Forward Pass (Frozen on-chip weights and biases)
// Computes unquantized signed INT32 logits for all 10 classes
// ============================================================================
void fc2(
    const int8_t input[FC2_IN_FEATURES],
    int32_t output[FC2_OUT_CLASSES]
) {{
    for (int k = 0; k < FC2_OUT_CLASSES; ++k) {{
        int32_t acc = FC2_BIASES[k];
        for (int j = 0; j < FC2_IN_FEATURES; ++j) {{
            acc += static_cast<int32_t>(input[j]) * static_cast<int32_t>(FC2_WEIGHTS[k][j]);
        }}
        output[k] = acc;
    }}
}}

// ============================================================================
// Parameterized FC2 Forward Pass (For custom weights/biases testing)
// ============================================================================
void fc2_custom(
    const int8_t input[FC2_IN_FEATURES],
    int32_t output[FC2_OUT_CLASSES],
    const int8_t weights[FC2_OUT_CLASSES][FC2_IN_FEATURES],
    const int32_t biases[FC2_OUT_CLASSES]
) {{
    for (int k = 0; k < FC2_OUT_CLASSES; ++k) {{
        int32_t acc = biases[k];
        for (int j = 0; j < FC2_IN_FEATURES; ++j) {{
            acc += static_cast<int32_t>(input[j]) * static_cast<int32_t>(weights[k][j]);
        }}
        output[k] = acc;
    }}
}}

// ============================================================================
// Nine-Class Argmax Decision Policy
// Finds maximum logit over indices [0..8], strictly excluding index 9.
// Tie-breaking: preserves first occurrence (matching np.argmax).
// ============================================================================
int predict_9class(const int32_t logits[FC2_OUT_CLASSES]) {{
    int best_class = 0;
    int32_t max_logit = logits[0];

    for (int c = 1; c < NUM_DEPLOY_CLASSES; ++c) {{
        if (logits[c] > max_logit) {{
            max_logit = logits[c];
            best_class = c;
        }}
    }}

    return best_class;
}}
"""

    with open(fc2_cpp_path, "w", encoding="utf-8") as f:
        f.write(fc2_cpp_content)
    print(f"Successfully generated {fc2_cpp_path} ({len(fc2_cpp_content)} bytes)")

    # =========================================================================
    # 2. Compute Golden Test Vectors
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
    x4 = ref._spatial_adaptive_pool(x3)
    x5 = ref._fc1_block(x4)

    real_fc2_in = x5[0].astype(np.int8)  # (64,)
    golden_real_logits = ref._fc2_block(x5)[0].astype(np.int32)  # (10,)
    golden_real_pred = int(np.argmax(golden_real_logits[:TRANSISTOR_INDEX]))

    print(f"Real FC2 input: min {real_fc2_in.min()}, max {real_fc2_in.max()}, nonzeros {(real_fc2_in != 0).sum()}/64")
    print(f"Real FC2 logits: {golden_real_logits.tolist()}")
    print(f"Real predicted class: {golden_real_pred} ({PERMITTED_CLASSES[golden_real_pred]})")

    # Suite 2: All-Zero FC1 Input Vector
    zero_in = np.zeros(64, dtype=np.int8)
    golden_zero_logits = ref._fc2_block(zero_in.reshape(1, 64))[0].astype(np.int32)
    golden_zero_pred = int(np.argmax(golden_zero_logits[:TRANSISTOR_INDEX]))
    print(f"Zero FC2 logits: {golden_zero_logits.tolist()}, pred: {golden_zero_pred}")

    # Suite 3: Positive-Extreme FC1 Vector (+127)
    max_in = np.full(64, 127, dtype=np.int8)
    golden_max_logits = ref._fc2_block(max_in.reshape(1, 64))[0].astype(np.int32)
    golden_max_pred = int(np.argmax(golden_max_logits[:TRANSISTOR_INDEX]))
    print(f"Max +127 FC2 logits: {golden_max_logits.tolist()}, pred: {golden_max_pred}")

    # Suite 4: Negative Synthetic FC1 Vector (-50)
    neg_in = np.full(64, -50, dtype=np.int8)
    golden_neg_logits = ref._fc2_block(neg_in.reshape(1, 64))[0].astype(np.int32)
    golden_neg_pred = int(np.argmax(golden_neg_logits[:TRANSISTOR_INDEX]))
    print(f"Negative FC2 logits: {golden_neg_logits.tolist()}, pred: {golden_neg_pred}")

    # Suite 5: Deterministic Mixed-Value Vector
    mixed_in = np.array([(j * 7) % 128 for j in range(64)], dtype=np.int8)
    golden_mixed_logits = ref._fc2_block(mixed_in.reshape(1, 64))[0].astype(np.int32)
    golden_mixed_pred = int(np.argmax(golden_mixed_logits[:TRANSISTOR_INDEX]))
    print(f"Mixed FC2 logits: {golden_mixed_logits.tolist()}, pred: {golden_mixed_pred}")

    # Suite 6: Class 9 Has Largest Logit (MUST Be Excluded)
    # Synthesize input activating class 9 features
    c9_in = np.zeros(64, dtype=np.int8)
    for j in range(64):
        if w_fc2[9, j] > 20:
            c9_in[j] = 100
    golden_c9_logits = ref._fc2_block(c9_in.reshape(1, 64))[0].astype(np.int32)
    golden_c9_pred = int(np.argmax(golden_c9_logits[:TRANSISTOR_INDEX]))
    assert np.argmax(golden_c9_logits) == 9, "Expected class 9 to be global maximum"
    assert golden_c9_pred != 9, "Expected class 9 to be excluded by 9-class policy"
    print(f"Class 9 largest logits: {golden_c9_logits.tolist()}")
    print(f"  Global max index: {int(np.argmax(golden_c9_logits))} (logit={golden_c9_logits[9]})")
    print(f"  9-class policy prediction: {golden_c9_pred} (logit={golden_c9_logits[golden_c9_pred]})")

    # =========================================================================
    # 3. Generate test_fc2.cpp
    # =========================================================================
    testbench_path = PROJECT_DIR / "hls_project/testbench/test_fc2.cpp"
    print(f"\nWriting testbench to: {testbench_path}")

    real_in_str = format_1d_array(real_fc2_in, indent="    ", per_line=16)
    golden_real_logits_str = format_1d_array(golden_real_logits, indent="    ", per_line=10)

    golden_zero_logits_str = format_1d_array(golden_zero_logits, indent="    ", per_line=10)
    max_in_str = format_1d_array(max_in, indent="    ", per_line=16)
    golden_max_logits_str = format_1d_array(golden_max_logits, indent="    ", per_line=10)

    neg_in_str = format_1d_array(neg_in, indent="    ", per_line=16)
    golden_neg_logits_str = format_1d_array(golden_neg_logits, indent="    ", per_line=10)

    mixed_in_str = format_1d_array(mixed_in, indent="    ", per_line=16)
    golden_mixed_logits_str = format_1d_array(golden_mixed_logits, indent="    ", per_line=10)

    c9_in_str = format_1d_array(c9_in, indent="    ", per_line=16)
    golden_c9_logits_str = format_1d_array(golden_c9_logits, indent="    ", per_line=10)

    testbench_content = f"""/**
 * test_fc2.cpp
 * ------------
 * Standalone C++ verification testbench for FC2 Layer (classifier.2)
 * and Nine-Class Argmax Decision Policy.
 *
 * Compares C++ implementation against golden tensors generated by
 * integer_v5_reference.py on:
 *   1. Real camera test image passed through input LUT, Conv1, Conv2, Conv3,
 *      spatial pooling, and FC1 -> FC2 [64] -> [10] logits -> 9-class argmax
 *   2. All-zero input vector [64] (bias-only logits)
 *   3. Positive-extreme input vector (+127 for all 64 inputs)
 *   4. Negative synthetic input vector (-50 for all 64 inputs)
 *   5. Deterministic mixed-value vector
 *   6. Case where Class 9 has the largest logit and MUST still be excluded
 *   7. Deterministic tie-breaking among eligible classes (0..8)
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../src/fc2.h"

// ============================================================================
// EMBEDDED GOLDEN DATA ARRAYS (Generated from integer_v5_reference.py)
// ============================================================================

// 1. Real Image FC1 Output [64]
static const int8_t GOLDEN_REAL_IN[64] = {{
{real_in_str}
}};

// 1. Real Image Golden FC2 Logits [10]
static const int32_t GOLDEN_REAL_LOGITS[10] = {{
{golden_real_logits_str}
}};
static const int GOLDEN_REAL_PRED = {golden_real_pred};

// 2. All-Zero Golden FC2 Logits [10]
static const int32_t GOLDEN_ZERO_LOGITS[10] = {{
{golden_zero_logits_str}
}};
static const int GOLDEN_ZERO_PRED = {golden_zero_pred};

// 3. Positive-Extreme Input [64]
static const int8_t GOLDEN_MAX_IN[64] = {{
{max_in_str}
}};
static const int32_t GOLDEN_MAX_LOGITS[10] = {{
{golden_max_logits_str}
}};
static const int GOLDEN_MAX_PRED = {golden_max_pred};

// 4. Negative Input [64]
static const int8_t GOLDEN_NEG_IN[64] = {{
{neg_in_str}
}};
static const int32_t GOLDEN_NEG_LOGITS[10] = {{
{golden_neg_logits_str}
}};
static const int GOLDEN_NEG_PRED = {golden_neg_pred};

// 5. Mixed Input [64]
static const int8_t GOLDEN_MIXED_IN[64] = {{
{mixed_in_str}
}};
static const int32_t GOLDEN_MIXED_LOGITS[10] = {{
{golden_mixed_logits_str}
}};
static const int GOLDEN_MIXED_PRED = {golden_mixed_pred};

// 6. Class 9 Largest Logit Input [64]
static const int8_t GOLDEN_C9_IN[64] = {{
{c9_in_str}
}};
static const int32_t GOLDEN_C9_LOGITS[10] = {{
{golden_c9_logits_str}
}};
static const int GOLDEN_C9_PRED = {golden_c9_pred};

// ============================================================================
// Verification Helper Function
// ============================================================================
int verify_fc2_and_policy(
    const char* suite_name,
    const int32_t cpp_logits[FC2_OUT_CLASSES],
    const int32_t golden_logits[FC2_OUT_CLASSES],
    int expected_class
) {{
    int logit_mismatches = 0;
    int max_abs_diff = 0;

    for (int k = 0; k < FC2_OUT_CLASSES; ++k) {{
        int32_t cpp_val = cpp_logits[k];
        int32_t gold_val = golden_logits[k];
        int diff = std::abs(cpp_val - gold_val);

        if (diff > 0) {{
            logit_mismatches++;
            if (diff > max_abs_diff) max_abs_diff = diff;
            std::cout << "  [MISMATCH] [" << suite_name << "] Logit k=" << k
                      << ": C++ = " << cpp_val
                      << ", Golden = " << gold_val
                      << " (diff=" << diff << ")\\n";
        }}
    }}

    // Evaluate 9-class policy
    int predicted_class = predict_9class(cpp_logits);
    bool class_ok = (predicted_class == expected_class);

    if (!class_ok) {{
        std::cout << "  [POLICY FAIL] [" << suite_name << "] Predicted class = "
                  << predicted_class << ", Expected = " << expected_class << "\\n";
    }}

    bool pass = (logit_mismatches == 0) && class_ok;

    std::cout << "  " << std::left << std::setw(38) << suite_name
              << " | Logits: 10/10"
              << " | Mismatches: " << logit_mismatches
              << " | MaxDiff: " << max_abs_diff
              << " | Pred: " << predicted_class << " (exp: " << expected_class << ")"
              << " | Result: " << (pass ? "PASS" : "FAIL")
              << "\\n";

    return (pass ? 0 : 1);
}}

int main() {{
    std::cout << "================================================================================" << std::endl;
    std::cout << "  V5-Spatial HLS C++ Verification: FC2 Layer & 9-Class Argmax Policy (Step 3.8)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    int total_failures = 0;

    static int8_t test_input[FC2_IN_FEATURES];
    static int32_t test_output[FC2_OUT_CLASSES];

    // ========================================================================
    // TEST SUITE 1: Real Camera Image Forward Pass
    // ========================================================================
    std::cout << "\\n[Test Suite 1/7] Real Camera Image Forward Pass..." << std::endl;
    std::memcpy(test_input, GOLDEN_REAL_IN, sizeof(test_input));
    fc2(test_input, test_output);
    total_failures += verify_fc2_and_policy(
        "Suite 1: Real Test Image", test_output, GOLDEN_REAL_LOGITS, GOLDEN_REAL_PRED
    );

    // ========================================================================
    // TEST SUITE 2: All-Zero FC1 Input Vector (Bias-Only Logits)
    // ========================================================================
    std::cout << "\\n[Test Suite 2/7] All-Zero FC1 Input (Bias-Only Logits)..." << std::endl;
    std::memset(test_input, 0, sizeof(test_input));
    fc2(test_input, test_output);
    total_failures += verify_fc2_and_policy(
        "Suite 2: All-Zero Input", test_output, GOLDEN_ZERO_LOGITS, GOLDEN_ZERO_PRED
    );

    // ========================================================================
    // TEST SUITE 3: Positive-Extreme Input Vector (+127)
    // ========================================================================
    std::cout << "\\n[Test Suite 3/7] Positive-Extreme Input Vector (+127)..." << std::endl;
    std::memcpy(test_input, GOLDEN_MAX_IN, sizeof(test_input));
    fc2(test_input, test_output);
    total_failures += verify_fc2_and_policy(
        "Suite 3: Max +127 Input", test_output, GOLDEN_MAX_LOGITS, GOLDEN_MAX_PRED
    );

    // ========================================================================
    // TEST SUITE 4: Negative Synthetic Input Vector (-50)
    // ========================================================================
    std::cout << "\\n[Test Suite 4/7] Negative Synthetic Input Vector (-50)..." << std::endl;
    std::memcpy(test_input, GOLDEN_NEG_IN, sizeof(test_input));
    fc2(test_input, test_output);
    total_failures += verify_fc2_and_policy(
        "Suite 4: Negative Synthetic Input", test_output, GOLDEN_NEG_LOGITS, GOLDEN_NEG_PRED
    );

    // ========================================================================
    // TEST SUITE 5: Deterministic Mixed-Value Vector
    // ========================================================================
    std::cout << "\\n[Test Suite 5/7] Deterministic Mixed-Value Vector..." << std::endl;
    std::memcpy(test_input, GOLDEN_MIXED_IN, sizeof(test_input));
    fc2(test_input, test_output);
    total_failures += verify_fc2_and_policy(
        "Suite 5: Mixed Vector", test_output, GOLDEN_MIXED_LOGITS, GOLDEN_MIXED_PRED
    );

    // ========================================================================
    // TEST SUITE 6: Class 9 Has Largest Logit (Transistor Exclusion Verification)
    // ========================================================================
    std::cout << "\\n[Test Suite 6/7] Class 9 Has Largest Logit (Transistor Masking Test)..." << std::endl;
    std::memcpy(test_input, GOLDEN_C9_IN, sizeof(test_input));
    fc2(test_input, test_output);

    // Verify class 9 is indeed highest in full 10-class output
    int global_max_idx = 0;
    int32_t global_max_val = test_output[0];
    for (int k = 1; k < FC2_OUT_CLASSES; ++k) {{
        if (test_output[k] > global_max_val) {{
            global_max_val = test_output[k];
            global_max_idx = k;
        }}
    }}
    std::cout << "  Notice: Global max logit is class " << global_max_idx
              << " (" << global_max_val << "), but 9-class policy MUST choose class "
              << GOLDEN_C9_PRED << " (" << test_output[GOLDEN_C9_PRED] << ")\\n";

    total_failures += verify_fc2_and_policy(
        "Suite 6: Class 9 Masking Exclusion", test_output, GOLDEN_C9_LOGITS, GOLDEN_C9_PRED
    );

    // ========================================================================
    // TEST SUITE 7: Deterministic Tie-Breaking Among Eligible Classes (0..8)
    // ========================================================================
    std::cout << "\\n[Test Suite 7/7] Deterministic Tie-Breaking Unit Tests..." << std::endl;
    int tie_failures = 0;

    struct TieTestCase {{
        int32_t logits[10];
        int expected_pred;
        const char* desc;
    }};

    TieTestCase tie_cases[] = {{
        // Case 7a: Classes 1 and 4 tie at 5000, class 9 is 9999 (should pick class 1)
        {{ {{100, 5000, 200, 300, 5000, 0, -100, 400, 500, 9999}}, 1, "Tie at index 1 and 4 (max 5000), class 9=9999" }},
        // Case 7b: Classes 3, 6, 7 tie at 8000 (should pick class 3)
        {{ {{0, 100, 200, 8000, 500, 600, 8000, 8000, 100, 0}}, 3, "Tie at index 3, 6, 7 (max 8000)" }},
        // Case 7c: Classes 0 and 8 tie at 1200 (should pick class 0)
        {{ {{1200, -10, -20, 500, 100, 200, 300, 400, 1200, -500}}, 0, "Tie at boundary index 0 and 8 (max 1200)" }},
        // Case 7d: All classes 0..8 tie at 0, class 9 is 100 (should pick class 0)
        {{ {{0, 0, 0, 0, 0, 0, 0, 0, 0, 100}}, 0, "All classes 0..8 tie at 0, class 9=100" }},
        // Case 7e: Negative logits, tie between 2 and 5 at -10 (should pick class 2)
        {{ {{-100, -50, -10, -20, -30, -10, -80, -90, -40, -5}}, 2, "Negative logits tie at 2 and 5 (max -10)" }},
    }};

    int num_tie_cases = sizeof(tie_cases) / sizeof(tie_cases[0]);
    for (int i = 0; i < num_tie_cases; ++i) {{
        int pred = predict_9class(tie_cases[i].logits);
        if (pred != tie_cases[i].expected_pred) {{
            tie_failures++;
            std::cout << "  [TIE FAIL] Case " << i << " (" << tie_cases[i].desc << "): "
                      << "got " << pred << ", expected " << tie_cases[i].expected_pred << "\\n";
        }}
    }}

    std::cout << "  " << std::left << std::setw(38) << "Suite 7: Tie-Breaking Unit Tests"
              << " | Cases: " << num_tie_cases
              << " | Failures: " << tie_failures
              << " | Result: " << (tie_failures == 0 ? "PASS" : "FAIL")
              << "\\n";

    total_failures += tie_failures;

    // ========================================================================
    // Final Summary
    // ========================================================================
    std::cout << "\\n================================================================================" << std::endl;
    if (total_failures == 0) {{
        std::cout << "  >>> ALL FC2 & 9-CLASS POLICY C++ TESTS PASSED (0 MISMATCHES) <<<" << std::endl;
    }} else {{
        std::cout << "  >>> VERIFICATION FAILED: " << total_failures << " TOTAL FAILURES <<<" << std::endl;
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
