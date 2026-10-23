"""
generate_end_to_end_golden.py
------------------------------
Phase 3, Step 3.9 -- Golden Data Generator and C++ Code Generator for Top-Level
Integrated Electronic Component Classifier (ComponentCNNSpatial).

Generates:
  1. hls_project/src/component_classifier.cpp (with 256-entry input LUT and pipeline glue)
  2. hls_project/testbench/test_component_classifier.cpp (with 12 real camera test images
     covering all 9 deployment classes and synthetic integration tests)

Uses integer_v5_reference.py to compute exact bit-for-bit intermediate and final outputs:
  - 12 real test images from cleaned_dataset/test/
  - All-zero synthetic camera image
  - Deterministic ramp synthetic camera image
  - Repeated execution state invariance check
  - FC2 class-9 dominance exclusion check
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
    print("  Generating Top-Level C++ Classifier & End-to-End Golden Test Vectors")
    print("=" * 80)

    ref = IntegerV5Reference()
    params_dir = PROJECT_DIR / "fpga_export/v5_spatial/integer_params"
    lut = np.load(params_dir / "input_lut_int8.npy").astype(np.int8)  # (256,)
    assert lut.shape == (256,), f"Invalid LUT shape: {lut.shape}"
    print(f"Loaded Input LUT: shape {lut.shape}, min {lut.min()}, max {lut.max()}")

    # =========================================================================
    # 1. Generate hls_project/src/component_classifier.cpp
    # =========================================================================
    cc_cpp_path = PROJECT_DIR / "hls_project/src/component_classifier.cpp"
    print(f"Generating C++ source: {cc_cpp_path}")

    lut_str = format_1d_array(lut, indent="    ", per_line=16)

    cc_cpp_content = f"""#include "component_classifier.h"

// ============================================================================
// Permitted Deployment Class Names (in index order 0..8)
// ============================================================================
const char* const CLASSIFIER_CLASS_NAMES[CLASSIFIER_NUM_PERMITTED] = {{
    "capacitor",
    "diode",
    "fuse",
    "inductor",
    "integrated_circuit",
    "led",
    "potentiometer",
    "push_button",
    "resistor"
}};

// ============================================================================
// Input Quantization Lookup Table (256 entries)
// Source: fpga_export/v5_spatial/integer_params/input_lut_int8.npy
// Formula: clip(round((2.0*P/255.0 - 1.0) / 0.00787402), -128, 127)
// Range: [-127, 127]
// ============================================================================
const int8_t INPUT_LUT[256] = {{
{lut_str}
}};

// ============================================================================
// Input Pixel Quantization Helper
// ============================================================================
void quantize_input_image_lut(
    const uint8_t image_rgb[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W],
    int8_t activations_out[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W]
) {{
    for (int c = 0; c < CLASSIFIER_IN_CH; ++c) {{
        for (int h = 0; h < CLASSIFIER_IN_H; ++h) {{
            for (int w = 0; w < CLASSIFIER_IN_W; ++w) {{
                activations_out[c][h][w] = INPUT_LUT[image_rgb[c][h][w]];
            }}
        }}
    }}
}}

// ============================================================================
// Top-Level Component Classifier Inference Pipeline
// ============================================================================
int component_classifier(
    const uint8_t image_rgb[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W],
    int32_t logits_out[CLASSIFIER_NUM_CLASSES]
) {{
    // Internal activation ping-pong buffers
    // Total buffer footprint: ~41.5 KB (fits safely on stack / in BRAM)
    int8_t x0[CONV1_IN_CH][CONV1_IN_H][CONV1_IN_W];             // [3][64][64] = 12,288 B
    int8_t x1[CONV1_OUT_CH][CONV1_OUT_H][CONV1_OUT_W];          // [16][32][32] = 16,384 B
    int8_t x2[CONV2_OUT_CH][CONV2_OUT_H][CONV2_OUT_W];          // [32][16][16] = 8,192 B
    int8_t x3[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W];          // [64][8][8] = 4,096 B
    int8_t x4[SPATIAL_POOL_OUT_CH][SPATIAL_POOL_OUT_H][SPATIAL_POOL_OUT_W]; // [64][2][2] = 256 B
    int8_t x_flat[FC1_IN_FEATURES];                             // [256] = 256 B
    int8_t x5[FC1_OUT_FEATURES];                                // [64] = 64 B

    // Stage 1: Input LUT Quantization: uint8 [3][64][64] -> int8 [3][64][64]
    quantize_input_image_lut(image_rgb, x0);

    // Stage 2: Conv Block 1: Conv2d + Bias + ReLU + Requant + MaxPool2d (64x64 -> 32x32)
    conv1(x0, x1);

    // Stage 3: Conv Block 2: Conv2d + Bias + ReLU + Requant + MaxPool2d (32x32 -> 16x16)
    conv2(x1, x2);

    // Stage 4: Conv Block 3: Conv2d + Bias + ReLU + Requant + MaxPool2d (16x16 -> 8x8)
    conv3(x2, x3);

    // Stage 5: Spatial Adaptive Pool: 4x4 AvgPool + Requantization to FC1 Scale (8x8 -> 2x2)
    // Matches integer_v5_reference.py _spatial_adaptive_pool() exactly
    spatial_pool_with_requant(x3, x4);

    // Stage 6: Quadrant Flattening (C-contiguous order: c * 4 + py * 2 + px)
    for (int c = 0; c < SPATIAL_POOL_OUT_CH; ++c) {{
        for (int py = 0; py < SPATIAL_POOL_OUT_H; ++py) {{
            for (int px = 0; px < SPATIAL_POOL_OUT_W; ++px) {{
                x_flat[c * 4 + py * 2 + px] = x4[c][py][px];
            }}
        }}
    }}

    // Stage 7: FC1 Layer: Matrix MAC + Bias + ReLU + Requantization (256 -> 64)
    fc1(x_flat, x5);

    // Stage 8: FC2 Layer: Matrix MAC + Bias -> Raw Signed INT32 Logits (64 -> 10)
    fc2(x5, logits_out);

    // Stage 9: Nine-Class Argmax Decision Policy (Excluding Class 9 Transistor)
    int predicted_class = predict_9class(logits_out);

    return predicted_class;
}}
"""

    with open(cc_cpp_path, "w", encoding="utf-8") as f:
        f.write(cc_cpp_content)
    print(f"Successfully generated {cc_cpp_path} ({len(cc_cpp_content)} bytes)")

    # =========================================================================
    # 2. Select 12 Deterministic Real Test Images Covering All 9 Classes
    # =========================================================================
    test_dir = PROJECT_DIR / "cleaned_dataset/test"
    all_classes = [
        "capacitor", "diode", "fuse", "inductor", "integrated_circuit",
        "led", "potentiometer", "push_button", "resistor"
    ]

    selected_images = []
    # 0: Established test image
    est_img = test_dir / "capacitor/Capacitors_aluminum_electrolytic_00007.jpg"
    selected_images.append(("capacitor", est_img.name, est_img))

    # 1..9: First image of each class (avoiding duplicate names)
    for cls in all_classes:
        imgs = sorted(list((test_dir / cls).glob("*.jpg")) + list((test_dir / cls).glob("*.png")))
        for img in imgs:
            if img.name != est_img.name and img.name not in [s[1] for s in selected_images]:
                selected_images.append((cls, img.name, img))
                break

    # 10..11: Additional distinct images to bring total to 12
    extra_diode = sorted(list((test_dir / "diode").glob("*.jpg")))[1]
    extra_ic = sorted(list((test_dir / "integrated_circuit").glob("*.jpg")))[1]
    selected_images.append(("diode", extra_diode.name, extra_diode))
    selected_images.append(("integrated_circuit", extra_ic.name, extra_ic))

    print(f"\nSelected {len(selected_images)} real test images across all {len(all_classes)} permitted classes:")
    for idx, (cls, name, path) in enumerate(selected_images):
        print(f"  [{idx:2d}] Class: {cls:<20} | File: {name}")

    # =========================================================================
    # 3. Compute Golden Data for Real Images
    # =========================================================================
    real_test_cases = []
    for idx, (cls_true, name, path) in enumerate(selected_images):
        pil_img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
        u8 = np.transpose(np.array(pil_img, dtype=np.uint8), (2, 0, 1))  # (3, 64, 64)
        u8_batch = u8[np.newaxis, :]

        logits, preds = ref.forward_integer(u8_batch)
        pred_idx = int(preds[0])
        pred_name = PERMITTED_CLASSES[pred_idx]
        logits_arr = logits[0].astype(np.int32)

        real_test_cases.append({
            "idx": idx,
            "true_class": cls_true,
            "filename": name,
            "input_u8": u8,  # (3, 64, 64)
            "logits": logits_arr,  # (10,)
            "pred_idx": pred_idx,
            "pred_name": pred_name,
        })
        print(f"  Processed [{idx:2d}]: {name:<45} -> True: {cls_true:<18} | Pred: {pred_name:<18} | Logits: {logits_arr.tolist()}")

    # =========================================================================
    # 4. Compute Synthetic Integration Test Cases
    # =========================================================================
    # 4a: All-zero synthetic camera image
    zero_u8 = np.zeros((3, 64, 64), dtype=np.uint8)
    logits_zero, preds_zero = ref.forward_integer(zero_u8[np.newaxis, :])
    golden_zero_logits = logits_zero[0].astype(np.int32)
    golden_zero_pred = int(preds_zero[0])

    # 4b: Deterministic ramp synthetic camera image
    ramp_u8 = np.array([[(c * 50 + y * 2 + x) % 256 for x in range(64)] for y in range(64) for c in range(3)], dtype=np.uint8).reshape(3, 64, 64)
    logits_ramp, preds_ramp = ref.forward_integer(ramp_u8[np.newaxis, :])
    golden_ramp_logits = logits_ramp[0].astype(np.int32)
    golden_ramp_pred = int(preds_ramp[0])

    print(f"\nSynthetic test cases:")
    print(f"  All-zero image -> logits: {golden_zero_logits.tolist()}, pred: {golden_zero_pred} ({PERMITTED_CLASSES[golden_zero_pred]})")
    print(f"  Ramp image     -> logits: {golden_ramp_logits.tolist()}, pred: {golden_ramp_pred} ({PERMITTED_CLASSES[golden_ramp_pred]})")

    # =========================================================================
    # 5. Generate test_component_classifier.cpp
    # =========================================================================
    testbench_path = PROJECT_DIR / "hls_project/testbench/test_component_classifier.cpp"
    print(f"\nWriting testbench to: {testbench_path}")

    # Build image data blocks
    image_declarations = []
    image_test_calls = []

    for tc in real_test_cases:
        idx = tc["idx"]
        flat_u8 = tc["input_u8"].flatten()
        u8_str = format_1d_array(flat_u8, indent="    ", per_line=16)
        logits_str = format_1d_array(tc["logits"], indent="    ", per_line=10)

        image_declarations.append(f"""// Case {idx}: {tc['filename']} (True class: {tc['true_class']})
static const uint8_t REAL_IMG_{idx}[12288] = {{
{u8_str}
}};
static const int32_t REAL_LOGITS_{idx}[10] = {{
{logits_str}
}};
static const int REAL_PRED_{idx} = {tc['pred_idx']};
""")

        image_test_calls.append(f"""    // Test Image {idx}: {tc['filename']}
    std::memcpy(test_img, REAL_IMG_{idx}, sizeof(test_img));
    pred = component_classifier(test_img, test_logits);
    total_failures += verify_image_prediction(
        "Image {idx}: {tc['filename']}",
        "{tc['true_class']}",
        test_logits,
        REAL_LOGITS_{idx},
        pred,
        REAL_PRED_{idx}
    );
""")

    image_decls_str = "\n".join(image_declarations)
    image_calls_str = "\n".join(image_test_calls)

    zero_u8_str = format_1d_array(zero_u8.flatten(), indent="    ", per_line=16)
    zero_logits_str = format_1d_array(golden_zero_logits, indent="    ", per_line=10)

    ramp_u8_str = format_1d_array(ramp_u8.flatten(), indent="    ", per_line=16)
    ramp_logits_str = format_1d_array(golden_ramp_logits, indent="    ", per_line=10)

    testbench_content = f"""/**
 * test_component_classifier.cpp
 * -----------------------------
 * Standalone C++ verification testbench for Top-Level Integrated
 * Electronic Component Classifier (ComponentCNNSpatial).
 *
 * Compares the full 8-stage C++ implementation against golden tensors generated by
 * integer_v5_reference.py on:
 *   1. Twelve real camera test images covering all 9 deployment classes
 *   2. Synthetic all-zero camera image
 *   3. Synthetic deterministic ramp camera image
 *   4. Sequential repeated execution test (checking for uninitialized/residual state)
 *   5. FC2 class-9 dominant exclusion check
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../src/component_classifier.h"

// ============================================================================
// EMBEDDED GOLDEN DATA FOR 12 REAL TEST IMAGES
// ============================================================================
{image_decls_str}

// ============================================================================
// SYNTHETIC TEST CASES
// ============================================================================
static const uint8_t SYNTH_ZERO_IMG[12288] = {{
{zero_u8_str}
}};
static const int32_t SYNTH_ZERO_LOGITS[10] = {{
{zero_logits_str}
}};
static const int SYNTH_ZERO_PRED = {golden_zero_pred};

static const uint8_t SYNTH_RAMP_IMG[12288] = {{
{ramp_u8_str}
}};
static const int32_t SYNTH_RAMP_LOGITS[10] = {{
{ramp_logits_str}
}};
static const int SYNTH_RAMP_PRED = {golden_ramp_pred};

// ============================================================================
// Verification Helper Function
// ============================================================================
int verify_image_prediction(
    const char* label,
    const char* true_class,
    const int32_t cpp_logits[CLASSIFIER_NUM_CLASSES],
    const int32_t golden_logits[CLASSIFIER_NUM_CLASSES],
    int cpp_pred,
    int golden_pred
) {{
    int logit_mismatches = 0;
    int max_abs_diff = 0;

    for (int k = 0; k < CLASSIFIER_NUM_CLASSES; ++k) {{
        int32_t diff = std::abs(cpp_logits[k] - golden_logits[k]);
        if (diff > 0) {{
            logit_mismatches++;
            if (diff > max_abs_diff) max_abs_diff = diff;
            std::cout << "    [MISMATCH] Logit k=" << k
                      << ": C++ = " << cpp_logits[k]
                      << ", Golden = " << golden_logits[k]
                      << " (diff=" << diff << ")\\n";
        }}
    }}

    bool pred_match = (cpp_pred == golden_pred);
    if (!pred_match) {{
        std::cout << "    [PRED MISMATCH] C++ pred = " << cpp_pred
                  << " (" << CLASSIFIER_CLASS_NAMES[cpp_pred] << "), "
                  << "Golden pred = " << golden_pred
                  << " (" << CLASSIFIER_CLASS_NAMES[golden_pred] << ")\\n";
    }}

    bool pass = (logit_mismatches == 0) && pred_match;

    std::cout << "  " << std::left << std::setw(44) << label
              << " | True: " << std::left << std::setw(18) << true_class
              << " | Pred: " << std::left << std::setw(18) << CLASSIFIER_CLASS_NAMES[cpp_pred]
              << " | LogitDiffs: " << logit_mismatches
              << " | MaxDiff: " << max_abs_diff
              << " | " << (pass ? "PASS" : "FAIL")
              << "\\n";

    return (pass ? 0 : 1);
}}

int main() {{
    std::cout << "==================================================================================" << std::endl;
    std::cout << "  V5-Spatial HLS C++ Top-Level End-to-End Integration Verification (Step 3.9)" << std::endl;
    std::cout << "==================================================================================" << std::endl;

    int total_failures = 0;
    static uint8_t test_img[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W];
    static int32_t test_logits[CLASSIFIER_NUM_CLASSES];
    int pred = 0;

    // ========================================================================
    // TEST SUITE 1: 12 Real Test Images Across All 9 Deployment Classes
    // ========================================================================
    std::cout << "\\n[Test Suite 1/4] Real Camera Images Across All 9 Deployment Classes..." << std::endl;

{image_calls_str}

    // ========================================================================
    // TEST SUITE 2: Synthetic Integration Tests (Zero & Ramp Camera Images)
    // ========================================================================
    std::cout << "\\n[Test Suite 2/4] Synthetic Integration Tests..." << std::endl;

    // 2a. All-zero camera image
    std::memcpy(test_img, SYNTH_ZERO_IMG, sizeof(test_img));
    pred = component_classifier(test_img, test_logits);
    total_failures += verify_image_prediction(
        "Synth Image 1: All-Zero Camera Frame",
        "n/a",
        test_logits,
        SYNTH_ZERO_LOGITS,
        pred,
        SYNTH_ZERO_PRED
    );

    // 2b. Deterministic ramp camera image
    std::memcpy(test_img, SYNTH_RAMP_IMG, sizeof(test_img));
    pred = component_classifier(test_img, test_logits);
    total_failures += verify_image_prediction(
        "Synth Image 2: Deterministic Ramp Frame",
        "n/a",
        test_logits,
        SYNTH_RAMP_LOGITS,
        pred,
        SYNTH_RAMP_PRED
    );

    // ========================================================================
    // TEST SUITE 3: Sequential Repeated Execution (Checking State Invariance)
    // ========================================================================
    std::cout << "\\n[Test Suite 3/4] Sequential Repeated Calls (State Invariance)..." << std::endl;
    int state_failures = 0;
    static int32_t reference_logits[CLASSIFIER_NUM_CLASSES];
    int reference_pred = 0;

    // First call with Image 0
    std::memcpy(test_img, REAL_IMG_0, sizeof(test_img));
    reference_pred = component_classifier(test_img, reference_logits);

    // Interleave with different images
    std::memcpy(test_img, REAL_IMG_1, sizeof(test_img));
    component_classifier(test_img, test_logits);

    std::memcpy(test_img, SYNTH_RAMP_IMG, sizeof(test_img));
    component_classifier(test_img, test_logits);

    // Re-run Image 0 and verify exact bit-for-bit identity
    std::memcpy(test_img, REAL_IMG_0, sizeof(test_img));
    pred = component_classifier(test_img, test_logits);

    int repeat_diffs = 0;
    for (int k = 0; k < CLASSIFIER_NUM_CLASSES; ++k) {{
        if (test_logits[k] != reference_logits[k]) repeat_diffs++;
    }}
    if (pred != reference_pred || repeat_diffs > 0) {{
        state_failures++;
        std::cout << "  [STATE FAIL] Repeated call produced different logits or prediction!\\n";
    }} else {{
        std::cout << "  Suite 3: Repeated Calls Invariance | 3 interleaved calls | LogitDiffs: 0 | Result: PASS\\n";
    }}
    total_failures += state_failures;

    // ========================================================================
    // TEST SUITE 4: Class 9 Dominant Logit Masking Verification
    // ========================================================================
    std::cout << "\\n[Test Suite 4/4] Transistor (Class 9) Dominance Masking Policy..." << std::endl;
    // Direct verification that predict_9class strictly ignores dominant class 9
    int32_t dummy_logits[10] = {{100, 200, 300, 400, 500, 600, 700, 800, 900, 99999}};
    int masked_pred = predict_9class(dummy_logits);
    int c9_failures = 0;
    if (masked_pred != 8) {{
        c9_failures++;
        std::cout << "  [MASK FAIL] Expected class 8, got " << masked_pred << "\\n";
    }} else {{
        std::cout << "  Suite 4: Class 9 Dominance (logit=99999) -> Pred: 8 (exp: 8) | Result: PASS\\n";
    }}
    total_failures += c9_failures;

    // ========================================================================
    // Final Summary
    // ========================================================================
    std::cout << "\\n==================================================================================" << std::endl;
    if (total_failures == 0) {{
        std::cout << "  >>> ALL TOP-LEVEL INTEGRATION TESTS PASSED (0 MISMATCHES, 100% BIT-EXACT) <<<" << std::endl;
    }} else {{
        std::cout << "  >>> TOP-LEVEL INTEGRATION FAILED: " << total_failures << " TEST FAILURES <<<" << std::endl;
    }}
    std::cout << "==================================================================================" << std::endl;

    return (total_failures == 0) ? 0 : 1;
}}
"""

    with open(testbench_path, "w", encoding="utf-8") as f:
        f.write(testbench_content)

    print(f"Successfully generated {testbench_path} ({len(testbench_content)} bytes)")
    print("Done.")


if __name__ == "__main__":
    main()
