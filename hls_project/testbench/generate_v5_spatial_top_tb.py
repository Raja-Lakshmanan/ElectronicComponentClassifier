"""
generate_v5_spatial_top_tb.py
-----------------------------
Generates test_v5_spatial_top.cpp from test_component_classifier.cpp,
adapting all tests to invoke the top-level synthesizable wrapper:
    - v5_spatial_top(image_ddr, &logit_0, ..., &logit_9, &pred)  [Direct HW C entry point]
    - v5_spatial_top_array(image_ddr, logits, &pred)             [Convenience array wrapper]
"""

import re
from pathlib import Path

def main():
    tb_dir = Path(__file__).resolve().parent
    src_file = tb_dir / "test_component_classifier.cpp"
    dst_file = tb_dir / "test_v5_spatial_top.cpp"

    with open(src_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update file header comment
    content = content.replace(
        "test_component_classifier.cpp\n * -----------------------------\n * Standalone C++ verification testbench for Top-Level Integrated",
        "test_v5_spatial_top.cpp\n * --------------------\n * Standalone C++ verification testbench for Top-Level Synthesizable Wrapper (v5_spatial_top)"
    )

    # 2. Update include
    content = content.replace(
        '#include "../src/component_classifier.h"',
        '#include "../src/v5_spatial_top.h"'
    )

    # 3. Update main title banner
    content = content.replace(
        "V5-Spatial HLS C++ Top-Level End-to-End Integration Verification (Step 3.9)",
        "V5-Spatial HLS C++ Synthesizable Core Verification (v5_spatial_top - Step 3.13)"
    )

    # Update variable types to int32_t
    content = content.replace("    int pred = 0;", "    int32_t pred = 0;")
    content = content.replace("    int reference_pred = 0;", "    int32_t reference_pred = 0;")

    # 4. Replace function calls:
    # Pattern A: std::memcpy(test_img, IMG, sizeof(test_img)); \n pred = component_classifier(test_img, test_logits);
    pattern_a = re.compile(
        r"std::memcpy\(test_img,\s*([A-Za-z0-9_]+),\s*sizeof\(test_img\)\);\s*\n\s*pred\s*=\s*component_classifier\(test_img,\s*test_logits\);"
    )
    content = pattern_a.sub(r"v5_spatial_top_array(\1, test_logits, &pred);", content)

    # Pattern B: std::memcpy(test_img, IMG, sizeof(test_img)); \n reference_pred = component_classifier(test_img, reference_logits);
    pattern_b = re.compile(
        r"std::memcpy\(test_img,\s*([A-Za-z0-9_]+),\s*sizeof\(test_img\)\);\s*\n\s*reference_pred\s*=\s*component_classifier\(test_img,\s*reference_logits\);"
    )
    content = pattern_b.sub(r"v5_spatial_top_array(\1, reference_logits, &reference_pred);", content)

    # Pattern C: std::memcpy(test_img, IMG, sizeof(test_img)); \n component_classifier(test_img, test_logits);
    pattern_c = re.compile(
        r"std::memcpy\(test_img,\s*([A-Za-z0-9_]+),\s*sizeof\(test_img\)\);\s*\n\s*component_classifier\(test_img,\s*test_logits\);"
    )
    content = pattern_c.sub(r"v5_spatial_top_array(\1, test_logits, &pred);", content)

    # In Suite 2 (synthetic tests), call the direct 12-argument hardware entry point v5_spatial_top
    direct_hw_call_zero = (
        "// 2a. All-zero camera image — direct hardware entry point v5_spatial_top()\n"
        "    v5_spatial_top(\n"
        "        SYNTH_ZERO_IMG,\n"
        "        &test_logits[0], &test_logits[1], &test_logits[2], &test_logits[3], &test_logits[4],\n"
        "        &test_logits[5], &test_logits[6], &test_logits[7], &test_logits[8], &test_logits[9],\n"
        "        &pred\n"
        "    );"
    )
    content = content.replace(
        "// 2a. All-zero camera image\n    v5_spatial_top_array(SYNTH_ZERO_IMG, test_logits, &pred);",
        direct_hw_call_zero
    )

    direct_hw_call_ramp = (
        "// 2b. Deterministic ramp camera image — direct hardware entry point v5_spatial_top()\n"
        "    v5_spatial_top(\n"
        "        SYNTH_RAMP_IMG,\n"
        "        &test_logits[0], &test_logits[1], &test_logits[2], &test_logits[3], &test_logits[4],\n"
        "        &test_logits[5], &test_logits[6], &test_logits[7], &test_logits[8], &test_logits[9],\n"
        "        &pred\n"
        "    );"
    )
    content = content.replace(
        "// 2b. Deterministic ramp camera image\n    v5_spatial_top_array(SYNTH_RAMP_IMG, test_logits, &pred);",
        direct_hw_call_ramp
    )

    with open(dst_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Generated {dst_file} successfully ({len(content):,} bytes).")

if __name__ == "__main__":
    main()
