#ifndef V5_SPATIAL_TOP_H
#define V5_SPATIAL_TOP_H

#include <cstdint>
#include <cstddef>
#include "component_classifier.h"

// Top-level input image dimensions and total bytes in planar CHW order
constexpr int V5_SPATIAL_TOP_IN_CH    = CLASSIFIER_IN_CH; // 3
constexpr int V5_SPATIAL_TOP_IN_H     = CLASSIFIER_IN_H;  // 64
constexpr int V5_SPATIAL_TOP_IN_W     = CLASSIFIER_IN_W;  // 64
constexpr int V5_SPATIAL_TOP_IN_BYTES = V5_SPATIAL_TOP_IN_CH * V5_SPATIAL_TOP_IN_H * V5_SPATIAL_TOP_IN_W; // 12,288 bytes

constexpr int V5_SPATIAL_TOP_NUM_CLASSES   = CLASSIFIER_NUM_CLASSES;   // 10 raw logits
constexpr int V5_SPATIAL_TOP_NUM_PERMITTED = CLASSIFIER_NUM_PERMITTED; // 9 deployment classes (0..8)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Top-Level Synthesizable CNN Inference IP Core for PYNQ-Z2
 *
 * Implements the complete 8-stage integer V5-Spatial CNN pipeline.
 *
 * Hardware Interface Semantics:
 *   - image_ddr:   Input Pointer argument mapped to AXI4 Master (m_axi) with base address
 *                  register in s_axilite. Software writes the 32-bit physical DDR address
 *                  of the 12,288-byte camera image to the IMAGE_DDR_DATA register.
 *   - logit_0..9:  Output Pointer arguments mapped ONLY to s_axilite (NOT m_axi).
 *                  Because these arguments have no m_axi channel and are written by hardware,
 *                  Vitis HLS synthesizes them as dedicated 32-bit READ-ONLY DATA REGISTERS
 *                  in the s_axilite adapter. Software does NOT supply pointer addresses;
 *                  software reads the 32-bit signed integer result values directly from
 *                  these registers once AP_DONE is asserted.
 *   - pred_class:  Output Pointer argument mapped ONLY to s_axilite.
 *                  Synthesized as a dedicated 32-bit READ-ONLY DATA REGISTER holding the
 *                  winning class index in range [0, 8]. Software reads the predicted class
 *                  directly from this register.
 *   - return:      Mapped to s_axilite, generating standard block control registers
 *                  (AP_START, AP_DONE, AP_IDLE, AP_READY, GIE, IER, ISR).
 *
 * Execution flow:
 *   1. Burst reads 12,288 bytes from DDR (image_ddr) into on-chip BRAM in planar CHW order.
 *   2. Converts uint8 pixels [0..255] to signed int8 [-127..127] via 256-entry INPUT_LUT.
 *   3. Evaluates Conv1 (3x3, 16ch, ReLU, Requant, MaxPool 64x64 -> 32x32).
 *   4. Evaluates Conv2 (3x3, 32ch, ReLU, Requant, MaxPool 32x32 -> 16x16).
 *   5. Evaluates Conv3 (3x3, 64ch, ReLU, Requant, MaxPool 16x16 -> 8x8).
 *   6. Evaluates Spatial Adaptive Pooling (4x4 regions, Banker's round-even, Requant -> 64x2x2).
 *   7. Flattens quadrant map into 256-element vector in C-contiguous order.
 *   8. Evaluates FC1 (256 -> 64, ReLU, Requant).
 *   9. Evaluates FC2 (64 -> 10, unquantized signed int32 logits).
 *  10. Computes 9-class argmax over indices [0..8], strictly masking transistor at index 9.
 *  11. Exports predicted class and all 10 raw logits to dedicated AXI-Lite scalar registers.
 *
 * @param image_ddr   Physical pointer in DDR memory to planar CHW RGB uint8 frame (12,288 bytes)
 * @param logit_0..9  Pointers to 10 signed 32-bit integer logits mapped to AXI-Lite result registers
 * @param pred_class  Pointer to winning class index in range [0, 8] mapped to AXI-Lite result register
 */
void v5_spatial_top(
    const uint8_t *image_ddr,
    int32_t       *logit_0,
    int32_t       *logit_1,
    int32_t       *logit_2,
    int32_t       *logit_3,
    int32_t       *logit_4,
    int32_t       *logit_5,
    int32_t       *logit_6,
    int32_t       *logit_7,
    int32_t       *logit_8,
    int32_t       *logit_9,
    int32_t       *pred_class
);

#ifdef __cplusplus
}
#endif

/**
 * @brief Convenience host-side wrapper accepting a 10-element logit array.
 * Maps array elements to the 10 scalar pointer arguments of v5_spatial_top.
 */
inline void v5_spatial_top_array(
    const uint8_t *image_ddr,
    int32_t        logits_out[V5_SPATIAL_TOP_NUM_CLASSES],
    int32_t       *pred_class
) {
    v5_spatial_top(
        image_ddr,
        &logits_out[0], &logits_out[1], &logits_out[2], &logits_out[3], &logits_out[4],
        &logits_out[5], &logits_out[6], &logits_out[7], &logits_out[8], &logits_out[9],
        pred_class
    );
}

#endif // V5_SPATIAL_TOP_H
