#include "v5_spatial_top.h"

extern "C" {

/**
 * @brief Top-Level Synthesizable CNN Inference IP Core for PYNQ-Z2
 *
 * Implements the complete 8-stage integer V5-Spatial CNN pipeline with:
 *   - AXI4 Master (m_axi) burst interface for DDR frame ingestion (12,288 bytes, read-only)
 *   - AXI4-Lite (s_axilite) memory-mapped slave interface for control, DDR pointer,
 *     ten explicit 32-bit scalar logit output registers (logit_0 .. logit_9),
 *     and winning 9-class prediction register (pred_class).
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
) {
    // ------------------------------------------------------------------------
    // Vitis HLS Interface Directives
    // ------------------------------------------------------------------------
    // AXI4-Master port for DDR memory burst transfers (depth=12288 bytes, read-only)
    #pragma HLS INTERFACE m_axi port=image_ddr offset=slave bundle=gmem depth=12288 max_read_burst_length=256 num_read_outstanding=4

    // AXI4-Lite slave port for control, memory base address, logits, and prediction
    #pragma HLS INTERFACE s_axilite port=image_ddr  bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_0    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_1    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_2    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_3    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_4    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_5    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_6    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_7    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_8    bundle=control
    #pragma HLS INTERFACE s_axilite port=logit_9    bundle=control
    #pragma HLS INTERFACE s_axilite port=pred_class bundle=control
    #pragma HLS INTERFACE s_axilite port=return     bundle=control

    // ------------------------------------------------------------------------
    // On-Chip Camera Frame Buffer
    // ------------------------------------------------------------------------
    // Planar CHW RGB layout [3][64][64] = 12,288 bytes
    uint8_t local_image[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W];

    // ------------------------------------------------------------------------
    // Step 1: Burst Ingest Planar CHW Image from DDR into On-Chip Memory
    // ------------------------------------------------------------------------
    int idx = 0;
    LOAD_CH: for (int c = 0; c < CLASSIFIER_IN_CH; ++c) {
        LOAD_ROW: for (int h = 0; h < CLASSIFIER_IN_H; ++h) {
            LOAD_COL: for (int w = 0; w < CLASSIFIER_IN_W; ++w) {
                #pragma HLS PIPELINE II=1
                local_image[c][h][w] = image_ddr[idx++];
            }
        }
    }

    // ------------------------------------------------------------------------
    // Step 2: Execute Verified 8-Stage Component Classifier Pipeline
    // ------------------------------------------------------------------------
    int32_t local_logits[CLASSIFIER_NUM_CLASSES];
    int predicted = component_classifier(local_image, local_logits);

    // ------------------------------------------------------------------------
    // Step 3: Export Outputs to AXI-Lite Scalar Registers
    // ------------------------------------------------------------------------
    // Winning 9-class predicted index (transistor class 9 excluded)
    *pred_class = static_cast<int32_t>(predicted);

    // All 10 raw signed 32-bit integer logits (class 9 preserved for diagnostics)
    *logit_0 = local_logits[0];
    *logit_1 = local_logits[1];
    *logit_2 = local_logits[2];
    *logit_3 = local_logits[3];
    *logit_4 = local_logits[4];
    *logit_5 = local_logits[5];
    *logit_6 = local_logits[6];
    *logit_7 = local_logits[7];
    *logit_8 = local_logits[8];
    *logit_9 = local_logits[9];
}

} // extern "C"
