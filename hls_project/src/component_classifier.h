#ifndef COMPONENT_CLASSIFIER_H
#define COMPONENT_CLASSIFIER_H

#include <cstdint>
#include <cstddef>

#include "conv1.h"
#include "conv2.h"
#include "conv3.h"
#include "spatial_pool.h"
#include "fc1.h"
#include "fc2.h"

// Top-level input image dimensions (Planar CHW RGB format: [3][64][64])
constexpr int CLASSIFIER_IN_CH = 3;
constexpr int CLASSIFIER_IN_H  = 64;
constexpr int CLASSIFIER_IN_W  = 64;

// Output definitions
constexpr int CLASSIFIER_NUM_CLASSES   = 10; // Total raw logit outputs
constexpr int CLASSIFIER_NUM_PERMITTED = 9;  // Permitted deployment classes (0..8)

// Permitted deployment class names in order 0..8 (Host/testbench only)
#ifndef __SYNTHESIS__
extern const char* const CLASSIFIER_CLASS_NAMES[CLASSIFIER_NUM_PERMITTED];
#endif

// 256-entry Input LUT mapping uint8 camera pixel [0..255] to signed INT8 activation [-127..127]
extern const int8_t INPUT_LUT[256];

/**
 * @brief Top-Level Electronic Component Classifier Inference Wrapper
 *
 * Connects the 8 pipelined stages:
 *   Stage 1: Input LUT Quantization:     uint8_t [3][64][64] -> int8_t [3][64][64]
 *   Stage 2: Conv Block 1:              Conv2d(3->16) + Bias + ReLU + Requant + MaxPool2d -> int8_t [16][32][32]
 *   Stage 3: Conv Block 2:              Conv2d(16->32) + Bias + ReLU + Requant + MaxPool2d -> int8_t [32][16][16]
 *   Stage 4: Conv Block 3:              Conv2d(32->64) + Bias + ReLU + Requant + MaxPool2d -> int8_t [64][8][8]
 *   Stage 5: Spatial Adaptive Pool:     4x4 AvgPool + Requantization -> int8_t [64][2][2]
 *   Stage 6: Quadrant Flattening:       C-contiguous order -> int8_t [256]
 *   Stage 7: FC1 Layer:                 Linear(256->64) + Bias + ReLU + Requant -> int8_t [64]
 *   Stage 8: FC2 Layer:                 Linear(64->10) + Bias -> int32_t [10] (Raw Logits)
 *   Stage 9: Argmax Decision Policy:    predict_9class(logits) -> class index in [0, 8]
 *
 * @param image_rgb  Planar CHW camera image [3][64][64] in uint8 [0..255] (R=0, G=1, B=2)
 * @param logits_out Output array [10] for raw signed 32-bit integer logits (all 10 classes preserved)
 * @return int Predicted class index in [0, 8] (transistor at index 9 strictly excluded)
 */
int component_classifier(
    const uint8_t image_rgb[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W],
    int32_t logits_out[CLASSIFIER_NUM_CLASSES]
);

/**
 * @brief Input Pixel Quantization Helper
 * Applies the 256-entry lookup table to convert raw uint8 pixels to signed INT8 activations.
 */
void quantize_input_image_lut(
    const uint8_t image_rgb[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W],
    int8_t activations_out[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W]
);

#endif // COMPONENT_CLASSIFIER_H
