#ifndef CONV3_H
#define CONV3_H

#include <cstdint>
#include <cstddef>

/**
 * ============================================================================
 * V5-Spatial HLS C++ Implementation: Conv3 Layer
 * ============================================================================
 * Architecture:
 *   - Input tensor:  INT8 CHW [32][16][16]
 *   - Weights:       INT8 [64][32][3][3] (18,432 elements, scale = 0.0046567879)
 *   - Biases:        INT32 [64] (scale = 0.00067621288)
 *   - Convolution:   3x3 kernel, stride 1, zero padding 1
 *   - Bias addition: INT32 pre-scaled bias added to accumulator before ReLU
 *   - Activation:    Integer ReLU (max(acc, 0))
 *   - Requantize:    Fixed-point (M=1465242147, S=40) with round-half-up
 *   - Saturation:    Clamped to INT8 [-128, 127] (post-ReLU [0, 127])
 *   - Max Pooling:   2x2 window, stride 2
 *   - Output tensor: INT8 CHW [64][8][8]
 * ============================================================================
 */

// Dimension constants
constexpr int CONV3_IN_CH = 32;
constexpr int CONV3_IN_H = 16;
constexpr int CONV3_IN_W = 16;

constexpr int CONV3_OUT_CH = 64;
constexpr int CONV3_OUT_H = 8; // Height after 2x2 max pool
constexpr int CONV3_OUT_W = 8; // Width after 2x2 max pool

constexpr int CONV3_CONV_H = 16; // Height before max pool
constexpr int CONV3_CONV_W = 16; // Width before max pool

constexpr int CONV3_KERNEL_SIZE = 3;
constexpr int CONV3_PAD = 1;
constexpr int CONV3_STRIDE = 1;

// Requantization parameters (Primary 31-bit format from hardware manifest)
constexpr int64_t CONV3_M = 1465242147LL;
constexpr int     CONV3_S = 40;

// Exported weights and biases (declared extern; defined in conv3.cpp)
extern const int8_t  CONV3_WEIGHTS[CONV3_OUT_CH][CONV3_IN_CH][CONV3_KERNEL_SIZE][CONV3_KERNEL_SIZE];
extern const int32_t CONV3_BIASES[CONV3_OUT_CH];

/**
 * @brief Conv3 Layer Forward Pass (Frozen on-chip weights and biases)
 *
 * @param input  Input activation tensor [32][16][16] (signed INT8)
 * @param output Output activation tensor [64][8][8] (signed INT8)
 */
void conv3(
    const int8_t input[CONV3_IN_CH][CONV3_IN_H][CONV3_IN_W],
    int8_t output[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W]
);

/**
 * @brief Parameterized Conv3 Layer Forward Pass (For custom weights/biases/params testing)
 *
 * @param input      Input activation tensor [32][16][16]
 * @param output     Output activation tensor [64][8][8]
 * @param weights    Weight tensor [64][32][3][3]
 * @param biases     Bias tensor [64]
 * @param multiplier Fixed-point multiplier M
 * @param shift      Right shift S
 */
void conv3_custom(
    const int8_t input[CONV3_IN_CH][CONV3_IN_H][CONV3_IN_W],
    int8_t output[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W],
    const int8_t weights[CONV3_OUT_CH][CONV3_IN_CH][CONV3_KERNEL_SIZE][CONV3_KERNEL_SIZE],
    const int32_t biases[CONV3_OUT_CH],
    int64_t multiplier,
    int shift
);

/**
 * @brief Scalar fixed-point requantization with round-half-up and saturation
 * Formula: ((acc * M + (1LL << (S - 1))) >> S), clamped to [-128, 127]
 */
int8_t conv3_requantize(int64_t acc, int64_t M, int S);

#endif // CONV3_H
