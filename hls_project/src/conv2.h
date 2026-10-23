#ifndef CONV2_H
#define CONV2_H

#include <cstdint>
#include <cstddef>

/**
 * ============================================================================
 * V5-Spatial HLS C++ Implementation: Conv2 Layer
 * ============================================================================
 * Architecture:
 *   - Input tensor:  INT8 CHW [16][32][32]
 *   - Weights:       INT8 [32][16][3][3] (4,608 elements, scale = 0.0054134346)
 *   - Biases:        INT32 [32] (scale = 0.00014679555)
 *   - Convolution:   3x3 kernel, stride 1, zero padding 1
 *   - Bias addition: INT32 pre-scaled bias added to accumulator before ReLU
 *   - Activation:    Integer ReLU (max(acc, 0))
 *   - Requantize:    Fixed-point (M=1111516105, S=40) with round-half-up
 *   - Saturation:    Clamped to INT8 [-128, 127] (post-ReLU [0, 127])
 *   - Max Pooling:   2x2 window, stride 2
 *   - Output tensor: INT8 CHW [32][16][16]
 * ============================================================================
 */

// Dimension constants
constexpr int CONV2_IN_CH = 16;
constexpr int CONV2_IN_H = 32;
constexpr int CONV2_IN_W = 32;

constexpr int CONV2_OUT_CH = 32;
constexpr int CONV2_OUT_H = 16; // Height after 2x2 max pool
constexpr int CONV2_OUT_W = 16; // Width after 2x2 max pool

constexpr int CONV2_CONV_H = 32; // Height before max pool
constexpr int CONV2_CONV_W = 32; // Width before max pool

constexpr int CONV2_KERNEL_SIZE = 3;
constexpr int CONV2_PAD = 1;
constexpr int CONV2_STRIDE = 1;

// Requantization parameters (Primary 31-bit format from hardware manifest)
constexpr int64_t CONV2_M = 1111516105LL;
constexpr int     CONV2_S = 40;

// Exported weights and biases (declared extern; defined in conv2.cpp)
extern const int8_t  CONV2_WEIGHTS[CONV2_OUT_CH][CONV2_IN_CH][CONV2_KERNEL_SIZE][CONV2_KERNEL_SIZE];
extern const int32_t CONV2_BIASES[CONV2_OUT_CH];

/**
 * @brief Conv2 Layer Forward Pass (Frozen on-chip weights and biases)
 *
 * @param input  Input activation tensor [16][32][32] (signed INT8)
 * @param output Output activation tensor [32][16][16] (signed INT8)
 */
void conv2(
    const int8_t input[CONV2_IN_CH][CONV2_IN_H][CONV2_IN_W],
    int8_t output[CONV2_OUT_CH][CONV2_OUT_H][CONV2_OUT_W]
);

/**
 * @brief Parameterized Conv2 Layer Forward Pass (For custom weights/biases/params testing)
 *
 * @param input      Input activation tensor [16][32][32]
 * @param output     Output activation tensor [32][16][16]
 * @param weights    Weight tensor [32][16][3][3]
 * @param biases     Bias tensor [32]
 * @param multiplier Fixed-point multiplier M
 * @param shift      Right shift S
 */
void conv2_custom(
    const int8_t input[CONV2_IN_CH][CONV2_IN_H][CONV2_IN_W],
    int8_t output[CONV2_OUT_CH][CONV2_OUT_H][CONV2_OUT_W],
    const int8_t weights[CONV2_OUT_CH][CONV2_IN_CH][CONV2_KERNEL_SIZE][CONV2_KERNEL_SIZE],
    const int32_t biases[CONV2_OUT_CH],
    int64_t multiplier,
    int shift
);

/**
 * @brief Scalar fixed-point requantization with round-half-up and saturation
 * Formula: ((acc * M + (1LL << (S - 1))) >> S), clamped to [-128, 127]
 */
int8_t conv2_requantize(int64_t acc, int64_t M, int S);

#endif // CONV2_H
