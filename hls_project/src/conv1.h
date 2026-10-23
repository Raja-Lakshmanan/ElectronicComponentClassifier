#ifndef CONV1_H
#define CONV1_H

#include <cstdint>
#include <cstddef>

/**
 * ============================================================================
 * V5-Spatial HLS C++ Implementation: Conv1 Layer
 * ============================================================================
 * Architecture:
 *   - Input tensor:  INT8 CHW [3][64][64]
 *   - Weights:       INT8 [16][3][3][3] (432 elements, scale = 0.0033164834)
 *   - Biases:        INT32 [16] (scale = 2.611395e-05)
 *   - Convolution:   3x3 kernel, stride 1, zero padding 1
 *   - Bias addition: INT32 pre-scaled bias added to accumulator before ReLU
 *   - Activation:    Integer ReLU (max(acc, 0))
 *   - Requantize:    Fixed-point (M=2117698473, S=41) with round-half-up
 *   - Saturation:    Clamped to INT8 [-128, 127] (post-ReLU [0, 127])
 *   - Max Pooling:   2x2 window, stride 2
 *   - Output tensor: INT8 CHW [16][32][32]
 * ============================================================================
 */

// Dimension constants
constexpr int CONV1_IN_CH = 3;
constexpr int CONV1_IN_H = 64;
constexpr int CONV1_IN_W = 64;

constexpr int CONV1_OUT_CH = 16;
constexpr int CONV1_OUT_H = 32; // Height after 2x2 max pool
constexpr int CONV1_OUT_W = 32; // Width after 2x2 max pool

constexpr int CONV1_CONV_H = 64; // Height before max pool
constexpr int CONV1_CONV_W = 64; // Width before max pool

constexpr int CONV1_KERNEL_SIZE = 3;
constexpr int CONV1_PAD = 1;
constexpr int CONV1_STRIDE = 1;

// Requantization parameters (Primary 31-bit format from hardware manifest)
constexpr int64_t CONV1_M = 2117698473LL;
constexpr int     CONV1_S = 41;

// Exported weights and biases (declared extern; defined in conv1.cpp)
extern const int8_t  CONV1_WEIGHTS[CONV1_OUT_CH][CONV1_IN_CH][CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE];
extern const int32_t CONV1_BIASES[CONV1_OUT_CH];

/**
 * @brief Conv1 Layer Forward Pass (Frozen on-chip weights and biases)
 *
 * @param input  Input activation tensor [3][64][64] (signed INT8)
 * @param output Output activation tensor [16][32][32] (signed INT8)
 */
void conv1(
    const int8_t input[CONV1_IN_CH][CONV1_IN_H][CONV1_IN_W],
    int8_t output[CONV1_OUT_CH][CONV1_OUT_H][CONV1_OUT_W]
);

/**
 * @brief Parameterized Conv1 Layer Forward Pass (For custom weights/biases/params testing)
 *
 * @param input      Input activation tensor [3][64][64]
 * @param output     Output activation tensor [16][32][32]
 * @param weights    Weight tensor [16][3][3][3]
 * @param biases     Bias tensor [16]
 * @param multiplier Fixed-point multiplier M
 * @param shift      Right shift S
 */
void conv1_custom(
    const int8_t input[CONV1_IN_CH][CONV1_IN_H][CONV1_IN_W],
    int8_t output[CONV1_OUT_CH][CONV1_OUT_H][CONV1_OUT_W],
    const int8_t weights[CONV1_OUT_CH][CONV1_IN_CH][CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE],
    const int32_t biases[CONV1_OUT_CH],
    int64_t multiplier,
    int shift
);

/**
 * @brief Scalar fixed-point requantization with round-half-up and saturation
 * Formula: ((acc * M + (1LL << (S - 1))) >> S), clamped to [-128, 127]
 */
int8_t conv1_requantize(int64_t acc, int64_t M, int S);

#endif // CONV1_H
