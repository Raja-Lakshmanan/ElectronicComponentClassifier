#ifndef FC1_H
#define FC1_H

#include <cstdint>
#include <cstddef>

/**
 * ============================================================================
 * V5-Spatial HLS C++ Implementation: FC1 Layer (classifier.0)
 * ============================================================================
 * Architecture:
 *   - Input tensor:  Flattened INT8 vector [256] (scale = 0.32026483)
 *   - Weights:       INT8 [64][256] (16,384 elements, scale = 0.0037154248)
 *   - Biases:        INT32 [64] (scale = 0.0011899198, range [-86, 125])
 *   - Matrix MAC:    64 neurons, 256 multiply-accumulates each
 *   - Accumulator:   int32_t (safe width, max theoretical sum ~4.13M)
 *   - Bias addition: INT32 pre-scaled bias added before ReLU
 *   - Activation:    Integer ReLU (max(acc, 0))
 *   - Requantize:    Fixed-point (M=1531277679, S=38) with round-half-up
 *   - Saturation:    Clamped to INT8 [-128, 127] (post-ReLU [0, 127])
 *   - Output tensor: INT8 vector [64] (scale = 0.21360116)
 * ============================================================================
 */

// Dimension constants
constexpr int FC1_IN_FEATURES  = 256;
constexpr int FC1_OUT_FEATURES = 64;

// Requantization parameters (Primary 31-bit format from hardware manifest)
constexpr int64_t FC1_M = 1531277679LL;
constexpr int     FC1_S = 38;

// Exported weights and biases (declared extern; defined in fc1.cpp)
extern const int8_t  FC1_WEIGHTS[FC1_OUT_FEATURES][FC1_IN_FEATURES];
extern const int32_t FC1_BIASES[FC1_OUT_FEATURES];

/**
 * @brief FC1 Layer Forward Pass (Frozen on-chip weights and biases)
 *
 * Computes:
 *   acc[k] = bias[k] + sum_{j=0}^{255} (input[j] * weight[k][j])
 *   acc_relu = max(acc[k], 0)
 *   q_out = clip(((acc_relu * M + (1LL << (S - 1))) >> S), -128, 127)
 *
 * @param input  Flattened input activation vector [256] (signed INT8)
 * @param output Output activation vector [64] (signed INT8)
 */
void fc1(
    const int8_t input[FC1_IN_FEATURES],
    int8_t output[FC1_OUT_FEATURES]
);

/**
 * @brief Parameterized FC1 Layer Forward Pass (For custom weights/biases/params testing)
 *
 * @param input      Flattened input activation vector [256]
 * @param output     Output activation vector [64]
 * @param weights    Weight matrix [64][256]
 * @param biases     Bias vector [64]
 * @param multiplier Fixed-point multiplier M
 * @param shift      Right shift S
 */
void fc1_custom(
    const int8_t input[FC1_IN_FEATURES],
    int8_t output[FC1_OUT_FEATURES],
    const int8_t weights[FC1_OUT_FEATURES][FC1_IN_FEATURES],
    const int32_t biases[FC1_OUT_FEATURES],
    int64_t multiplier,
    int shift
);

/**
 * @brief Scalar fixed-point requantization with round-half-up and saturation
 * Formula: ((acc * M + (1LL << (S - 1))) >> S), clamped to [-128, 127]
 */
int8_t fc1_requantize(int64_t acc, int64_t M, int S);

#endif // FC1_H
