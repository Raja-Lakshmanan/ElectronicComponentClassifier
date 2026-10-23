#ifndef FC2_H
#define FC2_H

#include <cstdint>
#include <cstddef>

/**
 * ============================================================================
 * V5-Spatial HLS C++ Implementation: FC2 Layer (classifier.2) & 9-Class Policy
 * ============================================================================
 * Architecture:
 *   - Input tensor:  INT8 vector [64] from FC1 (scale = 0.21360116)
 *   - Weights:       INT8 [10][64] (640 elements, scale = 0.0051096753)
 *   - Biases:        INT32 [10] (scale = 0.0010914326, range [-190, 76])
 *   - Matrix MAC:    10 output neurons, 64 multiply-accumulates each
 *   - Accumulator:   int32_t (safe width, max theoretical sum ~1.03M)
 *   - Bias addition: INT32 pre-scaled bias added directly to accumulator
 *   - Requantize:    Identity (M=1, S=0) -- unquantized signed INT32 logits
 *   - Output tensor: signed INT32 vector [10] (scale = 0.0010914326)
 *
 * Nine-Class Decision Policy:
 *   - Index 9 corresponds to 'transistor' and is strictly masked from prediction.
 *   - Output prediction is argmax over classes 0..8:
 *       predicted_class = argmax_{c in [0..8]} (logits[c])
 *   - Tie-breaking: returns the first/lowest index where maximum occurs
 *     (matching NumPy np.argmax convention).
 *   - Output logits tensor is preserved completely unmodified.
 * ============================================================================
 */

// Dimension constants
constexpr int FC2_IN_FEATURES    = 64;
constexpr int FC2_OUT_CLASSES    = 10;
constexpr int NUM_DEPLOY_CLASSES = 9;  // Permitted classes 0..8
constexpr int TRANSISTOR_INDEX   = 9;  // Excluded deployment class

// Exported weights and biases (declared extern; defined in fc2.cpp)
extern const int8_t  FC2_WEIGHTS[FC2_OUT_CLASSES][FC2_IN_FEATURES];
extern const int32_t FC2_BIASES[FC2_OUT_CLASSES];

/**
 * @brief FC2 Layer Forward Pass (Frozen on-chip weights and biases)
 *
 * Computes raw signed 32-bit integer logits for all 10 classes:
 *   logits[k] = bias[k] + sum_{j=0}^{63} (input[j] * weight[k][j])
 *
 * All 10 logits are computed and stored in INT32 format without ReLU or requantization.
 *
 * @param input  Input activation vector [64] from FC1 (signed INT8)
 * @param output Output logits vector [10] (signed INT32)
 */
void fc2(
    const int8_t input[FC2_IN_FEATURES],
    int32_t output[FC2_OUT_CLASSES]
);

/**
 * @brief Parameterized FC2 Forward Pass (For custom weights/biases testing)
 *
 * @param input   Input activation vector [64]
 * @param output  Output logits vector [10]
 * @param weights Weight matrix [10][64]
 * @param biases  Bias vector [10]
 */
void fc2_custom(
    const int8_t input[FC2_IN_FEATURES],
    int32_t output[FC2_OUT_CLASSES],
    const int8_t weights[FC2_OUT_CLASSES][FC2_IN_FEATURES],
    const int32_t biases[FC2_OUT_CLASSES]
);

/**
 * @brief Nine-Class Argmax Decision Policy with Transistor Exclusion
 *
 * Evaluates logits only over indices [0..8] (classes 0..8):
 *   argmax_{c in [0..8]} (logits[c])
 * Class 9 (transistor) is strictly excluded from decision.
 * Tie-breaking returns the lowest/first class index (matching np.argmax).
 * The input logits tensor is NOT modified.
 *
 * @param logits 10-class logit vector (signed INT32)
 * @return int Winning class index in [0, 8]
 */
int predict_9class(const int32_t logits[FC2_OUT_CLASSES]);

#endif // FC2_H
