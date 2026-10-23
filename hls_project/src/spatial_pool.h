#ifndef SPATIAL_POOL_H
#define SPATIAL_POOL_H

#include <cstdint>
#include <cstddef>

/**
 * ============================================================================
 * V5-Spatial HLS C++ Implementation: Spatial Adaptive Average Pooling
 * ============================================================================
 * Architecture:
 *   - Input tensor:  signed INT8 CHW [64][8][8] (post Conv3+ReLU+MaxPool3)
 *   - Input range:   [0, 127] (non-negative due to preceding ReLU and MaxPool)
 *   - Output tensor: signed INT8 CHW [64][2][2] (4 spatial quadrants per channel)
 *   - Spatial partitioning:
 *       Quadrant (0, 0): rows [0..3], cols [0..3] (16 elements)
 *       Quadrant (0, 1): rows [0..3], cols [4..7] (16 elements)
 *       Quadrant (1, 0): rows [4..7], cols [0..3] (16 elements)
 *       Quadrant (1, 1): rows [4..7], cols [4..7] (16 elements)
 *   - Accumulator:   int32_t (max sum for 16 non-negative INT8 values is 16*127 = 2032)
 *   - Division:      Exact round-to-nearest-even division by 16 (Banker's rounding)
 *                    matching IEEE 754 / np.round() / integer_v5_reference.py
 *   - Requantization (optional / for FC1 interface):
 *                    M_pool = 103835, S_pool = 16
 *                    Scaling: 0.50742737 (Conv3 output) -> 0.32026483 (FC1 input)
 * ============================================================================
 */

// Dimension constants
constexpr int SPATIAL_POOL_IN_CH = 64;
constexpr int SPATIAL_POOL_IN_H  = 8;
constexpr int SPATIAL_POOL_IN_W  = 8;

constexpr int SPATIAL_POOL_OUT_CH = 64;
constexpr int SPATIAL_POOL_OUT_H  = 2;
constexpr int SPATIAL_POOL_OUT_W  = 2;

constexpr int SPATIAL_POOL_REGION_H = 4;
constexpr int SPATIAL_POOL_REGION_W = 4;
constexpr int SPATIAL_POOL_REGION_SIZE = 16;

// Requantization parameters to match FC1 input scale (M_pool = 103835, S_pool = 16)
constexpr int64_t SPATIAL_POOL_M = 103835LL;
constexpr int     SPATIAL_POOL_S = 16;

/**
 * @brief Round-to-nearest-even integer division by 16 (Banker's rounding).
 *
 * For non-negative integer sums:
 *   div = sum >> 4;
 *   rem = sum & 0xF;
 *   if (rem > 8)       return div + 1;
 *   else if (rem < 8)  return div;
 *   else               return (div & 1) ? (div + 1) : div; // tie to even
 *
 * This function guarantees bit-exact agreement with np.round(sum / 16.0)
 * across all non-negative sums [0, 2032] and handles negative sums via
 * arithmetic right-shift / floor division semantics.
 *
 * @param sum 16-element integer sum (range [0, 2032] for non-negative inputs)
 * @return int8_t Rounded quotient
 */
int8_t spatial_pool_div16_round_even(int32_t sum);

/**
 * @brief Spatial Adaptive Average Pooling Forward Pass (Pure Average).
 *
 * Partitions each 8x8 channel into four non-overlapping 4x4 regions, sums the
 * 16 elements in each region, and divides by 16 using round-to-nearest-even.
 *
 * @param input  Input activation tensor [64][8][8] (signed INT8)
 * @param output Output activation tensor [64][2][2] (signed INT8, average values)
 */
void spatial_pool(
    const int8_t input[SPATIAL_POOL_IN_CH][SPATIAL_POOL_IN_H][SPATIAL_POOL_IN_W],
    int8_t output[SPATIAL_POOL_OUT_CH][SPATIAL_POOL_OUT_H][SPATIAL_POOL_OUT_W]
);

/**
 * @brief Spatial Adaptive Average Pooling with Fixed-Point Requantization.
 *
 * Combines spatial average pooling with requantization to FC1 input scale:
 *   avg = round_even(sum / 16)
 *   q_out = clip(((avg * M_pool + (1 << (S_pool - 1))) >> S_pool), -128, 127)
 * Matches integer_v5_reference.py _spatial_adaptive_pool() return value.
 *
 * @param input      Input activation tensor [64][8][8]
 * @param output     Output activation tensor [64][2][2]
 * @param multiplier Fixed-point multiplier M (default SPATIAL_POOL_M = 103835)
 * @param shift      Right shift S (default SPATIAL_POOL_S = 16)
 */
void spatial_pool_with_requant(
    const int8_t input[SPATIAL_POOL_IN_CH][SPATIAL_POOL_IN_H][SPATIAL_POOL_IN_W],
    int8_t output[SPATIAL_POOL_OUT_CH][SPATIAL_POOL_OUT_H][SPATIAL_POOL_OUT_W],
    int64_t multiplier = SPATIAL_POOL_M,
    int shift = SPATIAL_POOL_S
);

#endif // SPATIAL_POOL_H
