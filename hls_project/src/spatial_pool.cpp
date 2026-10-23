#include "spatial_pool.h"

// ============================================================================
// Exact Round-to-Nearest-Even Integer Division by 16 (Banker's Rounding)
// ============================================================================
int8_t spatial_pool_div16_round_even(int32_t sum) {
    // Arithmetic right shift produces floor division: div = sum // 16
    // Lower 4 bits produce non-negative remainder: rem = sum & 15
    int32_t div = sum >> 4;
    int32_t rem = sum & 0xF;

    if (rem > 8) {
        return static_cast<int8_t>(div + 1);
    } else if (rem < 8) {
        return static_cast<int8_t>(div);
    } else {
        // Exactly on half-way boundary (rem == 8): round to nearest even quotient
        // If div is odd (div & 1 != 0), round up to div + 1 (which is even)
        // If div is even (div & 1 == 0), round down to div (which is already even)
        return static_cast<int8_t>((div & 1) ? (div + 1) : div);
    }
}

// ============================================================================
// Spatial Adaptive Average Pooling (Pure Average: 64x8x8 -> 64x2x2)
// ============================================================================
void spatial_pool(
    const int8_t input[SPATIAL_POOL_IN_CH][SPATIAL_POOL_IN_H][SPATIAL_POOL_IN_W],
    int8_t output[SPATIAL_POOL_OUT_CH][SPATIAL_POOL_OUT_H][SPATIAL_POOL_OUT_W]
) {
    for (int c = 0; c < SPATIAL_POOL_OUT_CH; ++c) {
        for (int py = 0; py < SPATIAL_POOL_OUT_H; ++py) {
            for (int px = 0; px < SPATIAL_POOL_OUT_W; ++px) {
                int32_t sum = 0;
                int start_y = py * SPATIAL_POOL_REGION_H;
                int start_x = px * SPATIAL_POOL_REGION_W;

                for (int dy = 0; dy < SPATIAL_POOL_REGION_H; ++dy) {
                    for (int dx = 0; dx < SPATIAL_POOL_REGION_W; ++dx) {
                        sum += input[c][start_y + dy][start_x + dx];
                    }
                }

                output[c][py][px] = spatial_pool_div16_round_even(sum);
            }
        }
    }
}

// ============================================================================
// Spatial Adaptive Average Pooling with Fixed-Point Requantization
// ============================================================================
void spatial_pool_with_requant(
    const int8_t input[SPATIAL_POOL_IN_CH][SPATIAL_POOL_IN_H][SPATIAL_POOL_IN_W],
    int8_t output[SPATIAL_POOL_OUT_CH][SPATIAL_POOL_OUT_H][SPATIAL_POOL_OUT_W],
    int64_t multiplier,
    int shift
) {
    int64_t rounding_bias = (shift > 0) ? (1LL << (shift - 1)) : 0LL;

    for (int c = 0; c < SPATIAL_POOL_OUT_CH; ++c) {
        for (int py = 0; py < SPATIAL_POOL_OUT_H; ++py) {
            for (int px = 0; px < SPATIAL_POOL_OUT_W; ++px) {
                int32_t sum = 0;
                int start_y = py * SPATIAL_POOL_REGION_H;
                int start_x = px * SPATIAL_POOL_REGION_W;

                for (int dy = 0; dy < SPATIAL_POOL_REGION_H; ++dy) {
                    for (int dx = 0; dx < SPATIAL_POOL_REGION_W; ++dx) {
                        sum += input[c][start_y + dy][start_x + dx];
                    }
                }

                int8_t avg = spatial_pool_div16_round_even(sum);
                int64_t prod = static_cast<int64_t>(avg) * multiplier;
                int64_t scaled = (shift > 0) ? ((prod + rounding_bias) >> shift) : prod;

                if (scaled > 127) scaled = 127;
                if (scaled < -128) scaled = -128;

                output[c][py][px] = static_cast<int8_t>(scaled);
            }
        }
    }
}
