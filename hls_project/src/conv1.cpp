#include "conv1.h"

// ============================================================================
// Exported Weights: [16][3][3][3] signed INT8 (432 elements)
// Source: fpga_export/v5_spatial/integer_params/conv1_weight_int8.npy
// Quantization Scale: 0.003316483394367488
// ============================================================================
const int8_t CONV1_WEIGHTS[CONV1_OUT_CH][CONV1_IN_CH][CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE] = {
    { // Filter 0
        { // In-channel 0
            {  47,   54,  -19},
            {  57,   -8,   16},
            { -29,   43,   60},
        },
        { // In-channel 1
            { -35,   60,   11},
            {  52,   17,   37},
            {  -3,   59,   23},
        },
        { // In-channel 2
            { -20,   23,  -30},
            {   1,  -16,   45},
            { -43,  -15,   -4},
        },
    },
    { // Filter 1
        { // In-channel 0
            { -35,   -3,  -62},
            {  51,  -47,   46},
            {  16,  -13,   37},
        },
        { // In-channel 1
            {  10,   41,    3},
            { -22,   15,  -18},
            {  32,   58,   36},
        },
        { // In-channel 2
            { -35,   17,   -3},
            {  12,  -55,  -77},
            { -30,  -56,   36},
        },
    },
    { // Filter 2
        { // In-channel 0
            {  31,   50,   36},
            {  31,   84,  -31},
            {  36,  -22,   24},
        },
        { // In-channel 1
            {  -3,   30,   -3},
            {  84,   -3,  -26},
            {  16,   94,   47},
        },
        { // In-channel 2
            {   4, -117, -119},
            { -86,  -90,  -38},
            {  11,   23,  -76},
        },
    },
    { // Filter 3
        { // In-channel 0
            {  22,   81,    9},
            {  67,   45,   75},
            {  -1,   26,   74},
        },
        { // In-channel 1
            {  35,   -2,   41},
            {  29,  -29,  -26},
            {  61,   -6,   -3},
        },
        { // In-channel 2
            { -96,  -84,  -31},
            { -82,  -96,  -54},
            { -89,  -96,  -75},
        },
    },
    { // Filter 4
        { // In-channel 0
            { -50,  -44,   27},
            {  -7,   -4,  -80},
            { -62,  -43,   -7},
        },
        { // In-channel 1
            { -67,  -46,  -84},
            { -28,  -52,  -45},
            { -78,  -79,  -66},
        },
        { // In-channel 2
            { -33,  -65,    3},
            { -32,   23,  -19},
            { -75,   10,  -38},
        },
    },
    { // Filter 5
        { // In-channel 0
            { -61,  -99, -117},
            { -13,  -52,  -29},
            { -76,   11,  -88},
        },
        { // In-channel 1
            {  -4,    8,   10},
            {  63,   62,  -56},
            {  31,  -14,   -9},
        },
        { // In-channel 2
            { -14,   76,  -35},
            {  67,   54,   38},
            {   9,   90,   41},
        },
    },
    { // Filter 6
        { // In-channel 0
            {  28,  -24,   52},
            { -14,  -35,   36},
            { -30,  -91,  -91},
        },
        { // In-channel 1
            {  37,   79,   29},
            { -26,  -15,  -47},
            { -12,   -7, -115},
        },
        { // In-channel 2
            {  65,   90,   66},
            {   3,   54,   46},
            {   5,  -41,  -55},
        },
    },
    { // Filter 7
        { // In-channel 0
            {  38,   16,   -7},
            {  65,   60,  -21},
            { -10,  -25,  -47},
        },
        { // In-channel 1
            {  56,   54,  -32},
            {  14,   66,   13},
            {   2,  -63,  -75},
        },
        { // In-channel 2
            { -33,   -4,   20},
            {  55,   24,  -82},
            { -42,  -50,    4},
        },
    },
    { // Filter 8
        { // In-channel 0
            {  71,   43,   66},
            { -22,  -57,   45},
            { -12,  -32,  -56},
        },
        { // In-channel 1
            {  60,   58,   25},
            { -92, -127,  -89},
            {  -4,  -50,   -7},
        },
        { // In-channel 2
            {  45,   23,   -2},
            { -95, -118, -100},
            { -18, -100,  -68},
        },
    },
    { // Filter 9
        { // In-channel 0
            { -22,   -6,  -39},
            { -26,   17,   -9},
            {  27,  -18,  -31},
        },
        { // In-channel 1
            { -21,  -44,  -33},
            { -68,    6,  -51},
            { -11,   -3,   25},
        },
        { // In-channel 2
            { -47,   36,  -36},
            {  20,   40,  -11},
            { -24,  -31,  -57},
        },
    },
    { // Filter 10
        { // In-channel 0
            {  68,  -19,   70},
            {  64,   74,    8},
            { -29,   25,   51},
        },
        { // In-channel 1
            { -66,  -23,  -48},
            { -25,  -47,  -41},
            { -49,  -87,   19},
        },
        { // In-channel 2
            {  10,  -48,  -50},
            { -64,  -89,  -95},
            { -25,  -77,  -80},
        },
    },
    { // Filter 11
        { // In-channel 0
            { -20,  -29,  -25},
            { -21,  -21,  -19},
            { -50,  -59,   49},
        },
        { // In-channel 1
            { -35,   10,  -22},
            { -14,   51,   33},
            { -70,   16,   61},
        },
        { // In-channel 2
            {  12,   43,   65},
            { -22,   15,  -29},
            { -28,   39,   37},
        },
    },
    { // Filter 12
        { // In-channel 0
            { -33,  -48,  -45},
            { -60,  -50,  -17},
            {  18,   -4,  -43},
        },
        { // In-channel 1
            { -61,   14,  -58},
            { -25,    5,  -44},
            { -42,  -39,  -38},
        },
        { // In-channel 2
            { -38,  -62,  -58},
            { -41,   40,  -40},
            { -60,  -10,   -1},
        },
    },
    { // Filter 13
        { // In-channel 0
            {  47,   78,   67},
            { -61,   33,   74},
            {  -2,  -11,   80},
        },
        { // In-channel 1
            { -19,   -6,  -20},
            { -15,   31,   38},
            {   4,  -40,  -46},
        },
        { // In-channel 2
            { -32,   55,   47},
            { -94,  -50,  -52},
            { -82,  -32,    3},
        },
    },
    { // Filter 14
        { // In-channel 0
            {  45,  -58,   33},
            { -53,   23,   52},
            {  13,   61,  -65},
        },
        { // In-channel 1
            { -30,   38,    6},
            { -29,  -28,   59},
            {  53,   30,  -42},
        },
        { // In-channel 2
            { -82,  -16,    5},
            { -68,  -50,   24},
            {   0,  -17,  -46},
        },
    },
    { // Filter 15
        { // In-channel 0
            { -38,  -41,  -18},
            { -26,   24,  -11},
            { -30,   41,   23},
        },
        { // In-channel 1
            {  27,  -44,  -41},
            {   3,  -49,  -48},
            { -33,  -54,  -60},
        },
        { // In-channel 2
            {  33,   42,    3},
            { -13,  -49,  -17},
            {   5,   17,   -4},
        },
    },
};

// ============================================================================
// Exported Biases: [16] signed INT32 (16 elements)
// Source: fpga_export/v5_spatial/integer_params/conv1_bias_int32.npy
// Pre-scaled by (s_in * s_w) = 2.611395015383921e-05
// ============================================================================
const int32_t CONV1_BIASES[CONV1_OUT_CH] = {
    -2489, 7842, -127, 19140, -2502, 1659, 1027, -467, -3159, 14547, 5359, 8109, 12870, -147, -5964, 15820
};

// ============================================================================
// Fixed-Point Requantization Scalar Function
// Formula: ((acc * M + (1LL << (S - 1))) >> S), clamped to [-128, 127]
// ============================================================================
int8_t conv1_requantize(int64_t acc, int64_t M, int S) {
    if (S <= 0) {
        int64_t prod = acc * M;
        if (prod > 127) return 127;
        if (prod < -128) return -128;
        return static_cast<int8_t>(prod);
    }
    int64_t rounding_bias = 1LL << (S - 1);
    int64_t prod = acc * M;
    int64_t rounded = (prod + rounding_bias) >> S;
    if (rounded > 127) return 127;
    if (rounded < -128) return -128;
    return static_cast<int8_t>(rounded);
}

// ============================================================================
// Parameterized Conv1 Execution Engine
// ============================================================================
void conv1_custom(
    const int8_t input[CONV1_IN_CH][CONV1_IN_H][CONV1_IN_W],
    int8_t output[CONV1_OUT_CH][CONV1_OUT_H][CONV1_OUT_W],
    const int8_t weights[CONV1_OUT_CH][CONV1_IN_CH][CONV1_KERNEL_SIZE][CONV1_KERNEL_SIZE],
    const int32_t biases[CONV1_OUT_CH],
    int64_t multiplier,
    int shift
) {
    // Intermediate feature map: 16 channels, 64 rows, 64 cols
    // Allocated as a local array (64 KB)
    int8_t conv_out[CONV1_OUT_CH][CONV1_CONV_H][CONV1_CONV_W];

    // 1. Convolution (3x3, pad 1, stride 1) + Bias Addition + ReLU + Requantization
    for (int m = 0; m < CONV1_OUT_CH; ++m) {
        const int32_t b_val = biases[m];

        for (int y = 0; y < CONV1_CONV_H; ++y) {
            for (int x = 0; x < CONV1_CONV_W; ++x) {
                int64_t acc = 0;

                for (int c = 0; c < CONV1_IN_CH; ++c) {
                    for (int ky = 0; ky < CONV1_KERNEL_SIZE; ++ky) {
                        int in_y = y + ky - CONV1_PAD;
                        for (int kx = 0; kx < CONV1_KERNEL_SIZE; ++kx) {
                            int in_x = x + kx - CONV1_PAD;

                            int8_t in_val = 0;
                            if (in_y >= 0 && in_y < CONV1_IN_H && in_x >= 0 && in_x < CONV1_IN_W) {
                                in_val = input[c][in_y][in_x];
                            }
                            acc += static_cast<int32_t>(in_val) * static_cast<int32_t>(weights[m][c][ky][kx]);
                        }
                    }
                }

                // Add pre-scaled INT32 bias
                acc += b_val;

                // Integer ReLU
                if (acc < 0) {
                    acc = 0;
                }

                // Fixed-point requantization with round-half-up and saturation
                conv_out[m][y][x] = conv1_requantize(acc, multiplier, shift);
            }
        }
    }

    // 2. 2x2 Max Pooling with stride 2 -> [16][32][32]
    for (int m = 0; m < CONV1_OUT_CH; ++m) {
        for (int oy = 0; oy < CONV1_OUT_H; ++oy) {
            for (int ox = 0; ox < CONV1_OUT_W; ++ox) {
                int iy = oy * 2;
                int ix = ox * 2;

                int8_t v00 = conv_out[m][iy][ix];
                int8_t v01 = conv_out[m][iy][ix + 1];
                int8_t v10 = conv_out[m][iy + 1][ix];
                int8_t v11 = conv_out[m][iy + 1][ix + 1];

                int8_t max_val = v00;
                if (v01 > max_val) max_val = v01;
                if (v10 > max_val) max_val = v10;
                if (v11 > max_val) max_val = v11;

                output[m][oy][ox] = max_val;
            }
        }
    }
}

// ============================================================================
// Standard Conv1 Forward Pass (Uses frozen on-chip parameters)
// ============================================================================
void conv1(
    const int8_t input[CONV1_IN_CH][CONV1_IN_H][CONV1_IN_W],
    int8_t output[CONV1_OUT_CH][CONV1_OUT_H][CONV1_OUT_W]
) {
    conv1_custom(input, output, CONV1_WEIGHTS, CONV1_BIASES, CONV1_M, CONV1_S);
}
