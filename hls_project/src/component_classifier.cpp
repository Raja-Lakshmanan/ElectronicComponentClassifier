#include "component_classifier.h"

// ============================================================================
// Permitted Deployment Class Names (in index order 0..8)
#ifndef __SYNTHESIS__
const char* const CLASSIFIER_CLASS_NAMES[CLASSIFIER_NUM_PERMITTED] = {
    "capacitor",
    "diode",
    "fuse",
    "inductor",
    "integrated_circuit",
    "led",
    "potentiometer",
    "push_button",
    "resistor"
};
#endif

// ============================================================================
// Input Quantization Lookup Table (256 entries)
// Source: fpga_export/v5_spatial/integer_params/input_lut_int8.npy
// Formula: clip(round((2.0*P/255.0 - 1.0) / 0.00787402), -128, 127)
// Range: [-127, 127]
// ============================================================================
const int8_t INPUT_LUT[256] = {
      -127,   -126,   -125,   -124,   -123,   -122,   -121,   -120,   -119,   -118,   -117,   -116,   -115,   -114,   -113,   -112,
      -111,   -110,   -109,   -108,   -107,   -106,   -105,   -104,   -103,   -102,   -101,   -100,    -99,    -98,    -97,    -96,
       -95,    -94,    -93,    -92,    -91,    -90,    -89,    -88,    -87,    -86,    -85,    -84,    -83,    -82,    -81,    -80,
       -79,    -78,    -77,    -76,    -75,    -74,    -73,    -72,    -71,    -70,    -69,    -68,    -67,    -66,    -65,    -64,
       -63,    -62,    -61,    -60,    -59,    -58,    -57,    -56,    -55,    -54,    -53,    -52,    -51,    -50,    -49,    -48,
       -47,    -46,    -45,    -44,    -43,    -42,    -41,    -40,    -39,    -38,    -37,    -36,    -35,    -34,    -33,    -32,
       -31,    -30,    -29,    -28,    -27,    -26,    -25,    -24,    -23,    -22,    -21,    -20,    -19,    -18,    -17,    -16,
       -15,    -14,    -13,    -12,    -11,    -10,     -9,     -8,     -7,     -6,     -5,     -4,     -3,     -2,     -1,      0,
         0,      1,      2,      3,      4,      5,      6,      7,      8,      9,     10,     11,     12,     13,     14,     15,
        16,     17,     18,     19,     20,     21,     22,     23,     24,     25,     26,     27,     28,     29,     30,     31,
        32,     33,     34,     35,     36,     37,     38,     39,     40,     41,     42,     43,     44,     45,     46,     47,
        48,     49,     50,     51,     52,     53,     54,     55,     56,     57,     58,     59,     60,     61,     62,     63,
        64,     65,     66,     67,     68,     69,     70,     71,     72,     73,     74,     75,     76,     77,     78,     79,
        80,     81,     82,     83,     84,     85,     86,     87,     88,     89,     90,     91,     92,     93,     94,     95,
        96,     97,     98,     99,    100,    101,    102,    103,    104,    105,    106,    107,    108,    109,    110,    111,
       112,    113,    114,    115,    116,    117,    118,    119,    120,    121,    122,    123,    124,    125,    126,    127,
};

// ============================================================================
// Input Pixel Quantization Helper
// ============================================================================
void quantize_input_image_lut(
    const uint8_t image_rgb[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W],
    int8_t activations_out[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W]
) {
    for (int c = 0; c < CLASSIFIER_IN_CH; ++c) {
        for (int h = 0; h < CLASSIFIER_IN_H; ++h) {
            for (int w = 0; w < CLASSIFIER_IN_W; ++w) {
                activations_out[c][h][w] = INPUT_LUT[image_rgb[c][h][w]];
            }
        }
    }
}

// ============================================================================
// Top-Level Component Classifier Inference Pipeline
// ============================================================================
int component_classifier(
    const uint8_t image_rgb[CLASSIFIER_IN_CH][CLASSIFIER_IN_H][CLASSIFIER_IN_W],
    int32_t logits_out[CLASSIFIER_NUM_CLASSES]
) {
    // Internal activation ping-pong buffers
    // Total buffer footprint: ~41.5 KB (fits safely on stack / in BRAM)
    int8_t x0[CONV1_IN_CH][CONV1_IN_H][CONV1_IN_W];             // [3][64][64] = 12,288 B
    int8_t x1[CONV1_OUT_CH][CONV1_OUT_H][CONV1_OUT_W];          // [16][32][32] = 16,384 B
    int8_t x2[CONV2_OUT_CH][CONV2_OUT_H][CONV2_OUT_W];          // [32][16][16] = 8,192 B
    int8_t x3[CONV3_OUT_CH][CONV3_OUT_H][CONV3_OUT_W];          // [64][8][8] = 4,096 B
    int8_t x4[SPATIAL_POOL_OUT_CH][SPATIAL_POOL_OUT_H][SPATIAL_POOL_OUT_W]; // [64][2][2] = 256 B
    int8_t x_flat[FC1_IN_FEATURES];                             // [256] = 256 B
    int8_t x5[FC1_OUT_FEATURES];                                // [64] = 64 B

    // Stage 1: Input LUT Quantization: uint8 [3][64][64] -> int8 [3][64][64]
    quantize_input_image_lut(image_rgb, x0);

    // Stage 2: Conv Block 1: Conv2d + Bias + ReLU + Requant + MaxPool2d (64x64 -> 32x32)
    conv1(x0, x1);

    // Stage 3: Conv Block 2: Conv2d + Bias + ReLU + Requant + MaxPool2d (32x32 -> 16x16)
    conv2(x1, x2);

    // Stage 4: Conv Block 3: Conv2d + Bias + ReLU + Requant + MaxPool2d (16x16 -> 8x8)
    conv3(x2, x3);

    // Stage 5: Spatial Adaptive Pool: 4x4 AvgPool + Requantization to FC1 Scale (8x8 -> 2x2)
    // Matches integer_v5_reference.py _spatial_adaptive_pool() exactly
    spatial_pool_with_requant(x3, x4);

    // Stage 6: Quadrant Flattening (C-contiguous order: c * 4 + py * 2 + px)
    for (int c = 0; c < SPATIAL_POOL_OUT_CH; ++c) {
        for (int py = 0; py < SPATIAL_POOL_OUT_H; ++py) {
            for (int px = 0; px < SPATIAL_POOL_OUT_W; ++px) {
                x_flat[c * 4 + py * 2 + px] = x4[c][py][px];
            }
        }
    }

    // Stage 7: FC1 Layer: Matrix MAC + Bias + ReLU + Requantization (256 -> 64)
    fc1(x_flat, x5);

    // Stage 8: FC2 Layer: Matrix MAC + Bias -> Raw Signed INT32 Logits (64 -> 10)
    fc2(x5, logits_out);

    // Stage 9: Nine-Class Argmax Decision Policy (Excluding Class 9 Transistor)
    int predicted_class = predict_9class(logits_out);

    return predicted_class;
}
