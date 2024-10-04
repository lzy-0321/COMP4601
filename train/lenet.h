#pragma once

#define LENGTH_KERNEL 5

#define LENGTH_FEATURE0 32
#define LENGTH_FEATURE1 28  // (LENGTH_FEATURE0 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE2 14  // (LENGTH_FEATURE1 >> 1)
#define LENGTH_FEATURE3 10  // (LENGTH_FEATURE2 - LENGTH_KERNEL + 1)
#define LENGTH_FEATURE4 5   // (LENGTH_FEATURE3 >> 1)
#define LENGTH_FEATURE5 1   // (LENGTH_FEATURE4 - LENGTH_KERNEL + 1)

#define INPUT 1
#define LAYER1 6
#define LAYER2 6
#define LAYER3 16
#define LAYER4 16
#define LAYER5 120
#define OUTPUT 10

#define ALPHA 0.5
#define PADDING 2

typedef unsigned char uint8;
typedef uint8 image[28][28];

typedef struct LeNet5 {
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    double weight5_6[LAYER5][OUTPUT];

    double bias0_1[LAYER1];
    double bias2_3[LAYER3];
    double bias4_5[LAYER5];
    double bias5_6[OUTPUT];
} LeNet5;

typedef struct Feature {
    float input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0];
    float layer1[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1];
    float layer2[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2];
    float layer3[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3];
    float layer4[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4];
    float layer5[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5];
    float output[OUTPUT];
} Feature;

void TrainBatch(
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias0_1[LAYER1],
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias2_3[LAYER3],
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias4_5[LAYER5],
    double weight5_6[LAYER5][OUTPUT],
    double bias5_6[OUTPUT],
    image *inputs, uint8 *labels, int batchSize);

uint8 Predict(
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias0_1[LAYER1],
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias2_3[LAYER3],
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias4_5[LAYER5],
    double weight5_6[LAYER5][OUTPUT],
    double bias5_6[OUTPUT],
    image input, uint8 count);

void Initial(LeNet5 *lenet);