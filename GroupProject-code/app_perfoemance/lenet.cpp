#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

using namespace std::chrono;

// 累积时间变量
double total_time_conv1 = 0;
double total_time_pool1 = 0;
double total_time_conv2 = 0;
double total_time_pool2 = 0;
double total_time_conv3 = 0;
double total_time_fc = 0;

double relu(double x)
{
	return x*(x > 0);
}

static void convolution_forward(double input[1][32][32], double output[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], double weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER1]) {
    auto start = high_resolution_clock::now();
    // 原有卷积运算代码
    for (int x = 0; x < INPUT; ++x) {
        for (int y = 0; y < LAYER1; ++y) {
            for (int o0 = 0; o0 < LENGTH_FEATURE1; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE1; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] += (input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER1; j++) {
        for (int i = 0; i < LENGTH_FEATURE1 * LENGTH_FEATURE1; i++) {
            ((double *)output[j])[i] = relu(((double *)output[j])[i] + bias[j]);
        }
    }
    auto end = high_resolution_clock::now();
    total_time_conv1 += duration<double>(end - start).count();
}

static void convolution_forward2(double input[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], double output[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], double weight[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER3]) {
    auto start = high_resolution_clock::now();
    for (int x = 0; x < LAYER2; ++x) {
		for (int y = 0; y < LAYER3; ++y) {
			// convolute_valid(input[x], output[y], weight[x][y]);
            for (int o0 = 0; o0 < LENGTH_FEATURE3; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE3; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] += (input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER3; j++) {
        for (int i = 0; i < LENGTH_FEATURE3 * LENGTH_FEATURE3; i++) {
            ((double *)output[j])[i] = relu(((double *)output[j])[i] + bias[j]);
        }
    }
    auto end = high_resolution_clock::now();
    total_time_conv2 += duration<double>(end - start).count();
}

static void convolution_forward3(double input[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], double output[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], double weight[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER5]) {
    auto start = high_resolution_clock::now();
    for (int x = 0; x < LAYER4; ++x) {
		for (int y = 0; y < LAYER5; ++y) {
			// convolute_valid(input[x], output[y], weight[x][y]);
            for (int o0 = 0; o0 < LENGTH_FEATURE5; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE5; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] += (input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER5; j++) {
        for (int i = 0; i < LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
            ((double *)output[j])[i] = relu(((double *)output[j])[i] + bias[j]);
        }
    }
    auto end = high_resolution_clock::now();
    total_time_conv3 += duration<double>(end - start).count();
}

static void maxpool_forward(double input[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], double output[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2]) {
    auto start = high_resolution_clock::now();
    const int len0 = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    const int len1 = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    for (int i = 0; i < LAYER2; i++) {
        for (int o0 = 0; o0 < LENGTH_FEATURE2; o0++) {
            for (int o1 = 0; o1 < LENGTH_FEATURE2; o1++) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; l0++) {
                    for (int l1 = 0; l1 < len1; l1++) {
                        ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];
            }
        }
    }
    auto end = high_resolution_clock::now();
    total_time_pool1 += duration<double>(end - start).count();
}

static void maxpool_forward2(double input[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], double output[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4]) {
    auto start = high_resolution_clock::now();
    const int len0 = LENGTH_FEATURE3 / LENGTH_FEATURE4;
    const int len1 = LENGTH_FEATURE3 / LENGTH_FEATURE4;
    for (int i = 0; i < LAYER2; i++) {
        for (int o0 = 0; o0 < LENGTH_FEATURE4; o0++) {
            for (int o1 = 0; o1 < LENGTH_FEATURE4; o1++) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; l0++) {
                    for (int l1 = 0; l1 < len1; l1++) {
                        ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];
            }
        }
    }
    auto end = high_resolution_clock::now();
    total_time_pool2 += duration<double>(end - start).count();
}

static void fc_forward(double input[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], double output[OUTPUT], double weight[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], double bias[OUTPUT]) {
    auto start = high_resolution_clock::now();
    // Initialize output array to zero
    for (int y = 0; y < OUTPUT; y++) {
        output[y] = 0.0;
    }

    // Compute fully connected layer forward pass
    for (int i = 0; i < LAYER5; i++) {
        for (int j = 0; j < LENGTH_FEATURE5; j++) {
            for (int k = 0; k < LENGTH_FEATURE5; k++) {
                int idx = i + j + k;
                for (int y = 0; y < OUTPUT; y++) {
                    output[y] += input[i][j][k] * weight[idx][y];
                }
            }
        }
    }

    // Apply bias and activation function
    for (int j = 0; j < OUTPUT; j++) {
        output[j] = relu(output[j] + bias[j]);
    }
    auto end = high_resolution_clock::now();
    total_time_fc += duration<double>(end - start).count();
}

static void forward(
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias0_1[LAYER1],
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias2_3[LAYER3],
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias4_5[LAYER5],
    double weight5_6[LAYER5][OUTPUT],
    double bias5_6[OUTPUT],
    Feature *features, double(*action)(double))
{
    convolution_forward(features->input, features->layer1, weight0_1, bias0_1);
    maxpool_forward(features->layer1, features->layer2);
    convolution_forward2(features->layer2, features->layer3, weight2_3, bias2_3);
    maxpool_forward2(features->layer3, features->layer4);
    convolution_forward3(features->layer4, features->layer5, weight4_5, bias4_5);
    fc_forward(features->layer5, features->output, weight5_6, bias5_6);
}

static inline void load_input(Feature *features, image input)
{
    double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
    const long sz = 28 * 28; // Size of the image (28x28)
    double mean = 0, std = 0;

    for (int j = 0; j < 28; j++) {
        for (int k = 0; k < 28; k++) {
            mean += input[j][k];
            std += input[j][k] * input[j][k];
        }
    }
    mean /= sz;
    std = sqrt(std / sz - mean * mean);

    for (int j = 0; j < 28; j++) {
        for (int k = 0; k < 28; k++) {
            layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
        }
    }
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
    double inner = 0;
    for (int i = 0; i < count; ++i)
    {
        double res = 0;
        for (int j = 0; j < count; ++j)
        {
            res += exp(input[j] - input[i]);
        }
        loss[i] = 1. / res;
        inner -= loss[i] * loss[i];
    }
    inner += loss[label];
    for (int i = 0; i < count; ++i)
    {
        loss[i] *= (i == label) - loss[i] - inner;
    }
}

static uint8 get_result(Feature *features, uint8 count)
{
    double *output = (double *)features->output;
    uint8 result = 0;
    double maxvalue = *output;
    for (uint8 i = 1; i < count; ++i)
    {
        if (output[i] > maxvalue)
        {
            maxvalue = output[i];
            result = i;
        }
    }
    return result;
}

uint8 Predict(
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias0_1[LAYER1],
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias2_3[LAYER3],
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias4_5[LAYER5],
    double weight5_6[LAYER5][OUTPUT],
    double bias5_6[OUTPUT],
    image input, uint8 count)
{
    Feature features = { 0 };
    load_input(&features, input);
    forward(weight0_1, bias0_1, weight2_3, bias2_3, weight4_5, bias4_5, weight5_6, bias5_6, &features, relu);
    return get_result(&features, count);
}
