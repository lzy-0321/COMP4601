#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))
#define GETCOUNT(array)  (sizeof(array)/sizeof(float))
#define FOREACH(i,count) for (int i = 0; i < count; ++i)

float relu(float x) {
    return x * (x > 0);
}

float relugrad(float y) {
    return y > 0;
}

static void convert_weights(float *dest, double *src, int size) {
    for (int i = 0; i < size; ++i) {
        dest[i] = (float)src[i];
    }
}

static void convert_weights_back(double *dest, float *src, int size) {
    for (int i = 0; i < size; ++i) {
        dest[i] = (double)src[i];
    }
}

// 前向传播函数
static void convolution_forward(float input[1][32][32], float output[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], double weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER1]) {
    float weight_f[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    float bias_f[LAYER1];
    convert_weights((float *)weight_f, (double *)weight, sizeof(weight_f) / sizeof(float));
    convert_weights((float *)bias_f, (double *)bias, sizeof(bias_f) / sizeof(float));
    
    for (int x = 0; x < INPUT; ++x) {
        for (int y = 0; y < LAYER1; ++y) {
            for (int o0 = 0; o0 < LENGTH_FEATURE1; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE1; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            output[y][o0][o1] += input[x][o0 + w0][o1 + w1] * weight_f[x][y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER1; j++) {
        for (int i = 0; i < LENGTH_FEATURE1 * LENGTH_FEATURE1; i++) {
            ((float *)output[j])[i] = relu(((float *)output[j])[i] + bias_f[j]);
        }
    }
}

// 其他前向传播函数类似地修改为使用 float 类型
static void convolution_forward2(float input[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], float output[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], double weight[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER3]) {
    float weight_f[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    float bias_f[LAYER3];
    convert_weights((float *)weight_f, (double *)weight, sizeof(weight_f) / sizeof(float));
    convert_weights((float *)bias_f, (double *)bias, sizeof(bias_f) / sizeof(float));
    
    for (int x = 0; x < LAYER2; ++x) {
        for (int y = 0; y < LAYER3; ++y) {
            for (int o0 = 0; o0 < LENGTH_FEATURE3; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE3; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            output[y][o0][o1] += input[x][o0 + w0][o1 + w1] * weight_f[x][y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER3; j++) {
        for (int i = 0; i < LENGTH_FEATURE3 * LENGTH_FEATURE3; i++) {
            ((float *)output[j])[i] = relu(((float *)output[j])[i] + bias_f[j]);
        }
    }
}

static void convolution_forward3(float input[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], float output[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], double weight[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER5]) {
    float weight_f[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    float bias_f[LAYER5];
    convert_weights((float *)weight_f, (double *)weight, sizeof(weight_f) / sizeof(float));
    convert_weights((float *)bias_f, (double *)bias, sizeof(bias_f) / sizeof(float));
    
    for (int x = 0; x < LAYER4; ++x) {
        for (int y = 0; y < LAYER5; ++y) {
            for (int o0 = 0; o0 < LENGTH_FEATURE5; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE5; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            output[y][o0][o1] += input[x][o0 + w0][o1 + w1] * weight_f[x][y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER5; j++) {
        for (int i = 0; i < LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
            ((float *)output[j])[i] = relu(((float *)output[j])[i] + bias_f[j]);
        }
    }
}

// 反向传播函数
static void convolution_backward(float input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0], float inerror[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0], float outerror[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], double weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], double wd[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], double bd[LAYER1]) {
    float weight_f[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL];
    float wd_f[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL] = {0};
    float bd_f[LAYER1] = {0};
    convert_weights((float *)weight_f, (double *)weight, sizeof(weight_f) / sizeof(float));
    
    for (int x = 0; x < INPUT; ++x) {
        for (int y = 0; y < LAYER1; ++y) {
            for (int i0 = 0; i0 < LENGTH_FEATURE1; i0++) {
                for (int i1 = 0; i1 < LENGTH_FEATURE1; i1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            inerror[x][i0 + w0][i1 + w1] += outerror[y][i0][i1] * weight_f[x][y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < INPUT * LENGTH_FEATURE0 * LENGTH_FEATURE0; i++) {
        ((float *)inerror)[i] *= relugrad(((float *)input)[i]);
    }
    for (int j = 0; j < LAYER1; j++) {
        for (int i = 0; i < LENGTH_FEATURE1 * LENGTH_FEATURE1; i++) {
            bd_f[j] += ((float *)outerror[j])[i];
        }
    }
    for (int x = 0; x < INPUT; x++) {
        for (int y = 0; y < LAYER1; y++) {
            for (int o0 = 0; o0 < LENGTH_KERNEL; o0++) {
                for (int o1 = 0; o1 < LENGTH_KERNEL; o1++) {
                    for (int w0 = 0; w0 < LENGTH_FEATURE1; w0++) {
                        for (int w1 = 0; w1 < LENGTH_FEATURE1; w1++) {
                            wd_f[x][y][o0][o1] += input[x][o0 + w0][o1 + w1] * outerror[y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    
    convert_weights_back((double *)wd, (float *)wd_f, sizeof(wd_f) / sizeof(float));
    convert_weights_back((double *)bd, (float *)bd_f, sizeof(bd_f) / sizeof(float));
}

static void convolution_backward2(float input[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], float inerror[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], float outerror[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], double weight[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], double wd[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], double bd[LAYER3]) {
    float weight_f[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL];
    float wd_f[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL] = {0};
    float bd_f[LAYER3] = {0};
    convert_weights((float *)weight_f, (double *)weight, sizeof(weight_f) / sizeof(float));
    
    for (int x = 0; x < LAYER2; ++x) {
        for (int y = 0; y < LAYER3; ++y) {
            for (int i0 = 0; i0 < LENGTH_FEATURE3; i0++) {
                for (int i1 = 0; i1 < LENGTH_FEATURE3; i1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            inerror[x][i0 + w0][i1 + w1] += outerror[y][i0][i1] * weight_f[x][y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < LAYER2 * LENGTH_FEATURE2 * LENGTH_FEATURE2; i++) {
        ((float *)inerror)[i] *= relugrad(((float *)input)[i]);
    }
    for (int j = 0; j < LAYER3; j++) {
        for (int i = 0; i < LENGTH_FEATURE3 * LENGTH_FEATURE3; i++) {
            bd_f[j] += ((float *)outerror[j])[i];
        }
    }
    for (int x = 0; x < LAYER2; x++) {
        for (int y = 0; y < LAYER3; y++) {
            for (int o0 = 0; o0 < LENGTH_KERNEL; o0++) {
                for (int o1 = 0; o1 < LENGTH_KERNEL; o1++) {
                    for (int w0 = 0; w0 < LENGTH_FEATURE3; w0++) {
                        for (int w1 = 0; w1 < LENGTH_FEATURE3; w1++) {
                            wd_f[x][y][o0][o1] += input[x][o0 + w0][o1 + w1] * outerror[y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    
    convert_weights_back((double *)wd, (float *)wd_f, sizeof(wd_f) / sizeof(float));
    convert_weights_back((double *)bd, (float *)bd_f, sizeof(bd_f) / sizeof(float));
}

static void convolution_backward3(float input[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], float inerror[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], float outerror[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], double weight[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], double wd[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], double bd[LAYER5]) {
    float weight_f[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL];
    float wd_f[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL] = {0};
    float bd_f[LAYER5] = {0};
    convert_weights((float *)weight_f, (double *)weight, sizeof(weight_f) / sizeof(float));
    
    for (int x = 0; x < LAYER4; ++x) {
        for (int y = 0; y < LAYER5; ++y) {
            for (int i0 = 0; i0 < LENGTH_FEATURE5; i0++) {
                for (int i1 = 0; i1 < LENGTH_FEATURE5; i1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            inerror[x][i0 + w0][i1 + w1] += outerror[y][i0][i1] * weight_f[x][y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < LAYER4 * LENGTH_FEATURE4 * LENGTH_FEATURE4; i++) {
        ((float *)inerror)[i] *= relugrad(((float *)input)[i]);
    }
    for (int j = 0; j < LAYER5; j++) {
        for (int i = 0; i < LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
            bd_f[j] += ((float *)outerror[j])[i];
        }
    }
    for (int x = 0; x < LAYER4; x++) {
        for (int y = 0; y < LAYER5; y++) {
            for (int o0 = 0; o0 < LENGTH_KERNEL; o0++) {
                for (int o1 = 0; o1 < LENGTH_KERNEL; o1++) {
                    for (int w0 = 0; w0 < LENGTH_FEATURE5; w0++) {
                        for (int w1 = 0; w1 < LENGTH_FEATURE5; w1++) {
                            wd_f[x][y][o0][o1] += input[x][o0 + w0][o1 + w1] * outerror[y][w0][w1];
                        }
                    }
                }
            }
        }
    }
    
    convert_weights_back((double *)wd, (float *)wd_f, sizeof(wd_f) / sizeof(float));
    convert_weights_back((double *)bd, (float *)bd_f, sizeof(bd_f) / sizeof(float));
}

static void fc_forward(float input[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], float output[OUTPUT], double weight[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], double bias[OUTPUT]) {
    float weight_f[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];
    float bias_f[OUTPUT];
    convert_weights((float *)weight_f, (double *)weight, sizeof(weight_f) / sizeof(float));
    convert_weights((float *)bias_f, (double *)bias, sizeof(bias_f) / sizeof(float));
    
    for (int y = 0; y < OUTPUT; y++) {
        output[y] = 0.0;
    }

    for (int i = 0; i < LAYER5; i++) {
        for (int j = 0; j < LENGTH_FEATURE5; j++) {
            for (int k = 0; j < LENGTH_FEATURE5; j++) {
                int idx = i * LENGTH_FEATURE5 * LENGTH_FEATURE5 + j * LENGTH_FEATURE5 + k;
                for (int y = 0; y < OUTPUT; y++) {
                    output[y] += input[i][j][k] * weight_f[idx][y];
                }
            }
        }
    }
    
    for (int j = 0; j < OUTPUT; j++) {
        output[j] = relu(output[j] + bias_f[j]);
    }
}

static void fc_backward(float input[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], float inerror[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], float outerror[OUTPUT], double weight[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], double wd[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], double bd[OUTPUT]) {
    float weight_f[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT];
    float wd_f[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT] = {0};
    float bd_f[OUTPUT] = {0};
    convert_weights((float *)weight_f, (double *)weight, sizeof(weight_f) / sizeof(float));
    
    for (int x = 0; x < LAYER5; x++) {
        for (int j = 0; j < LENGTH_FEATURE5; j++) {
            for (int k = 0; k < LENGTH_FEATURE5; k++) {
                inerror[x][j][k] = 0.0;
            }
        }
    }
    
    for (int x = 0; x < LAYER5; x++) {
        for (int j = 0; j < LENGTH_FEATURE5; j++) {
            for (int k = 0; k < LENGTH_FEATURE5; k++) {
                int idx = x * LENGTH_FEATURE5 * LENGTH_FEATURE5 + j * LENGTH_FEATURE5 + k;
                for (int y = 0; y < OUTPUT; y++) {
                    inerror[x][j][k] += outerror[y] * weight_f[idx][y];
                }
                inerror[x][j][k] *= relugrad(input[x][j][k]);
            }
        }
    }
    
    for (int j = 0; j < OUTPUT; j++) {
        bd_f[j] += outerror[j];
    }
    for (int x = 0; x < LAYER5; x++) {
        for (int j = 0; j < LENGTH_FEATURE5; j++) {
            for (int k = 0; k < LENGTH_FEATURE5; k++) {
                int idx = x * LENGTH_FEATURE5 * LENGTH_FEATURE5 + j * LENGTH_FEATURE5 + k;
                for (int y = 0; y < OUTPUT; y++) {
                    wd_f[idx][y] += input[x][j][k] * outerror[y];
                }
            }
        }
    }
    
    convert_weights_back((double *)wd, (float *)wd_f, sizeof(wd_f) / sizeof(float));
    convert_weights_back((double *)bd, (float *)bd_f, sizeof(bd_f) / sizeof(float));
}

void forward(
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias0_1[LAYER1],
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias2_3[LAYER3],
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias4_5[LAYER5],
    double weight5_6[LAYER5][OUTPUT],
    double bias5_6[OUTPUT],
    Feature *features, float(*action)(float))
{
    convolution_forward(features->input, features->layer1, weight0_1, bias0_1);
    convolution_forward2(features->layer2, features->layer3, weight2_3, bias2_3);
    convolution_forward3(features->layer4, features->layer5, weight4_5, bias4_5);
    fc_forward(features->layer5, features->output, weight5_6, bias5_6);
}

void backward(
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL],
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL],
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL],
    double weight5_6[LAYER5][OUTPUT],
    LeNet5 *deltas, Feature *errors, Feature *features, float(*actiongrad)(float))
{
    fc_backward(features->layer5, errors->layer5, errors->output, weight5_6, deltas->weight5_6, deltas->bias5_6);
    convolution_backward3(features->layer4, errors->layer4, errors->layer5, weight4_5, deltas->weight4_5, deltas->bias4_5);
    convolution_backward2(features->layer2, errors->layer2, errors->layer3, weight2_3, deltas->weight2_3, deltas->bias2_3);
    convolution_backward(features->input, errors->input, errors->layer1, weight0_1, deltas->weight0_1, deltas->bias0_1);
}

static inline void load_input(Feature *features, image input) {
    float (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
    const long sz = 28 * 28; // Size of the image (28x28)
    float mean = 0, std = 0;

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

static inline void softmax(float input[OUTPUT], float loss[OUTPUT], int label, int count) {
    float inner = 0;
    for (int i = 0; i < count; ++i) {
        float res = 0;
        for (int j = 0; j < count; ++j) {
            res += exp(input[j] - input[i]);
        }
        loss[i] = 1. / res;
        inner -= loss[i] * loss[i];
    }
    inner += loss[label];
    for (int i = 0; i < count; ++i) {
        loss[i] *= (i == label) - loss[i] - inner;
    }
}

static void load_target(Feature *features, Feature *errors, int label) {
    float *output = (float *)features->output;
    float *error = (float *)errors->output;
    softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count) {
    float *output = (float *)features->output;
    uint8 result = 0;
    float maxvalue = *output;
    for (uint8 i = 1; i < count; ++i) {
        if (output[i] > maxvalue) {
            maxvalue = output[i];
            result = i;
        }
    }
    return result;
}

static unsigned long next = 1;
int my_rand(void) {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % 32768;
}

void my_srand(unsigned int seed) {
    next = seed;
}

float f64rand() {
    static int randbit = 0;
    if (!randbit) {
        my_srand(1); // Initialize with a seed, e.g., 1
        for (int i = 32767; i; i >>= 1, ++randbit);
    }
    unsigned long long lvalue = 0x4000000000000000L;
    int i = 52 - randbit;
    for (; i > 0; i -= randbit)
        lvalue |= (unsigned long long)my_rand() << i;
    lvalue |= (unsigned long long)my_rand() >> -i;
    return *(float *)&lvalue - 3;
}

void TrainBatch(
    double weight0_1[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias0_1[LAYER1],
    double weight2_3[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias2_3[LAYER3],
    double weight4_5[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL],
    double bias4_5[LAYER5],
    double weight5_6[LAYER5][OUTPUT],
    double bias5_6[OUTPUT],
    image *inputs, uint8 *labels, int batchSize)
{
    LeNet5 buffer = { 0 }; // Ensure buffer is initialized
    int i, j;

    // Parallel processing for each batch
    for (i = 0; i < batchSize; ++i)
    {
        Feature features = { 0 };
        Feature errors = { 0 };
        LeNet5 deltas = { 0 };
        load_input(&features, inputs[i]);
        forward(weight0_1, bias0_1, weight2_3, bias2_3, weight4_5, bias4_5, weight5_6, bias5_6, &features, relu);
        load_target(&features, &errors, labels[i]);
        backward(weight0_1, weight2_3, weight4_5, weight5_6, &deltas, &errors, &features, relugrad);

        // Aggregate deltas into buffer
        for (j = 0; j < sizeof(buffer.weight0_1) / sizeof(double); ++j) {
            ((double *)buffer.weight0_1)[j] += ((double *)deltas.weight0_1)[j];
        }
        for (j = 0; j < sizeof(buffer.bias0_1) / sizeof(double); ++j) {
            ((double *)buffer.bias0_1)[j] += ((double *)deltas.bias0_1)[j];
        }
        for (j = 0; j < sizeof(buffer.weight2_3) / sizeof(double); ++j) {
            ((double *)buffer.weight2_3)[j] += ((double *)deltas.weight2_3)[j];
        }
        for (j = 0; j < sizeof(buffer.bias2_3) / sizeof(double); ++j) {
            ((double *)buffer.bias2_3)[j] += ((double *)deltas.bias2_3)[j];
        }
        for (j = 0; j < sizeof(buffer.weight4_5) / sizeof(double); ++j) {
            ((double *)buffer.weight4_5)[j] += ((double *)deltas.weight4_5)[j];
        }
        for (j = 0; j < sizeof(buffer.bias4_5) / sizeof(double); ++j) {
            ((double *)buffer.bias4_5)[j] += ((double *)deltas.bias4_5)[j];
        }
        for (j = 0; j < sizeof(buffer.weight5_6) / sizeof(double); ++j) {
            ((double *)buffer.weight5_6)[j] += ((double *)deltas.weight5_6)[j];
        }
        for (j = 0; j < sizeof(buffer.bias5_6) / sizeof(double); ++j) {
            ((double *)buffer.bias5_6)[j] += ((double *)deltas.bias5_6)[j];
        }
    }

    double k = ALPHA / batchSize;
    // Update the original weights and biases with buffer values
    for (j = 0; j < sizeof(buffer.weight0_1) / sizeof(double); ++j) {
        ((double *)weight0_1)[j] += k * ((double *)buffer.weight0_1)[j];
    }
    for (j = 0; j < sizeof(buffer.bias0_1) / sizeof(double); ++j) {
        ((double *)bias0_1)[j] += k * ((double *)buffer.bias0_1)[j];
    }
    for (j = 0; j < sizeof(buffer.weight2_3) / sizeof(double); ++j) {
        ((double *)weight2_3)[j] += k * ((double *)buffer.weight2_3)[j];
    }
    for (j = 0; j < sizeof(buffer.bias2_3) / sizeof(double); ++j) {
        ((double *)bias2_3)[j] += k * ((double *)buffer.bias2_3)[j];
    }
    for (j = 0; j < sizeof(buffer.weight4_5) / sizeof(double); ++j) {
        ((double *)weight4_5)[j] += k * ((double *)buffer.weight4_5)[j];
    }
    for (j = 0; j < sizeof(buffer.bias4_5) / sizeof(double); ++j) {
        ((double *)bias4_5)[j] += k * ((double *)buffer.bias4_5)[j];
    }
    for (j = 0; j < sizeof(buffer.weight5_6) / sizeof(double); ++j) {
        ((double *)weight5_6)[j] += k * ((double *)buffer.weight5_6)[j];
    }
    for (j = 0; j < sizeof(buffer.bias5_6) / sizeof(double); ++j) {
        ((double *)bias5_6)[j] += k * ((double *)buffer.bias5_6)[j];
    }
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

void Initial(LeNet5 *lenet) {
    for (float *pos = (float *)lenet->weight0_1; pos < (float *)lenet->bias0_1; *pos++ = f64rand());
    for (float *pos = (float *)lenet->weight0_1; pos < (float *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    for (float *pos = (float *)lenet->weight2_3; pos < (float *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    for (float *pos = (float *)lenet->weight4_5; pos < (float *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    for (float *pos = (float *)lenet->weight5_6; pos < (float *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
    for (float *pos = (float *)lenet->bias0_1; pos < (float *)lenet->bias5_6 + OUTPUT; *pos++ = 0.0);
}