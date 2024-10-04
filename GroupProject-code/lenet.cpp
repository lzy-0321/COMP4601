#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cstdint>
#include <limits>
#include <cassert>

typedef int64_t fixed_point_t; // 使用更大的整数类型

const fixed_point_t FIXED_POINT_MAX = std::numeric_limits<int64_t>::max();
const fixed_point_t FIXED_POINT_MIN = std::numeric_limits<int64_t>::min();

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(fixed_point_t))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)


fixed_point_t relu(fixed_point_t x)
{
	return x * (x > 0);
}

fixed_point_t relugrad(fixed_point_t y) {
    return y > 0 ? 1 : 0;
}

fixed_point_t double_to_fixed(double value) {
    return static_cast<fixed_point_t>(value * (1 << FRACTIONAL_BITS));
}

double fixed_to_double(fixed_point_t value) {
    return static_cast<double>(value) / (1 << FRACTIONAL_BITS);
}

fixed_point_t fixed_add(fixed_point_t a, fixed_point_t b) {
    int64_t result = static_cast<int64_t>(a) + static_cast<int64_t>(b);
    if (result > FIXED_POINT_MAX || result < FIXED_POINT_MIN) {
        return (result > FIXED_POINT_MAX) ? FIXED_POINT_MAX : FIXED_POINT_MIN;
    }
    return static_cast<fixed_point_t>(result);
}

fixed_point_t fixed_sub(fixed_point_t a, fixed_point_t b) {
    int64_t result = static_cast<int64_t>(a) - static_cast<int64_t>(b);
    if (result > FIXED_POINT_MAX || result < FIXED_POINT_MIN) {
        return (result > FIXED_POINT_MAX) ? FIXED_POINT_MAX : FIXED_POINT_MIN;
    }
    return static_cast<fixed_point_t>(result);
}

fixed_point_t fixed_mul(fixed_point_t a, fixed_point_t b) {
    int64_t result = static_cast<int64_t>(a) * static_cast<int64_t>(b);
    if (result > FIXED_POINT_MAX || result < FIXED_POINT_MIN) {
        return (result > FIXED_POINT_MAX) ? FIXED_POINT_MAX : FIXED_POINT_MIN;
    }
    return static_cast<fixed_point_t>(result >> FRACTIONAL_BITS);
}

fixed_point_t fixed_div(fixed_point_t a, fixed_point_t b) {
    if (b == 0) {
        return (a >= 0) ? FIXED_POINT_MAX : FIXED_POINT_MIN;
    }
    int64_t result = (static_cast<int64_t>(a) << FRACTIONAL_BITS) / b;
    if (result > FIXED_POINT_MAX || result < FIXED_POINT_MIN) {
        return (result > FIXED_POINT_MAX) ? FIXED_POINT_MAX : FIXED_POINT_MIN;
    }
    return static_cast<fixed_point_t>(result);
}

fixed_point_t fixed_sqrt(fixed_point_t x) {
    return double_to_fixed(sqrt(fixed_to_double(x)));
}

fixed_point_t fixed_exp(fixed_point_t x) {
    return double_to_fixed(exp(fixed_to_double(x)));
}

static void convolution_forward(fixed_point_t input[1][32][32], fixed_point_t output[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], fixed_point_t weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t bias[LAYER1]) {
    for (int x = 0; x < INPUT; ++x) {
		for (int y = 0; y < LAYER1; ++y) {
			// convolute_valid(input[x], output[y], weight[x][y]);
            for (int o0 = 0; o0 < LENGTH_FEATURE1; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE1; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] = fixed_add((output[y])[o0][o1], fixed_mul((input[x])[o0 + w0][o1 + w1], (weight[x][y])[w0][w1]));
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER1; j++) {
        for (int i = 0; i < LENGTH_FEATURE1 * LENGTH_FEATURE1; i++) {
            ((fixed_point_t *)output[j])[i] = relu(fixed_add(((fixed_point_t *)output[j])[i], bias[j]));
        }
    }
}

static void convolution_forward2(fixed_point_t input[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], fixed_point_t output[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], fixed_point_t weight[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t bias[LAYER3]) {
    for (int x = 0; x < LAYER2; ++x) {									
		for (int y = 0; y < LAYER3; ++y) {							
			// convolute_valid(input[x], output[y], weight[x][y]);
            for (int o0 = 0; o0 < LENGTH_FEATURE3; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE3; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] = fixed_add((output[y])[o0][o1], fixed_mul((input[x])[o0 + w0][o1 + w1], (weight[x][y])[w0][w1]));
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER3; j++) {
        for (int i = 0; i < LENGTH_FEATURE3 * LENGTH_FEATURE3; i++) {
            ((fixed_point_t *)output[j])[i] = relu(fixed_add(((fixed_point_t *)output[j])[i], bias[j]));
        }
    }
}

static void convolution_forward3(fixed_point_t input[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], fixed_point_t output[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], fixed_point_t weight[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t bias[LAYER5]) {
    for (int x = 0; x < LAYER4; ++x) {									
		for (int y = 0; y < LAYER5; ++y) {							
			// convolute_valid(input[x], output[y], weight[x][y]);
            for (int o0 = 0; o0 < LENGTH_FEATURE5; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE5; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] = fixed_add((output[y])[o0][o1], fixed_mul((input[x])[o0 + w0][o1 + w1], (weight[x][y])[w0][w1]));
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER5; j++) {
        for (int i = 0; i < LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
            ((fixed_point_t *)output[j])[i] = relu(fixed_add(((fixed_point_t *)output[j])[i], bias[j]));
        }
    }
}


static void convolution_backward(fixed_point_t input[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0], fixed_point_t inerror[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0], fixed_point_t outerror[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], fixed_point_t weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t wd[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t bd[LAYER1]) {
    for (int x = 0; x < INPUT; ++x) {
        for (int y = 0; y < LAYER1; ++y) {
            for (int i0 = 0; i0 < LENGTH_FEATURE1; i0++) {
                for (int i1 = 0; i1 < LENGTH_FEATURE1; i1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            inerror[x][i0 + w0][i1 + w1] = fixed_add(inerror[x][i0 + w0][i1 + w1], fixed_mul(outerror[y][i0][i1], weight[x][y][w0][w1]));
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < INPUT * LENGTH_FEATURE0 * LENGTH_FEATURE0; i++) {
        ((fixed_point_t *)inerror)[i] = fixed_mul(((fixed_point_t *)inerror)[i], relugrad(((fixed_point_t *)input)[i]));
    }
    for (int j = 0; j < LAYER1; j++) {
        for (int i = 0; i < LENGTH_FEATURE1 * LENGTH_FEATURE1; i++) {
            bd[j] = fixed_add(bd[j], ((fixed_point_t *)outerror[j])[i]);
        }
    }
    for (int x = 0; x < INPUT; x++) {
        for (int y = 0; y < LAYER1; y++) {
            for (int o0 = 0; o0 < LENGTH_KERNEL; o0++) {
                for (int o1 = 0; o1 < LENGTH_KERNEL; o1++) {
                    for (int w0 = 0; w0 < LENGTH_FEATURE1; w0++) {
                        for (int w1 = 0; w1 < LENGTH_FEATURE1; w1++) {
                            (wd[x][y])[o0][o1] = fixed_add((wd[x][y])[o0][o1], fixed_mul((input[x])[o0 + w0][o1 + w1], (outerror[y])[w0][w1]));
                        }
                    }
                }
            }
        }
    }
}

static void convolution_backward2(fixed_point_t input[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], fixed_point_t inerror[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], fixed_point_t outerror[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], fixed_point_t weight[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t wd[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t bd[LAYER3]) {
    for (int x = 0; x < LAYER2; ++x) {
        for (int y = 0; y < LAYER3; ++y) {
            for (int i0 = 0; i0 < LENGTH_FEATURE3; i0++) {
                for (int i1 = 0; i1 < LENGTH_FEATURE3; i1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            int index0 = i0 + w0;
                            int index1 = i1 + w1;
                            assert(index0 < LENGTH_FEATURE2 && index1 < LENGTH_FEATURE2); // 断言捕获越界访问
                            if (index0 >= LENGTH_FEATURE2 || index1 >= LENGTH_FEATURE2) {
                                std::cerr << "Index out of bounds: index0 = " << index0 << ", index1 = " << index1 << std::endl;
                                continue;
                            }
                            inerror[x][index0][index1] = fixed_add(inerror[x][index0][index1], fixed_mul(outerror[y][i0][i1], weight[x][y][w0][w1]));
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < LAYER2 * LENGTH_FEATURE2 * LENGTH_FEATURE2; i++) {
        ((fixed_point_t *)inerror)[i] = fixed_mul(((fixed_point_t *)inerror)[i], relugrad(((fixed_point_t *)input)[i]));
    }
    for (int j = 0; j < LAYER3; j++) {
        for (int i = 0; i < LENGTH_FEATURE3 * LENGTH_FEATURE3; i++) {
            bd[j] = fixed_add(bd[j], ((fixed_point_t *)outerror[j])[i]);
        }
    }
    for (int x = 0; x < LAYER2; x++) {
        for (int y = 0; y < LAYER3; y++) {
            for (int o0 = 0; o0 < LENGTH_KERNEL; o0++) {
                for (int o1 = 0; o1 < LENGTH_KERNEL; o1++) {
                    for (int w0 = 0; w0 < LENGTH_FEATURE3; w0++) {
                        for (int w1 = 0; w1 < LENGTH_FEATURE3; w1++) {
                            int index0 = o0 + w0;
                            int index1 = o1 + w1;
                            assert(index0 < LENGTH_FEATURE2 && index1 < LENGTH_FEATURE2); // 断言捕获越界访问
                            if (index0 >= LENGTH_FEATURE2 || index1 >= LENGTH_FEATURE2) {
                                std::cerr << "Index out of bounds: index0 = " << index0 << ", index1 = " << index1 << std::endl;
                                continue;
                            }
                            (wd[x][y])[o0][o1] = fixed_add((wd[x][y])[o0][o1], fixed_mul((input[x])[index0][index1], (outerror[y])[w0][w1]));
                        }
                    }
                }
            }
        }
    }
}

static void convolution_backward3(fixed_point_t input[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], fixed_point_t inerror[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], fixed_point_t outerror[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], fixed_point_t weight[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t wd[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], fixed_point_t bd[LAYER5]) {
    for (int x = 0; x < LAYER4; ++x) {
        for (int y = 0; y < LAYER5; ++y) {
            for (int i0 = 0; i0 < LENGTH_FEATURE5; i0++) {
                for (int i1 = 0; i1 < LENGTH_FEATURE5; i1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            int index0 = i0 + w0;
                            int index1 = i1 + w1;
                            assert(index0 < LENGTH_FEATURE4 && index1 < LENGTH_FEATURE4); // Assert to catch out-of-bounds
                            inerror[x][index0][index1] = fixed_add(inerror[x][index0][index1], fixed_mul(outerror[y][i0][i1], weight[x][y][w0][w1]));
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < LAYER4 * LENGTH_FEATURE4 * LENGTH_FEATURE4; i++) {
        ((fixed_point_t *)inerror)[i] = fixed_mul(((fixed_point_t *)inerror)[i], relugrad(((fixed_point_t *)input)[i]));
    }
    for (int j = 0; j < LAYER5; j++) {
        for (int i = 0; i < LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
            bd[j] = fixed_add(bd[j], ((fixed_point_t *)outerror[j])[i]);
        }
    }
    for (int x = 0; x < LAYER4; x++) {
        for (int y = 0; y < LAYER5; y++) {
            for (int o0 = 0; o0 < LENGTH_KERNEL; o0++) {
                for (int o1 = 0; o1 < LENGTH_KERNEL; o1++) {
                    for (int w0 = 0; w0 < LENGTH_FEATURE5; w0++) {
                        for (int w1 = 0; w1 < LENGTH_FEATURE5; w1++) {
                            int index0 = o0 + w0;
                            int index1 = o1 + w1;
                            assert(index0 < LENGTH_FEATURE4 && index1 < LENGTH_FEATURE4); // Assert to catch out-of-bounds
                            (wd[x][y])[o0][o1] = fixed_add((wd[x][y])[o0][o1], fixed_mul((input[x])[index0][index1], (outerror[y])[w0][w1]));
                        }
                    }
                }
            }
        }
    }
}

static void maxpool_forward(fixed_point_t input[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], fixed_point_t output[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2]) {
    const int len0 = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    const int len1 = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    for (int i = 0; i < LAYER2; i++) {
        for (int o0 = 0; o0 < LENGTH_FEATURE2; o0++) {
            for (int o1 = 0; o1 < LENGTH_FEATURE2; o1++) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; l0++) {
                    for (int l1 = 0; l1 < len1; l1++) {
                        ismax = input[i][o0 * len0 + l0][o1 * len1 + l1] > input[i][o0 * len0 + x0][o1 * len1 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                output[i][o0][o1] = input[i][o0 * len0 + x0][o1 * len1 + x1];
            }
        }
    }
}

static void maxpool_forward2(fixed_point_t input[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], fixed_point_t output[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4]) {
    const int len0 = LENGTH_FEATURE3 / LENGTH_FEATURE4;
    const int len1 = LENGTH_FEATURE3 / LENGTH_FEATURE4;
    for (int i = 0; i < LAYER4; i++) {
        for (int o0 = 0; o0 < LENGTH_FEATURE4; o0++) {
            for (int o1 = 0; o1 < LENGTH_FEATURE4; o1++) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; l0++) {
                    for (int l1 = 0; l1 < len1; l1++) {
                        ismax = input[i][o0 * len0 + l0][o1 * len1 + l1] > input[i][o0 * len0 + x0][o1 * len1 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                output[i][o0][o1] = input[i][o0 * len0 + x0][o1 * len1 + x1];
            }
        }
    }
}


static void maxpool_backward(fixed_point_t input[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], fixed_point_t inerror[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], fixed_point_t outerror[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2]) {
    const int len0 = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    const int len1 = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    for (int i = 0; i < LAYER2; i++) {
        for (int o0 = 0; o0 < LENGTH_FEATURE2; o0++) {
            for (int o1 = 0; o1 < LENGTH_FEATURE2; o1++) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; l0++) {
                    for (int l1 = 0; l1 < len1; l1++) {
                        ismax = input[i][o0 * len0 + l0][o1 * len1 + l1] > input[i][o0 * len0 + x0][o1 * len1 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                inerror[i][o0 * len0 + x0][o1 * len1 + x1] = outerror[i][o0][o1];
            }
        }
    }
}

static void maxpool_backward2(fixed_point_t input[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], fixed_point_t inerror[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], fixed_point_t outerror[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4]) {
    const int len0 = LENGTH_FEATURE3 / LENGTH_FEATURE4;
    const int len1 = LENGTH_FEATURE3 / LENGTH_FEATURE4;
    for (int i = 0; i < LAYER4; i++) {
        for (int o0 = 0; o0 < LENGTH_FEATURE4; o0++) {
            for (int o1 = 0; o1 < LENGTH_FEATURE4; o1++) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; l0++) {
                    for (int l1 = 0; l1 < len1; l1++) {
                        ismax = input[i][o0 * len0 + l0][o1 * len1 + l1] > input[i][o0 * len0 + x0][o1 * len1 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                inerror[i][o0 * len0 + x0][o1 * len1 + x1] = outerror[i][o0][o1];
            }
        }
    }
}



static void fc_forward(fixed_point_t input[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], fixed_point_t output[OUTPUT], fixed_point_t weight[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], fixed_point_t bias[OUTPUT]) {
    for (int x = 0; x < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; x++) {
        for (int y = 0; y < OUTPUT; y++) {
            ((fixed_point_t *)output)[y] = fixed_add(((fixed_point_t *)output)[y], fixed_mul(((fixed_point_t *)input)[x], weight[x][y]));
        }
    }
    for (int j = 0; j < OUTPUT; j++) {
        ((fixed_point_t *)output)[j] = relu(fixed_add(((fixed_point_t *)output)[j], bias[j]));
    }
}


static void fc_backward(fixed_point_t input[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], fixed_point_t inerror[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], fixed_point_t outerror[OUTPUT], fixed_point_t weight[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], fixed_point_t wd[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], fixed_point_t bd[OUTPUT]) {
    for (int x = 0; x < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; x++) {
        for (int y = 0; y < OUTPUT; y++) {
            ((fixed_point_t *)inerror)[x] = fixed_add(((fixed_point_t *)inerror)[x], fixed_mul(((fixed_point_t *)outerror)[y], weight[x][y]));
        }
    }
    for (int i = 0; i < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
        ((fixed_point_t *)inerror)[i] = fixed_mul(((fixed_point_t *)inerror)[i], relugrad(((fixed_point_t *)input)[i]));
    }
    for (int j = 0; j < OUTPUT; j++) {
        bd[j] = fixed_add(bd[j], ((fixed_point_t *)outerror)[j]);
    }
    for (int x = 0; x < LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5; x++) {
        int layer = x / (LENGTH_FEATURE5 * LENGTH_FEATURE5);
        int row = (x / LENGTH_FEATURE5) % LENGTH_FEATURE5;
        int col = x % LENGTH_FEATURE5;
        for (int y = 0; y < OUTPUT; ++y) {
            wd[x][y] = fixed_add(wd[x][y], fixed_mul(input[layer][row][col], outerror[y]));
        }
    }
}

static void forward(LeNet5 *lenet, Feature *features, fixed_point_t(*action)(fixed_point_t))
{
    convolution_forward(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1);
    maxpool_forward(features->layer1, features->layer2);
    convolution_forward2(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3);
    maxpool_forward2(features->layer3, features->layer4);
    convolution_forward3(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5);
    fc_forward(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, fixed_point_t(*actiongrad)(fixed_point_t))
{
    fc_backward(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6);
    convolution_backward3(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5);
    maxpool_backward2(features->layer3, errors->layer3, errors->layer4);
    convolution_backward2(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3);
    maxpool_backward(features->layer1, errors->layer1, errors->layer2);
    convolution_backward(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1);
}

static inline void load_input(Feature *features, image input)
{
	fixed_point_t (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	fixed_point_t mean = 0, std = 0;

    for (int j = 0; j < sizeof(image) / sizeof(*input); j++) {
        for (int k = 0; k < sizeof(*input) / sizeof(**input); k++) {
            mean = fixed_add(mean, double_to_fixed(input[j][k]));
            std = fixed_add(std, fixed_mul(double_to_fixed(input[j][k]), double_to_fixed(input[j][k])));
        }
    }
	mean = fixed_div(mean, double_to_fixed(sz));
	std = fixed_sqrt(fixed_sub(fixed_div(std, double_to_fixed(sz)), fixed_mul(mean, mean)));

    for (int j = 0; j < sizeof(image) / sizeof(*input); j++) {
        for (int k = 0; k < sizeof(*input) / sizeof(**input); k++) {
            layer0[0][j + PADDING][k + PADDING] = fixed_div(fixed_sub(double_to_fixed(input[j][k]), mean), std);
        }
    }
}

static inline void softmax(fixed_point_t input[OUTPUT], fixed_point_t loss[OUTPUT], int label, int count)
{
	fixed_point_t inner = 0;
	for (int i = 0; i < count; ++i)
	{
		fixed_point_t res = 0;
		for (int j = 0; j < count; ++j)
		{
			res = fixed_add(res, fixed_exp(fixed_sub(input[j], input[i])));
		}
		loss[i] = fixed_div(double_to_fixed(1), res);
		inner = fixed_sub(inner, fixed_mul(loss[i], loss[i]));
	}
	inner = fixed_add(inner, loss[label]);
	for (int i = 0; i < count; ++i)
	{
		loss[i] = fixed_mul(loss[i], fixed_sub((i == label) ? double_to_fixed(1) : double_to_fixed(0), fixed_add(loss[i], inner)));
	}
}

static void load_target(Feature *features, Feature *errors, int label)
{
	fixed_point_t *output = (fixed_point_t *)features->output;
	fixed_point_t *error = (fixed_point_t *)errors->output;
	softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
	fixed_point_t *output = (fixed_point_t *)features->output; 
	const int outlen = GETCOUNT(features->output);
	uint8 result = 0;
	fixed_point_t maxvalue = *output;
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

static double f64rand()
{
	static int randbit = 0;
	if (!randbit)
	{
		srand((unsigned)time(0));
		for (int i = RAND_MAX; i; i >>= 1, ++randbit);
	}
	unsigned long long lvalue = 0x4000000000000000L;
	int i = 52 - randbit;
	for (; i > 0; i -= randbit)
		lvalue |= (unsigned long long)rand() << i;
	lvalue |= (unsigned long long)rand() >> -i;
	return *(double *)&lvalue - 3;
}

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	fixed_point_t buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		load_target(&features, &errors, labels[i]);
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
            for (int j = 0; j < GETCOUNT(LeNet5); j++) {
                buffer[j] = fixed_add(buffer[j], ((fixed_point_t *)&deltas)[j]);
            }
		}
	}
	fixed_point_t k = fixed_div(double_to_fixed(ALPHA), double_to_fixed(batchSize));

    for (int i = 0; i < GETCOUNT(LeNet5); i++) {
        ((fixed_point_t *)lenet)[i] = fixed_add(((fixed_point_t *)lenet)[i], fixed_mul(k, buffer[i]));
    }
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	load_target(&features, &errors, label);
	backward(lenet, &deltas, &errors, &features, relugrad);

    for (int i = 0; i < GETCOUNT(LeNet5); i++) {
        ((fixed_point_t *)lenet)[i] = fixed_add(((fixed_point_t *)lenet)[i], fixed_mul(double_to_fixed(ALPHA), ((fixed_point_t *)&deltas)[i]));
    }
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
    for (fixed_point_t *pos = (fixed_point_t *)lenet->weight0_1; pos < (fixed_point_t *)lenet->bias0_1; *pos++ = double_to_fixed(f64rand()));
    for (fixed_point_t *pos = (fixed_point_t *)lenet->weight0_1; pos < (fixed_point_t *)lenet->weight2_3; *pos++ = fixed_mul(*pos, double_to_fixed(sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))))));
    for (fixed_point_t *pos = (fixed_point_t *)lenet->weight2_3; pos < (fixed_point_t *)lenet->weight4_5; *pos++ = fixed_mul(*pos, double_to_fixed(sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))))));
    for (fixed_point_t *pos = (fixed_point_t *)lenet->weight4_5; pos < (fixed_point_t *)lenet->weight5_6; *pos++ = fixed_mul(*pos, double_to_fixed(sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))))));
    for (fixed_point_t *pos = (fixed_point_t *)lenet->weight5_6; pos < (fixed_point_t *)lenet->bias0_1; *pos++ = fixed_mul(*pos, double_to_fixed(sqrt(6.0 / (LAYER5 + OUTPUT)))));
    for (fixed_point_t *pos = (fixed_point_t *)lenet->bias0_1; pos < (fixed_point_t *)(lenet + 1); *pos++ = double_to_fixed(0.0));
}