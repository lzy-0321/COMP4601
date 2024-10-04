#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cstdio>

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

// Global variables to track the maximum number of digits
int global_max_digit = 0;
int max_integer_digits = 0;
int max_fractional_digits = 0;
double max = 0;
double temp = 0;

// Function to find the maximum digit in a number
int max_digit(double num) {
    num = fabs(num);
    int max_d = 0;
    while (num >= 1) {
        int digit = static_cast<int>(num) % 10;
        if (digit > max_d) {
            max_d = digit;
        }
        num /= 10;
    }
    return max_d;
}

// Function to count the number of digits in the integer part
int count_integer_digits(double num) {
    num = fabs(num);
    int count = 0;
    while (num >= 1) {
        num /= 10;
        count++;
    }
    return count;
}

// Function to count the number of digits in the fractional part
int count_fractional_digits(double num) {
    num = fabs(num);
    int count = 0;
    double frac_part = num - static_cast<int>(num);
    while (frac_part > 0 && count < 10) { // Limiting to 10 decimal places for practical reasons
        frac_part *= 10;
        frac_part -= static_cast<int>(frac_part);
        count++;
    }
    return count;
}

static void update_global_max_digit(double value) {
    int digit = max_digit(value);
    if (digit > global_max_digit) {
        global_max_digit = digit;
        max = value;
    }
    
    int integer_digits = count_integer_digits(value);
    int fractional_digits = count_fractional_digits(value);
    
    if (integer_digits > max_integer_digits) {
        max_integer_digits = integer_digits;
    }
    if (fractional_digits > max_fractional_digits) {
        max_fractional_digits = fractional_digits;
    }
}

double relu(double x)
{
	return x*(x > 0);
}

static void convolution_forward(double input[1][32][32], double output[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], double weight[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER1]) {
    for (int x = 0; x < INPUT; ++x) {
		for (int y = 0; y < LAYER1; ++y) {
			// convolute_valid(input[x], output[y], weight[x][y]);
            for (int o0 = 0; o0 < LENGTH_FEATURE1; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE1; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] += (input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1];
                            update_global_max_digit(output[y][o0][o1]);
                            update_global_max_digit((input[x])[o0 + w0][o1 + w1]);
                            update_global_max_digit((weight[x][y])[w0][w1]);
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER1; j++) {
        for (int i = 0; i < LENGTH_FEATURE1 * LENGTH_FEATURE1; i++) {
            update_global_max_digit(bias[j]);
            update_global_max_digit(((double *)output[j])[i]);
            ((double *)output[j])[i] = relu(((double *)output[j])[i] + bias[j]);
            update_global_max_digit(((double *)output[j])[i]);
        }
    }
}

static void convolution_forward2(double input[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2], double output[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], double weight[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER3]) {
    for (int x = 0; x < LAYER2; ++x) {
		for (int y = 0; y < LAYER3; ++y) {
			// convolute_valid(input[x], output[y], weight[x][y]);
            for (int o0 = 0; o0 < LENGTH_FEATURE3; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE3; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] += (input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1];
                            update_global_max_digit(output[y][o0][o1]);
                            update_global_max_digit((input[x])[o0 + w0][o1 + w1]);
                            update_global_max_digit((weight[x][y])[w0][w1]);
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER3; j++) {
        for (int i = 0; i < LENGTH_FEATURE3 * LENGTH_FEATURE3; i++) {
            update_global_max_digit(bias[j]);
            update_global_max_digit(((double *)output[j])[i]);
            ((double *)output[j])[i] = relu(((double *)output[j])[i] + bias[j]);
            update_global_max_digit(((double *)output[j])[i]);

        }
    }
}

static void convolution_forward3(double input[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4], double output[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], double weight[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL], double bias[LAYER5]) {
    for (int x = 0; x < LAYER4; ++x) {
		for (int y = 0; y < LAYER5; ++y) {
			// convolute_valid(input[x], output[y], weight[x][y]);
            for (int o0 = 0; o0 < LENGTH_FEATURE5; o0++) {
                for (int o1 = 0; o1 < LENGTH_FEATURE5; o1++) {
                    for (int w0 = 0; w0 < LENGTH_KERNEL; w0++) {
                        for (int w1 = 0; w1 < LENGTH_KERNEL; w1++) {
                            (output[y])[o0][o1] += (input[x])[o0 + w0][o1 + w1] * (weight[x][y])[w0][w1];
                            update_global_max_digit(output[y][o0][o1]);
                            update_global_max_digit((input[x])[o0 + w0][o1 + w1]);
                            update_global_max_digit((weight[x][y])[w0][w1]);
                        }
                    }
                }
            }
        }
    }
    for (int j = 0; j < LAYER5; j++) {
        for (int i = 0; i < LENGTH_FEATURE5 * LENGTH_FEATURE5; i++) {
            update_global_max_digit(bias[j]);
            update_global_max_digit(((double *)output[j])[i]);
            ((double *)output[j])[i] = relu(((double *)output[j])[i] + bias[j]);
            update_global_max_digit(((double *)output[j])[i]);
        }
    }
}

static void maxpool_forward(double input[LAYER1][LENGTH_FEATURE1][LENGTH_FEATURE1], double output[LAYER2][LENGTH_FEATURE2][LENGTH_FEATURE2]) {
    const int len0 = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    const int len1 = LENGTH_FEATURE1 / LENGTH_FEATURE2;
    for (int i = 0; i < LAYER2; i++) {
        for (int o0 = 0; o0 < LENGTH_FEATURE2; o0++) {
            for (int o1 = 0; o1 < LENGTH_FEATURE2; o1++) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; l0++) {
                    for (int l1 = 0; l1 < len1; l1++) {
                        update_global_max_digit(input[i][o0*len0 + l0][o1*len1 + l1]);
                        update_global_max_digit(input[i][o0*len0 + x0][o1*len1 + x1]);
                        ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];
                update_global_max_digit(output[i][o0][o1]);
                update_global_max_digit(input[i][o0*len0 + x0][o1*len1 + x1]);
            }
        }
    }
}

static void maxpool_forward2(double input[LAYER3][LENGTH_FEATURE3][LENGTH_FEATURE3], double output[LAYER4][LENGTH_FEATURE4][LENGTH_FEATURE4]) {
    const int len0 = LENGTH_FEATURE3 / LENGTH_FEATURE4;
    const int len1 = LENGTH_FEATURE3 / LENGTH_FEATURE4;
    for (int i = 0; i < LAYER2; i++) {
        for (int o0 = 0; o0 < LENGTH_FEATURE4; o0++) {
            for (int o1 = 0; o1 < LENGTH_FEATURE4; o1++) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; l0++) {
                    for (int l1 = 0; l1 < len1; l1++) {
                        update_global_max_digit(input[i][o0*len0 + l0][o1*len1 + l1]);
                        update_global_max_digit(input[i][o0*len0 + x0][o1*len1 + x1]);
                        ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                }
                output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];
                update_global_max_digit(output[i][o0][o1]);
                update_global_max_digit(input[i][o0*len0 + x0][o1*len1 + x1]);
            }
        }
    }
}

static void fc_forward(double input[LAYER5][LENGTH_FEATURE5][LENGTH_FEATURE5], double output[OUTPUT], double weight[LAYER5 * LENGTH_FEATURE5 * LENGTH_FEATURE5][OUTPUT], double bias[OUTPUT]) {
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
                    update_global_max_digit(output[y]);
                    update_global_max_digit(input[i][j][k]);
                    update_global_max_digit(weight[idx][y]);
                }
            }
        }
    }

    // Apply bias and activation function
    for (int j = 0; j < OUTPUT; j++) {
        update_global_max_digit(output[j]);
        output[j] = relu(output[j] + bias[j]);
        update_global_max_digit(bias[j]);
        update_global_max_digit(output[j]);

    }
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
    printf("%d/%d\n", max_integer_digits, max_fractional_digits);
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

static uint8 get_result(Feature *features, uint8 count)
{
    double *output = (double *)features->output;
    uint8 result = 0;
    double maxvalue = *output;
    for (uint8 i = 1; i < count; ++i)
    {
        update_global_max_digit(output[i]);
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
