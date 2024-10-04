// #include "lenet.h"

// void PredictLabel(image* input_image, uint8* predicted_label) {
// // #pragma HLS INTERFACE s_axilite port=predicted_label
// // #pragma HLS INTERFACE s_axilite port=input_image
// // #pragma HLS INTERFACE s_axilite port=return
//     // Use the statically initialized lenet object from lenet.h
//     *predicted_label = Predict(
//         lenet.weight0_1, lenet.bias0_1,
//         lenet.weight2_3, lenet.bias2_3,
//         lenet.weight4_5, lenet.bias4_5,
//         lenet.weight5_6, lenet.bias5_6,
//         *input_image, OUTPUT);
// }

#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "lenet.h"

#define FILE_TEST_IMAGE         "../t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL         "../t10k-labels-idx1-ubyte"
#define COUNT_TEST              10000

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image || !fp_label) return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data) * count, 1, fp_image);
    fread(label, count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i)
    {
        uint8 l = test_label[i];
        int p = Predict(
            lenet->weight0_1, lenet->bias0_1,
            lenet->weight2_3, lenet->bias2_3,
            lenet->weight4_5, lenet->bias4_5,
            lenet->weight5_6, lenet->bias5_6,
            test_data[i], 10);
        right += l == p;
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);
    }
    return right;
}

void foo() {
    const int count_test = COUNT_TEST;

    static image test_data[count_test];
    static uint8 test_label[count_test];

    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        printf("ERROR!!!\nDataset File Not Found! Please Copy Dataset to the Folder Included the exe\n");
        system("pause");
    }

    int right = testing(&lenet, test_data, test_label, count_test);
    printf("%d/%d\n", right, COUNT_TEST);

    system("pause");
}

int main() {
    foo();
    return 0;
}