#include <fstream>
#include <cstdlib>
#include <ctime>
#include <unistd.h>  // Include this for the access function and F_OK macro
#include "lenet.h"

#define FILE_TRAIN_IMAGE        "../train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL        "../train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE         "../t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL         "../t10k-labels-idx1-ubyte"
#define LENET_FILE              "model.dat"
#define COUNT_TRAIN             60000
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

void save_model(LeNet5 *lenet, const char* filename) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file for writing.\n");
        return;
    }
    fwrite(lenet, sizeof(LeNet5), 1, file);
    fclose(file);
    printf("Model saved to %s\n", filename);
}

void load_model(LeNet5 *lenet, const char* filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("Error opening file for reading.\n");
        return;
    }
    fread(lenet, sizeof(LeNet5), 1, file);
    fclose(file);
    printf("Model loaded from %s\n", filename);
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
    for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
    {
        TrainBatch(
            lenet->weight0_1, lenet->bias0_1,
            lenet->weight2_3, lenet->bias2_3,
            lenet->weight4_5, lenet->bias4_5,
            lenet->weight5_6, lenet->bias5_6,
            train_data + i, train_label + i, batch_size);

        if (i * 100 / total_size > percent)
            printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
    }
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

void foo()
{
    const int count_train = COUNT_TRAIN;
    const int count_test = COUNT_TEST;

    static image train_data[count_train];
    static uint8 train_label[count_train];
    static image test_data[count_test];
    static uint8 test_label[count_test];
    if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
    {
        printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the exe\n");
        system("pause");
    }
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
    {
        printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Folder Included the exe\n");
        system("pause");
    }

    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    Initial(lenet);

    // Load model if exists
    if (access(LENET_FILE, F_OK) == 0) {
        load_model(lenet, LENET_FILE);
    } else {
        clock_t start = clock();
        int batches[] = { 300 };
        for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i)
            training(lenet, train_data, train_label, batches[i], COUNT_TRAIN);
        save_model(lenet, LENET_FILE);
        printf("Time:%u\n", (unsigned)(clock() - start));
    }

    int right = testing(lenet, test_data, test_label, count_test);
    printf("%d/%d\n", right, COUNT_TEST);
    free(lenet);
    system("pause");
}

int main()
{
    foo();
    return 0;
}