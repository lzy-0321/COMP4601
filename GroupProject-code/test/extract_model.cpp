#include <iostream>
#include <fstream>
#include "lenet.h"

void print_array(double *array, int size) {
    std::cout << "{";
    for (int i = 0; i < size; ++i) {
        std::cout << array[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "}";
}

void print_weights0_1(double weights[INPUT][LAYER1][LENGTH_KERNEL][LENGTH_KERNEL]) {
    std::cout << "{";
    for (int i = 0; i < INPUT; ++i) {
        std::cout << "{";
        for (int j = 0; j < LAYER1; ++j) {
            std::cout << "{";
            for (int k = 0; k < LENGTH_KERNEL; ++k) {
                print_array(weights[i][j][k], LENGTH_KERNEL);
                if (k < LENGTH_KERNEL - 1) std::cout << ", ";
            }
            std::cout << "}";
            if (j < LAYER1 - 1) std::cout << ", ";
        }
        std::cout << "}";
        if (i < INPUT - 1) std::cout << ", ";
    }
    std::cout << "}";
}

void print_weights2_3(double weights[LAYER2][LAYER3][LENGTH_KERNEL][LENGTH_KERNEL]) {
    std::cout << "{";
    for (int i = 0; i < LAYER2; ++i) {
        std::cout << "{";
        for (int j = 0; j < LAYER3; ++j) {
            std::cout << "{";
            for (int k = 0; k < LENGTH_KERNEL; ++k) {
                print_array(weights[i][j][k], LENGTH_KERNEL);
                if (k < LENGTH_KERNEL - 1) std::cout << ", ";
            }
            std::cout << "}";
            if (j < LAYER3 - 1) std::cout << ", ";
        }
        std::cout << "}";
        if (i < LAYER2 - 1) std::cout << ", ";
    }
    std::cout << "}";
}

void print_weights4_5(double weights[LAYER4][LAYER5][LENGTH_KERNEL][LENGTH_KERNEL]) {
    std::cout << "{";
    for (int i = 0; i < LAYER4; ++i) {
        std::cout << "{";
        for (int j = 0; j < LAYER5; ++j) {
            std::cout << "{";
            for (int k = 0; k < LENGTH_KERNEL; ++k) {
                print_array(weights[i][j][k], LENGTH_KERNEL);
                if (k < LENGTH_KERNEL - 1) std::cout << ", ";
            }
            std::cout << "}";
            if (j < LAYER5 - 1) std::cout << ", ";
        }
        std::cout << "}";
        if (i < LAYER4 - 1) std::cout << ", ";
    }
    std::cout << "}";
}

void print_weights5_6(double weights[LAYER5][OUTPUT]) {
    std::cout << "{";
    for (int i = 0; i < LAYER5; ++i) {
        print_array(weights[i], OUTPUT);
        if (i < LAYER5 - 1) std::cout << ", ";
    }
    std::cout << "}";
}

int main() {
    LeNet5 lenet;

    std::ifstream file("../train/model.dat", std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading.\n";
        return 1;
    }

    file.read(reinterpret_cast<char*>(&lenet), sizeof(LeNet5));
    file.close();

    std::cout << ".weight0_1 = ";
    print_weights0_1(lenet.weight0_1);
    std::cout << ",\n";

    std::cout << ".weight2_3 = ";
    print_weights2_3(lenet.weight2_3);
    std::cout << ",\n";

    std::cout << ".weight4_5 = ";
    print_weights4_5(lenet.weight4_5);
    std::cout << ",\n";

    std::cout << ".weight5_6 = ";
    print_weights5_6(lenet.weight5_6);
    std::cout << ",\n";

    std::cout << ".bias0_1 = ";
    print_array(lenet.bias0_1, LAYER1);
    std::cout << ",\n";

    std::cout << ".bias2_3 = ";
    print_array(lenet.bias2_3, LAYER3);
    std::cout << ",\n";

    std::cout << ".bias4_5 = ";
    print_array(lenet.bias4_5, LAYER5);
    std::cout << ",\n";

    std::cout << ".bias5_6 = ";
    print_array(lenet.bias5_6, OUTPUT);
    std::cout << " //\n";

    return 0;
}