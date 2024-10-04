#include <iostream>

#include "lenet.h"

int main() {
    // Calculate the number of float elements in the LeNet5 structure
    const int lenet_size = sizeof(LeNet5) / sizeof(float);

    // Print the result
    std::cout << "Number of float elements in LeNet5: " << lenet_size << std::endl;

    return 0;
}