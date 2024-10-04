# 新模型训练和测试指南

## 1. 训练新模型
1. 进入 `train` 文件夹。
    ```sh
    cd train
    ```
2. 删除 `model.dat` 文件。
    ```sh
    rm model.dat
    ```
3. 运行 `make`。
    ```sh
    make
    ```
4. 运行 `./lenet`。
    ```sh
    ./lenet
    ```
5. 训练好的新模型会存储在 `model.dat` 文件中。

## 2. 测试模型
1. 进入 `test` 文件夹。
    ```sh
    cd ../test
    ```
2. 运行 `./extract_model` 将第一步生成的 `model.dat` 转换为 `model.parameters.txt`。
    ```sh
    ./extract_model
    ```
3. 打开 `model.parameters.txt`，将其中的参数复制到 `lenet_model.cpp` 文件中。
4. 运行 `make`。
    ```sh
    make
    ```
5. 运行 `./lenet_wsl` 进行参数测试。
    ```sh
    ./lenet_wsl
    ```

## 3. 将模型参数应用到 HLS
1. 从 `test` 文件夹中将 `lenet_model.cpp` 文件复制到 `app` 文件夹中。
    ```sh
    cp test/lenet_model.cpp app/
    ```
2. 在 HLS 中选择 `app` 文件夹中，的所有文件作为源文件。
