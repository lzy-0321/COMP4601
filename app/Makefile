#############
#############
# for main
#############
#############

# 定义编译器和编译选项
CXX = g++
CXXFLAGS = -fopenmp

# 定义目标文件
TARGET = lenet

# 默认目标
all: $(TARGET)

# 目标文件的规则
$(TARGET): main.cpp lenet.cpp lenet_model.cpp lenet.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp lenet.cpp lenet_model.cpp

# 清理
clean:
	rm -f $(TARGET)

