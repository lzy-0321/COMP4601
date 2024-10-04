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
$(TARGET): main.cpp lenet.cpp lenet.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp lenet.cpp

# 清理
clean:
	rm -f $(TARGET)




#############
#############
# for testbench
#############
#############
# 定义编译器和编译器标志
# CXX = g++
# CXXFLAGS = -Wall -O2 -std=c++11

# # 定义目标文件和可执行文件
# TARGET = lenet_test
# OBJS = lenet.o main.o testbench.o

# # 默认目标
# all: $(TARGET)

# # 链接目标文件生成可执行文件
# $(TARGET): $(OBJS)
# 	$(CXX) $(OBJS) -o $(TARGET)

# # 生成对象文件
# lenet.o: lenet.cpp lenet.h
# 	$(CXX) $(CXXFLAGS) -c lenet.cpp -o lenet.o

# main.o: main.cpp lenet.h
# 	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

# testbench.o: testbench.cpp lenet.h
# 	$(CXX) $(CXXFLAGS) -c testbench.cpp -o testbench.o

# # 清理生成的文件
# clean:
# 	rm -f $(OBJS) $(TARGET)
