# WSL 编译器和选项
CXX_WSL = g++
CXXFLAGS_WSL = -fopenmp -g -Wall -O2 -std=c++11

# PetaLinux 编译器和选项
CXX_PETALINUX = aarch64-xilinx-linux-g++
SYSROOT = /opt/petalinux/2021.1/sysroots/cortexa72-cortexa53-xilinx-linux
LIBDIR = $(SYSROOT)/usr/lib
CXXFLAGS_PETALINUX = -fopenmp -g -Wall -O2 -std=c++11 --sysroot=$(SYSROOT) -I include/
LDFLAGS_PETALINUX = -L$(LIBDIR) -lpthread -lz -rdynamic

# 目标文件
TARGET_WSL = lenet_wsl
TARGET_PETALINUX = lenet_petalinux

# 源文件和头文件
SRCS = main.cpp lenet.cpp lenet_model.cpp
HEADERS = lenet.h

# 默认目标
all: $(TARGET_WSL) $(TARGET_PETALINUX)

# WSL 目标文件规则
$(TARGET_WSL): $(SRCS) $(HEADERS)
	$(CXX_WSL) $(CXXFLAGS_WSL) -o $(TARGET_WSL) $(SRCS)

# PetaLinux 目标文件规则
$(TARGET_PETALINUX): $(SRCS) $(HEADERS)
	$(CXX_PETALINUX) $(CXXFLAGS_PETALINUX) -o $(TARGET_PETALINUX) $(SRCS) $(LDFLAGS_PETALINUX)

# 清理
clean:
	rm -f $(TARGET_WSL) $(TARGET_PETALINUX)