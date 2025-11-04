#!/bin/bash

# 构建C++仿真程序

echo "Building RDMA Cache Simulation..."

# 创建构建目录
mkdir -p build
cd build

# 运行CMake
cmake ..

# 编译
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Executable: build/rdma_cache_sim"
else
    echo "Build failed!"
    exit 1
fi

