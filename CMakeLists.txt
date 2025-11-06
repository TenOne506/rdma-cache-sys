cmake_minimum_required(VERSION 3.15)
project(RDMACacheSim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# 包含目录
include_directories(${CMAKE_SOURCE_DIR}/include)

# 源文件
set(SOURCES
    src/tokens.cpp
    src/cache_layers.cpp
    src/access_patterns.cpp
    src/migration.cpp
    src/simulator.cpp
    src/main.cpp
)

# 可执行文件
add_executable(rdma_cache_sim ${SOURCES})

# 第三方库（如果需要）
find_package(Threads REQUIRED)
target_link_libraries(rdma_cache_sim Threads::Threads)

# Benchmarks / Experiments
add_executable(bench_token_compress tools/bench_token_compress.cpp src/tokens.cpp)
target_include_directories(bench_token_compress PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(bench_token_compress Threads::Threads)

add_executable(exp_a1_compress tools/exp_a1_compress.cpp src/tokens.cpp)
target_include_directories(exp_a1_compress PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(cache_experiments tools/cache_experiments.cpp src/tokens.cpp src/cache_layers.cpp src/access_patterns.cpp src/migration.cpp src/simulator.cpp)
target_include_directories(cache_experiments PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cache_experiments Threads::Threads)

add_executable(exp_b1_baseline tools/exp_b1_baseline.cpp src/tokens.cpp src/cache_layers.cpp src/access_patterns.cpp src/migration.cpp)
target_include_directories(exp_b1_baseline PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(exp_b1_baseline Threads::Threads)

add_executable(exp_b2_hot_inline tools/exp_b2_hot_inline.cpp src/tokens.cpp src/cache_layers.cpp src/access_patterns.cpp)
target_include_directories(exp_b2_hot_inline PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(exp_b2_hot_inline Threads::Threads)

add_executable(exp_b3_prefetched_batch tools/exp_b3_prefetched_batch.cpp src/tokens.cpp src/cache_layers.cpp src/access_patterns.cpp)
target_include_directories(exp_b3_prefetched_batch PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(exp_b3_prefetched_batch Threads::Threads)

