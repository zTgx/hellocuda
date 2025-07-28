cmake_minimum_required(VERSION 3.18)

set(PROJECT_NAME "hellocuda")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# MX150的计算能力是 ​​6.1（Maxwell架构）​​
# 从CUDA 12.x开始，NVIDIA逐步淘汰对 ​​计算能力 < 7.5（Volta+）​​ 的离线编译支持
# nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).

# Add 20208
# /home/ztgx/Desktop/playgrounds/hellocuda/src/main.cu(5): warning #20208-D: 'long double' is treated as 'double' in device code
# Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -diag-suppress 20208")

set(CMAKE_CUDA_ARCHITECTURES "61")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# 显式设置策略
cmake_policy(SET CMP0167 NEW)  # 使用新的Boost查找机制

set(CUTLASSPATH "$ENV{CUTLASSPATH}")
if(NOT CUTLASSPATH)
    set(CUTLASSPATH "$ENV{HOME}/Applications/cutlass")
endif()

message (${CUTLASSPATH})

# Verify CUTLASS exists
if(NOT EXISTS "${CUTLASSPATH}/cutlass/numeric_types.h")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASSPATH}\nPlease set CUTLASSPATH or install CUTLASS")
endif()

### libtorch
set(LIBTORCH_HOME "$ENV{LIBTORCH_HOME}")
if(NOT LIBTORCH_HOME)
    # message(FATAL_ERROR "Please set LIBTORCH_HOME environment variable")
    set(LIBTORCH_HOME "$ENV{HOME}/Applications/libtorch")
endif()

# Verify LibTorch exists
if(NOT EXISTS "${LIBTORCH_HOME}/include/torch/csrc/api/include/torch/torch.h")
    message(FATAL_ERROR "LibTorch not found at ${LIBTORCH_HOME}")
endif()

# 查找CUDA库
find_package(CUDAToolkit REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread json)

# Collect all C++/C/CUDA source files
file(GLOB CPP_SOURCES "src/*.cpp" "src/*.cc" "src/*.cxx" "src/*.C")
file(GLOB C_SOURCES "src/*.c")
file(GLOB CUDA_SOURCES "src/*.cu")

# Combine all sources
set(ALL_SOURCES ${CPP_SOURCES} ${C_SOURCES} ${CUDA_SOURCES})

# 添加可执行文件，并更改名称为${PROJECT_NAME}
add_executable(${PROJECT_NAME} ${ALL_SOURCES})

include_directories(
    ${CUTLASSPATH}
    ${CUDA_PATH}/include
)

link_directories(
    ${CUDA_PATH}/lib64
)

# 添加头文件包含路径
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUTLASSPATH}/include
    ${Boost_INCLUDE_DIRS} # Include Boost headers
    ${CUDAToolkit_INCLUDE_DIRS}

    ${LIBTORCH_HOME}/include
    ${LIBTORCH_HOME}/include/torch/csrc/api/include
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${LIBTORCH_HOME}/lib
)

# 链接cuBLAS和其他必要库
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart      # CUDA运行时
    CUDA::cublas      # cuBLAS库（新增）
    CUDA::cublasLt    # cuBLAS扩展（可选）
)

# 链接Boost库
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Boost::boost  # 头文件库
        Boost::system # shared_ptr实际依赖的组件
        Boost::thread # 如果需要线程支持
        Boost::json   # 如果使用了Boost JSON : sudo apt-get install libboost-json-dev
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    torch
    torch_cpu
    c10
    c10_cuda
    $<$<BOOL:${CUDA_FOUND}>:c10_cuda>
    $<$<BOOL:${CUDA_FOUND}>:torch_cuda>
)

# 针对MX150的优化选项
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    -Xcompiler=-Wall
    --default-stream per-thread
    >
)

# 添加编译定义
target_compile_definitions(${PROJECT_NAME} PRIVATE BOOST_ALL_NO_LIB=1)