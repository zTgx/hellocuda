cmake_minimum_required(VERSION 3.18)
project(HelloCudaExamples LANGUAGES CXX CUDA)

# 设置CUDA架构和编译选项
set(CMAKE_CUDA_ARCHITECTURES "61")  # MX150 (Maxwell)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -diag-suppress 20208")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

cmake_policy(SET CMP0167 NEW)

# 设置CUTLASS路径
set(CUTLASSPATH "$ENV{CUTLASSPATH}")
if(NOT CUTLASSPATH)
    set(CUTLASSPATH "$ENV{HOME}/Applications/cutlass")
endif()

# 设置LibTorch路径
set(LIBTORCH_HOME "$ENV{LIBTORCH_HOME}")
if(NOT LIBTORCH_HOME)
    set(LIBTORCH_HOME "$ENV{HOME}/Applications/libtorch")
endif()

# 查找依赖
find_package(CUDAToolkit REQUIRED)
# find_package(Boost REQUIRED COMPONENTS system thread json)

# 定义公共链接库
set(COMMON_LIBS
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    torch
    torch_cpu
    c10
    c10_cuda
    $<$<BOOL:${CUDA_FOUND}>:torch_cuda>
    $<$<BOOL:${CUDA_FOUND}>:c10_cuda>
)

# 编译examples目录下的指定示例
set(EXAMPLE_FILES
    # "simple.cpp"
    # "cublas.cpp"
    "cutlass.cu"
)

foreach(example_file ${EXAMPLE_FILES})
    get_filename_component(example_name ${example_file} NAME_WE)
    set(example_path "${CMAKE_CURRENT_SOURCE_DIR}/${example_file}")

    if(EXISTS ${example_path})
        message(STATUS "Adding example: ${example_name}")
        
        add_executable(${example_name} ${example_path})
        
        target_compile_options(${example_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=-Wall
            --default-stream per-thread
            >
        )

        target_include_directories(${example_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CUTLASSPATH}
            ${LIBTORCH_HOME}/include
            ${LIBTORCH_HOME}/include/torch/csrc/api/include
            ${Boost_INCLUDE_DIRS}
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        
        target_link_libraries(${example_name} PRIVATE ${COMMON_LIBS})
        
        # 输出到bin/examples目录
        set_target_properties(${example_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/examples"
        )
    else()
        message(WARNING "Example file not found: ${example_path}")
    endif()
endforeach()

# 创建方便的构建目标
add_custom_target(build_examples DEPENDS ${EXAMPLE_FILES})