#include <iostream>
#include <cutlass/numeric_types.h>
#include <stdio.h>
#include <cuda_runtime.h>
// #include <torch/torch.h>
#include <hellocuda.h>

__global__ void helloCudakernel(cutlass::half_t x) {
    printf("Device: %f\n", float(x*2.0_hf));
}

__global__ void helloWorldKernel() {
    for (int i = 0; i < 5; i++) {
        printf("Hello, World! from block %d, thread %d\n", blockIdx.x, threadIdx.x);
    }
}

int helloCutlass() {
    cutlass::half_t x = 0.5_hf;
    std::cout << "Host: " << 2.0_hf * x << std::endl;

    helloCudakernel<<< dim3(1,1), dim3(1,1,1) >>>(x);

    return 0;
}

int helloCuda() {
    // Check if CUDA is available
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (deviceCount == 0) {
        printf("No CUDA-capable devices found\n");
        return 1;
    }

    // printf("Found %d CUDA device(s)\n", deviceCount);

    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using device: %s\n", deviceProp.name);

    // Launch the kernel with 1 block and 5 threads
    helloWorldKernel<<<1, 5>>>();
    
    // Synchronize to ensure kernel execution completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset the device
    cudaDeviceReset();
    printf("Program completed successfully\n");

    return 0;
}

int main() {
    bool isCudaAvailable = checkCudaIsAvailable();
    if (!isCudaAvailable) {
        return 1;
    }
    
    helloCuda();

    helloCutlass();

    return 0;
}
