#include <iostream>
#include <cutlass/numeric_types.h>

__global__ void kernel(cutlass::half_t x) {
    printf("Device: %f\n", float(x * 2.0_hf));
}

int main() {
    cutlass::half_t x = 0.5_hf;

    std::cout << "Host: " << 2.0_hf * x << std::endl;

    kernel<<< dim3(1,1), dim3(1,1,1) >>>(x);

    cudaDeviceSynchronize();

    return 0;
}