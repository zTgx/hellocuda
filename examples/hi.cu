#include <stdio.h>

__global__ void helloFromGPU(void) {
    printf("Hello from GPU\n");
}

int main(void) {

    printf("Hello from CPU\n");

    helloFromGPU<<< 1, 10 >>>();

    cudaError_t e = cudaDeviceReset();
    if (e != cudaSuccess) {
        return -1;
    }

    return 0;
}