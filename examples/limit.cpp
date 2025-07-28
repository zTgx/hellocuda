#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    size_t stack_size_limit;
    cudaError_t err = cudaDeviceGetLimit(&stack_size_limit, cudaLimitStackSize);

    if (err == cudaSuccess) {
        printf("Current stack size limit: %zu bytes\n", stack_size_limit);
    } else {
        fprintf(stderr, "Error getting stack size limit: %s\n", cudaGetErrorString(err));
    }

    return 0;
}