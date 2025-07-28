#include <cuda_runtime.h>

#define CEIL_DIV(M, N) ((M) + (N)-1) / (N)

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  // compute position in C that this thread is responsible for
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

int main() {

    // dim3 gridDim(CEIl_DIV(M, 32), CEIl_DIV(N, 32), 1);
    // dim3 blockDim(32, 32, 1);

    // sgemm_naive<<< gridDim, blockDim >>>(M, N, K, alpha, A, B, beta, C);







    return 0;
}