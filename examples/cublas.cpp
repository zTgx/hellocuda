#include <cublas_v2.h>
#include <iostream>
#include <vector>

int main() {
    // 1. 初始化向量数据 (主机端)
    const int n = 4;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> y = {5.0f, 6.0f, 7.0f, 8.0f};
    float result = 0.0f;

    // 2. 分配设备内存
    float *d_x, *d_y;
    cudaMalloc((void**)&d_x, n * sizeof(float));
    cudaMalloc((void**)&d_y, n * sizeof(float));

    // 3. 拷贝数据到设备
    cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    // 4. 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 5. 调用cuBLAS计算点积 (x·y)
    cublasSdot(handle, n, d_x, 1, d_y, 1, &result);

    // 6. 输出结果
    std::cout << "Dot product: " << result << std::endl;  // 应输出 70 (1*5 + 2*6 + 3*7 + 4*8)

    // 7. 释放资源
    cublasDestroy(handle);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}