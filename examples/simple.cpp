#include <iostream>
#include <ATen/Functions.h>
#include <torch/types.h>
#include <torch/cuda.h>

int main() {
    // 检查 CUDA 是否可用
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available!" << std::endl;
        return -1;
    }

    // 创建一个随机 Tensor 并移动到 CUDA 设备
    torch::Tensor tensor = torch::rand({2, 3}).to(torch::kCUDA);
    
    // 打印 Tensor（需要先移回 CPU）
    std::cout << "Tensor on CUDA:\n" << tensor.to(torch::kCPU) << std::endl;
    
    // 检查设备信息
    std::cout << "Tensor device: " << tensor.device() << std::endl;
    
    return 0;
}