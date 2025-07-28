// #include <torch/torch.h>
#include <iostream>
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <torch/types.h>
#include <torch/cuda.h>

namespace hellocuda {

bool checkCudaIsAvailable() {
    std::cout << "PyTorch CUDA available: " << torch::cuda::is_available() << std::endl;
    std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
    
    if (torch::cuda::is_available()) {
        torch::Device device(torch::kCUDA);
        std::cout << "Current CUDA device: " << device << std::endl;
        
        torch::Tensor tensor = torch::rand({2, 3}).to(device);
        std::cout << "Tensor on CUDA:\n" << tensor.to(torch::kCPU) << std::endl;

        return true;
    } else {
        std::cerr << "CUDA is not available!" << std::endl;
        // 打印可能的错误原因
        std::cerr << "Possible reasons:" << std::endl;
        std::cerr << "1. libtorch was not compiled with CUDA support" << std::endl;
        std::cerr << "2. CUDA drivers are not properly installed" << std::endl;
        std::cerr << "3. Environment variables are not set correctly" << std::endl;
    }
    
    return false;
}

} // namespace hellocuda 
