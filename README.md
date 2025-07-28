# HelloCUDA

A minimal CUDA project demonstrating integration with CUTLASS and LibTorch and more.

## Prerequisites

### Environment Variables
Set these before building:
```bash
# CUTLASS path (required for matrix operations)
export CUTLASSPATH=/path/to/cutlass

# LibTorch path (required for PyTorch C++ API)
export LIBTORCH_HOME=/path/to/libtorch

# Example (adjust paths to your system):
export CUTLASSPATH="$HOME/Applications/cutlass"
export LIBTORCH_HOME="$HOME/libs/libtorch"
```

### System Requirements
- CUDA Toolkit (>= 11.0)
- CMake (>= 3.18)
- GCC (>= 7.5) or compatible C++17 compiler
- NVIDIA GPU with Compute Capability >= 6.1 (Maxwell+)

## Build Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hellocuda.git
cd hellocuda
```

2. Build the project:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

3. Run the executable:
```bash
./hellocuda
```

## Project Structure
```
HelloCUDA/
├── CMakeLists.txt      # Build configuration
├── books               # HelloCuda book files
├── include/            # Header files
│   └── hellocuda.h     # Main header
├── src/                # Source files
│   └── main.cu         # Main CUDA implementation
│   └── ...             # Other CUDA implementation
└── README.md           # This file
```

## Key Features
- Demonstrates CUTLASS for high-performance matrix operations
- Shows LibTorch C++ API integration
- Minimal CMake configuration for CUDA projects

## Troubleshooting

### Common Issues
1. **"CUTLASS not found"**:
   - Verify `CUTLASSPATH` points to the CUTLASS root directory
   - Clone CUTLASS if missing:
     ```bash
     git clone https://github.com/NVIDIA/cutlass.git $HOME/Applications/cutlass
     ```

2. **LibTorch errors**:
   - Download the correct LibTorch version from [pytorch.org](https://pytorch.org/)
   - Ensure CUDA versions match between LibTorch and your system

3. **CUDA architecture errors**:
   - Update `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt for your GPU
   - For example: `set(CMAKE_CUDA_ARCHITECTURES "75")` for Turing GPUs