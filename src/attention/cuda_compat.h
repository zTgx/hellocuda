#pragma once

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
#endif

#if defined(USE_ROCM) && defined(__GFX9__)
  #define WARP_SIZE 64
#else
  #define WARP_SIZE 32
#endif

#ifndef USE_ROCM
  #define HELLOCUDA_LDG(arg) __ldg(arg)
#else
  #define HELLOCUDA_LDG(arg) *(arg)
#endif

#ifndef USE_ROCM
  #define HELLOCUDA_SHFL_XOR_SYNC(var, lane_mask) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask)
  #define HELLOCUDA_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#else
  #define HELLOCUDA_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor(var, lane_mask)
  #define HELLOCUDA_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) \
    __shfl_xor(var, lane_mask, width)
#endif

#ifndef USE_ROCM
  #define HELLOCUDA_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)
#else
  #define HELLOCUDA_SHFL_SYNC(var, src_lane) __shfl(var, src_lane)
#endif

#ifndef USE_ROCM
  #define HELLOCUDA_SHFL_DOWN_SYNC(var, lane_delta) \
    __shfl_down_sync(uint32_t(-1), var, lane_delta)
#else
  #define HELLOCUDA_SHFL_DOWN_SYNC(var, lane_delta) __shfl_down(var, lane_delta)
#endif

#ifndef USE_ROCM
  #define HELLOCUDA_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    cudaFuncSetAttribute(FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, VAL)
#else
  #define HELLOCUDA_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(FUNC, VAL) \
    hipFuncSetAttribute(FUNC, hipFuncAttributeMaxDynamicSharedMemorySize, VAL)
#endif
