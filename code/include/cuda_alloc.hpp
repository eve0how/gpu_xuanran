#ifndef CUDA_ALLOC_HPP
#define CUDA_ALLOC_HPP

// 文件说明：CUDA Managed Memory 分配辅助与设备端 new/delete。
// 原创性声明：参考已有代码（CUDA 统一内存惯例），封装函数独立实现。

#ifdef USE_CUDA
#include "cuda_device.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <new>

inline void *cudaManagedAlloc(std::size_t byteCount) {
    void *ptr = nullptr;
    if (cudaMallocManaged(&ptr, byteCount) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

inline void cudaManagedFree(void *ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

inline void enableCudaManagedAccess(void *ptr) {
    (void)ptr;
}

#ifdef __CUDACC__
HOST_DEVICE inline void *operator new(std::size_t size) {
    void *ptr = nullptr;
    if (cudaMallocManaged(&ptr, size) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

HOST_DEVICE inline void *operator new[](std::size_t size) {
    return operator new(size);
}

HOST_DEVICE inline void operator delete(void *ptr) noexcept {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

HOST_DEVICE inline void operator delete[](void *ptr) noexcept {
    operator delete(ptr);
}
#endif

#endif
