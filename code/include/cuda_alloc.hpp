#ifndef CUDA_ALLOC_HPP
#define CUDA_ALLOC_HPP

#ifdef USE_CUDA
#include "cuda_device.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <new>

inline void *cudaManagedAlloc(std::size_t size) {
    void *ptr = nullptr;
    if (cudaMallocManaged(&ptr, size) != cudaSuccess) {
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
