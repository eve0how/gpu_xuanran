#ifndef CUDA_DEVICE_HPP
#define CUDA_DEVICE_HPP

// 文件说明：CUDA 编译期 HOST/DEVICE 宏封装。
// 原创性声明：参考已有代码（常见 CUDA 宏模式），宏定义独立实现。

#ifdef __CUDACC__
// Callable from both host and device translation units.
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOST_DEVICE
#define DEVICE
#define HOST
#endif

#endif
