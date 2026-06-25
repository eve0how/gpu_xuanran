#ifndef CUDA_DEVICE_HPP
#define CUDA_DEVICE_HPP

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOST_DEVICE
#define DEVICE
#define HOST
#endif

#endif
