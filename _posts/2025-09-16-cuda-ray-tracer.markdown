---
layout: post
title:  "CUDA Ray Tracer"
date:   2025-09-17 10:20:00 -0400
categories: CUDA
---
During the weekend, I implemented a simple ray tracer using CUDA. The code is available on [Github](https://github.com/ronchuxia/RayTracingCUDA.git).

# Background

When I was an undergraduate, I took a computer graphics course and learned about ray tracing. One project of the course was to implement a ray tracer following "Ray Tracing in One Weekend" by Peter Shirley. The implementation was in C++. The original repository is [here](https://github.com/RayTracing/raytracing.github.io). 

In my implementation, I added support for triangle meshes, so that I could render models defined in STL files. My repository is [here](https://github.com/ronchuxia/RayTracing.git). 

During my experiments, I found that due to the great number of triangles, rendering triangle meshes took much more time than rendering spheres, even after using a bounding volume hierarchy (BVH) to accelerate ray-triangle intersection tests. Therefore, I decided to try using CUDA to speed up the rendering process. And I finally did it this weekend.

I've read CUDA code before, but this is the first time I wrote CUDA code. It was an interesting learning experience.

# CUDA Introduction

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by Nvidia. It is similar to C++, but there are some differences, as you will see below.

# CUDA Memory Management

Before writing CUDA code, it's better to understand how CUDA memory management works. 

There are mainly two types of memory in CUDA: **host memory** (CPU) and **device memory** (GPU). In the past, you need to allocate memory on host or device, and explicitly copy memory between host and device using `cudaMemcpy()`.

However, in CUDA8, nvidia introduced **unified memory**. With unified memory, you can allocate memory that is accessible by both the CPU and GPU using `cudaMallocManaged()`. The CUDA runtime automatically manages the memory transfers between host and device.

To construct an object in unified memory, first allocate memory using `cudaMallocManaged()`:

```c
Sphere* d_spheres;
cudaMallocManaged((void**)&d_spheres, sizeof(Sphere));
```

Then use placement new to construct the object in the allocated memory:

```c
new(d_spheres) Sphere(center, radius, material);
```

Objects in unified memory will be automatically moved to GPU memory when accessed by GPU code. However, this does not mean that every move will be successful. For example, if an object contains a pointer, and this pointer points to CPU memory (which is not accessible to GPU), then, when CUDA runtime tries to move the object to GPU memory, an invalid memory error will be raised.

Therefore, if an object contains pointers, you need to **make sure that the pointers point to GPU accessible memory (device memory or unified memory)**. 

# CUDA Multithreading

## CUDA functions

There are three types of **CUDA function qualifiers**:
- `__global__`: a **kernel** function, executed on device, called on host
	- Can be launched with `<<< >>>`
	- Must return `void`
- `__device__`: a device function, executed on device, called on device
	- Cannot be launched with `<<< >>>`
- `__host__`: a host function, executed on host, called on host
	- Cannot be launched with `<<< >>>`

`__global__` functions are called **kernels**, they are executed by many threads in parallel on the GPU.

## Grid, block and warp

CUDA GPUs have many parallel processors grouped into **Streaming Multiprocessors**, or SMs. 

Here are some important concepts:
- **Grid**: all threads launched by a kernel
	- A grid is made up of multiple blocks
- **Block**: a group of threads that must fit on one SM
	- The number of threads per block cannot exceed the number of threads an SM can host
	- A block is made up of multiple warps
	- The number of threads per block is a multiple of 32
- **Warp**: 32 threads that the SM actually schedules together

Here is the relationship between sm, grid, and block:
- Each SM can run **multiple** concurrent thread blocks.
- Multiple threads from the same block must run on a **single** SM.
- Multiple blocks from the same grid can run on **different** SMs.

# CUDA Synchronization

CUDA kernel launches don't block the calling CPU thread. Call `cudaDeviceSynchronize()` to wait until CUDA kernel finish executing.

# CUDA Random Numbers

CUDA doesn't support C++ `rand()`. You need to use **cuRAND** library to generate random numbers on GPU.

1. Include cuRAND header.
```c
#include <curand_kernel.h>
```

2. Initialize random states. Each GPU thread needs its own random state. You typically set them up in a kernel.
```c
__global__ void initRand(curandState *states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}
```

3. Generate random numbers in kernels.
```c
__global__ void generateRand(curandState *states, float *results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // uniform [0,1)
    float r = curand_uniform(&states[idx]);

    results[idx] = r;
}
```

# CUDA Class Design

When designing classes for CUDA, keep the following in mind:
- Member functions cannot be kernels (i.e., cannot be `__global__`).
- **Avoid virtual functions**, as they are not well supported in CUDA.
- If a class contains pointers, ensure the pointers point to GPU-accessible memory (device memory or unified memory).

For a class with virtual functions, the compiler generates a **vtable** to support dynamic dispatch. The vtable is stored in CPU memory, which is not accessible to GPU. Therefore, avoid using virtual functions in CUDA classes.

# CUDA Data Structures
CUDA doesn't support C++ STL. You can use libraries like Thrust, or implement your own data structures. However, note that Thrust containers can only be used in host code, not device code.

# CUDA Compilation

To compile CUDA code, use `nvcc`, the NVIDIA CUDA Compiler. For example:
```shell
nvcc -o ray_tracer ray_tracer.cu
```

# CUDA Debugging
When running CUDA code, you may encounter CUDA errors. You can refer to the error code list [here](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038).

To debug, compile in debug mode:
```shell
nvcc -o ray_tracer ray_tracer.cu -G -g -O0
```
- `-g`: CPU debug
- `-G`: GPU debug
- `-O0`: no optimization

You can use gdb to debug host code, and cuda-gdb to debug device code. You can also use `compute-sanitizer` to check for memory errors.

A lot of bugs arise from invalid memory access. So learning about memory space is helpful for debugging:
- `05x`: heap memory
- `07x`: stack memory / GPU unified memory

# TODO

This is only a very basic CUDA ray tracer. There are many things to improve:
- Add triangle mesh support
- Add BVH acceleration
- Simplify my class design

Adding triangle mesh support is straightforward, as I already have the code in my CPU ray tracer. I just need to port the code to CUDA.

BVH acceleration is critical for rendering triangle meshes in a reasonable time. However, the original code in "Ray Tracing in One Weekend" implements BVH using recursion, which is not well supported in CUDA. Therefore, I need to redesign the BVH data structure and implement an iterative BVH traversal algorithm.

Constructing BVH on GPU is another challenge. I may need to implement a parallel BVH construction algorithm, like LBVH.

My code follows the original design in "Ray Tracing in One Weekend", which uses virtual functions. To remove the virtual functions, I introduced a wrapper for both materials and hittables, which makes constructing an object very cumbersome. I may need to redesign my classes to make them more CUDA-friendly.

# Resources
If you are interested in Computer Graphics, I recommend GAMES101 by Dr. Lingqi Yan.

# References
- [An Even Easier Introduction to CUDA]( https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)






