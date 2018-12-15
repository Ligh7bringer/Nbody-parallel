#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>

#include "Body.h"
#include "Constants.h"

const unsigned int THREAD_NUM = 1024;

// error checking function
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort);

// functions which run the kernels
void interact_bodies_cuda(std::vector<Body>& bodies, unsigned int num_bodies);

// actual kernels
__global__ void interact_kernel(Body* bodies, unsigned int num_bodies);

__global__ void orig(Body* bodies, unsigned int num_bodies);

#endif  // KERNELS_H
