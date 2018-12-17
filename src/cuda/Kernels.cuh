#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>

#include "Body.h"
#include "Constants.h"

const unsigned int THREAD_NUM = 1024 / 4;

// error checking function
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort);

// functions which run the kernels
void interact_bodies_cuda(std::vector<Body>& bodies, unsigned int num_bodies);

// actual kernels
__global__ void calc_forces_and_update(Body* bodies, unsigned int num_bodies);
__device__ void calculate_forces(Body* bodies, unsigned int num_bodies);
__device__ void update_bodies(Body* bodies, unsigned int num_bodies);

__device__ void test(Body* bodies, unsigned int num_bodies);
#endif  // KERNELS_H
