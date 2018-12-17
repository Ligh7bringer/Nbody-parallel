#include "Kernels.cuh"
#include "Util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

/*
 * Cuda kernels and interface functions.
 */

// this macro calls the error checking function so that the file and line number
// can be shown. Wrap around functions calls to use.
#define CUDA_WARN(func) \
  { gpuAssert((func), __FILE__, __LINE__); }

// check for CUDA errors
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(code) << " in file " << file
              << " on line " << line << std::flush;
    if (abort) getchar();
  }
}

// this function is can be called from the host
// it does all the necessary cuda mem allocations and copies data from the host
// to the device and vice versa
void interact_bodies_cuda(std::vector<Body>& bodies, unsigned int num_bodies) {
  // data size
  auto data_size = sizeof(Body) * num_bodies;
  // buffer for the bodies vector
  Body* buf_bodies;

  // allocate memory
  CUDA_WARN(cudaMalloc((void**)&buf_bodies, data_size));

  // copy data to device
  CUDA_WARN(
      cudaMemcpy(buf_bodies, &bodies[0], data_size, cudaMemcpyHostToDevice));

  auto num_blocks = num_bodies / THREAD_NUM;
  // run the kernel
  calc_forces_and_update<<<num_blocks, THREAD_NUM>>>(buf_bodies, num_bodies);
  // synchronize
  CUDA_WARN(cudaDeviceSynchronize());

  // copy results back to host
  // no need to use another vector, the original one can be overwritten
  CUDA_WARN(
      cudaMemcpy(&bodies[0], buf_bodies, data_size, cudaMemcpyDeviceToHost));

  // free the buffer
  CUDA_WARN(cudaFree(buf_bodies));
}

/* kernels */
// this kernel runs the 2 functions which will calculate the forces and update
// the bodies
__global__ void calc_forces_and_update(Body* bodies, unsigned int num_bodies) {
  calculate_forces(bodies, num_bodies);
  // it doesn't seem like the threads have to be explicitly synchronized
  //__syncthreads();
  update_bodies(bodies, num_bodies);
}

// calculates forces resulting from the interactions of all bodies
__device__ void calculate_forces(Body* bodies, unsigned int num_bodies) {
  // get global position
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  for (int j = i + 1; j < num_bodies; ++j) {
    // I am not sure why this if statement is needed
    // i and j should never be equal?
    if (i != j) {
      // vector to store the position difference between the 2 // bodies
      vec3 posDiff{};
      posDiff.x = (bodies[j].position().x - bodies[i].position().x) *
                  TO_METERS;  // calculate it
      posDiff.y = (bodies[j].position().y - bodies[i].position().y) * TO_METERS;
      posDiff.z = (bodies[j].position().z - bodies[i].position().z) * TO_METERS;
      // the actual distance is the length of the vector
      auto dist = sqrt(posDiff.x * posDiff.x + posDiff.y * posDiff.y +
                       posDiff.z * posDiff.z);
      double F =
          TIME_STEP * (G * bodies[i].mass() * bodies[j].mass()) /
          ((dist * dist + SOFTENING * SOFTENING) * dist);  // calculate force

      // set this body's acceleration
      bodies[j].acceleration().x -= F * posDiff.x / bodies[j].mass();
      bodies[j].acceleration().y -= F * posDiff.y / bodies[j].mass();
      bodies[j].acceleration().z -= F * posDiff.z / bodies[j].mass();

      // set the other body's acceleration
      bodies[i].acceleration().x += F * posDiff.x / bodies[i].mass();
      bodies[i].acceleration().y += F * posDiff.y / bodies[i].mass();
      bodies[i].acceleration().z += F * posDiff.z / bodies[i].mass();
    }
  }
}

// integrates positions
__device__ void update_bodies(Body* bodies, unsigned int num_bodies) {
  // get global position
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  // update this body:
  // update velocity
  bodies[i].velocity().x += bodies[i].acceleration().x;
  bodies[i].velocity().y += bodies[i].acceleration().y;
  bodies[i].velocity().z += bodies[i].acceleration().z;

  // reset acceleration
  bodies[i].acceleration().x = 0.0;
  bodies[i].acceleration().y = 0.0;
  bodies[i].acceleration().z = 0.0;

  // update position
  bodies[i].position().x += TIME_STEP * bodies[i].velocity().x / TO_METERS;
  bodies[i].position().y += TIME_STEP * bodies[i].velocity().y / TO_METERS;
  bodies[i].position().z += TIME_STEP * bodies[i].velocity().z / TO_METERS;
}

// this is based on GPU Gems
// it does increase performance slightly, however I couldn't get it to work
// fully. The body positions are not updated quite correctly
__device__ void test(Body* bodies, unsigned int num_bodies) {
  // get global position
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  for (int tile = 0; tile < gridDim.x; tile++) {
    // store positions in global memory for faster access
    __shared__ float3 spos[THREAD_NUM];
    auto tpos = bodies[tile * blockDim.x + threadIdx.x].position();
    spos[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z);
    // make sure all threads have reached this point before continuing
    __syncthreads();

    // loop unrolling supposedly increases performance. I couldn't find any
    // evidence this is true as apparently compilers to this by default
    //#pragma unroll 32
    for (int j = i + 1; j < THREAD_NUM; ++j) {
      if (i != j) {
        // vector to store the position difference between the 2 bodies
        vec3 posDiff{};
        posDiff.x = (spos[j].x - bodies[i].position().x) * TO_METERS;
        posDiff.y = (spos[j].y - bodies[i].position().y) * TO_METERS;
        posDiff.z = (spos[j].z - bodies[i].position().z) * TO_METERS;
        // the actual distance is the length of the vector
        auto dist = sqrtf(posDiff.x * posDiff.x + posDiff.y * posDiff.y +
                          posDiff.z * posDiff.z);
        // calculate force
        double F = TIME_STEP * (G * bodies[i].mass() * bodies[j].mass()) /
                   ((dist * dist + SOFTENING * SOFTENING) * dist);

        // set this body's acceleration
        bodies[j].acceleration().x -= F * posDiff.x / bodies[j].mass();
        bodies[j].acceleration().y -= F * posDiff.y / bodies[j].mass();
        bodies[j].acceleration().z -= F * posDiff.z / bodies[j].mass();

        // set the other body's acceleration
        bodies[i].acceleration().x += F * posDiff.x / bodies[i].mass();
        bodies[i].acceleration().y += F * posDiff.y / bodies[i].mass();
        bodies[i].acceleration().z += F * posDiff.z / bodies[i].mass();
      }
      // make sure all threads have reached this point
      __syncthreads();
    }
  }
}
