#include "Kernels.cuh"
#include "Util.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

// this macro calls the error checking function so that the file and line number
// can be shown
#define CUDA_WARN(func) \
  { gpuAssert((func), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "Error: " << cudaGetErrorString(code) << " in file " << file
              << " on line " << line << std::flush;
    if (abort) getchar();
  }
}

void interact_bodies_cuda(std::vector<Body>& bodies, unsigned int num_bodies) {
  auto data_size = sizeof(Body) * num_bodies;
  Body* buf_bodies;

  CUDA_WARN(cudaMalloc((void**)&buf_bodies, data_size));

  CUDA_WARN(
      cudaMemcpy(buf_bodies, &bodies[0], data_size, cudaMemcpyHostToDevice));

  orig<<<num_bodies / THREAD_NUM, THREAD_NUM>>>(buf_bodies, num_bodies);

  CUDA_WARN(cudaDeviceSynchronize());

  CUDA_WARN(
      cudaMemcpy(&bodies[0], buf_bodies, data_size, cudaMemcpyDeviceToHost));

  CUDA_WARN(cudaFree(buf_bodies));
}

/* kernels */
__global__ void interact_kernel(Body* bodies, unsigned int num_bodies) {
  // get global position
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  for (int tile = 0; tile < gridDim.x; tile++) {
    __shared__ float3 spos[THREAD_NUM];
    auto tpos = bodies[tile * blockDim.x + threadIdx.x].position();
    spos[threadIdx.x] = make_float3(tpos.x, tpos.y, tpos.z);
    __syncthreads();

    //#pragma unroll 32
    for (int j = 0; j < THREAD_NUM; ++j) {
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
      __syncthreads();
    }
  }

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

__global__ void orig(Body* bodies, unsigned int num_bodies) {
  // get global position
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  for (int j = 0; j < num_bodies; ++j) {
    if (i != j) {
      // vector to store the position difference between the 2 // bodies
      vec3 posDiff{};
      posDiff.x = (bodies[j].position().x - bodies[i].position().x) *
                  TO_METERS;  // calculate it
      posDiff.y = (bodies[j].position().y - bodies[i].position().y) * TO_METERS;
      posDiff.z = (bodies[j].position().z - bodies[i].position().z) * TO_METERS;
      // the actual distance is the length of the vector
      auto dist = sqrtf(posDiff.x * posDiff.x + posDiff.y * posDiff.y +
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
