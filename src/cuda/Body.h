#ifndef BODY_H
#define BODY_H

#include <cuda_runtime.h>
#include "Constants.h"
#include "Util.h"

#include <iostream>
#include <vector>

/*
 * This class defines a single body in the simulation.
 */

class Body {
 private:
  double _mass;    // mass
  vec3 _position;  // position
  vec3 _velocity;  // velocity
  vec3 _accel;     // acceleration

 public:
  Body() = default;
  Body(vec3 pos, vec3 vel, vec3 accel, double mass)
      : _position(pos), _velocity(vel), _accel(accel), _mass(mass) {}

  static std::vector<Body> generate(unsigned int);

  __device__ __host__ double mass() const { return _mass; }

  __device__ __host__ vec3& position() { return _position; }

  __device__ __host__ vec3& velocity() { return _velocity; }

  __device__ __host__ vec3& acceleration() { return _accel; }
};

#endif /* BODY_H */
