#ifndef BODY_H
#define BODY_H

#include "Constants.h"
#include "Util.h"

#include <iostream>
#include <vector>

/*
 * This class describes a single body in the simulation.
 */

class Body {
  vec3 _position;  // position
  vec3 _velocity;  // velocity
  vec3 _accel;     // acceleration
  double _mass;    // mass

 public:
  Body(vec3 pos, vec3 vel, vec3 accel, double mass)
      : _position(pos), _velocity(vel), _accel(accel), _mass(mass) {}

  Body& update(double dt);
  void interact(Body& other, double dt);

  static std::vector<Body> generate(unsigned int);

  const vec3& position() const;
  const vec3& velocity() const;
  vec3& acceleration();  // not const so that the acceleration can be modified;
                         // probably should add a setter instead
  double mass() const;

  // will be used for debugging
  friend std::ostream& operator<<(std::ostream& str, const Body& p) {
    return str << "Position: " << p.position() << "; Velocity: "
               << p.velocity()
               //<< "; Acceleration: " << p.acceleration()
               << "; Mass: " << p.mass() << std::endl;
  }
};

#endif /* BODY_H */
