#include "Body.h"

#include <cmath>
#include <random>

// generates n random bodies, stores them in a vector and returns it
std::vector<Body> Body::generate(unsigned int n) {
  // vector to store the results
  std::vector<Body> bodies;
  // make sure there is enough space in it
  bodies.reserve(n);

  // random distributions which will be used to generate a number of values
  using std::uniform_real_distribution;
  uniform_real_distribution<double> randAngle(0.0, 200.0 * PI);  // random angle
  uniform_real_distribution<double> randRadius(
      INNER_BOUND, SYSTEM_SIZE);  // random radius for the system
  uniform_real_distribution<double> randHeight(0.0,
                                               SYSTEM_THICKNESS);  // random z
  uniform_real_distribution<double> randMass(10.0, 100.0);  // random mass
  std::random_device rd;  // make it more random!
  std::mt19937 gen(rd());

  // predeclare some variables
  double angle;
  double radius;
  double velocity;
  // velocity = 0.67*sqrt((G*SOLAR_MASS) / (4 * BINARY_SEPARATION*TO_METERS));
  // sun - put a really heavy body at the centre of the universe
  bodies.emplace_back(vec3{0.0, 0.0, 0.0}, vec3{0.0, 0.0, 0.0},
                      vec3{0.0, 0.0, 0.0}, SOLAR_MASS);

  // extra mass
  double totalExtraMass = 0.0;
  // start at 1 because the sun is at index 0
  for (int index = 1; index < NUM_BODIES; ++index) {
    // generate a random body:
    angle = randAngle(gen);  // get a random angle
    radius =
        sqrt(SYSTEM_SIZE) *
        sqrt(randRadius(gen));  // get a random radius within the system bounds
    auto t = ((G * (SOLAR_MASS + ((radius - INNER_BOUND) / SYSTEM_SIZE) *
                                     EXTRA_MASS * SOLAR_MASS)));
    velocity = t / (radius * TO_METERS);
    // calculate velocity
    velocity = pow(velocity, 0.5);
    auto mass =
        (EXTRA_MASS * SOLAR_MASS) / NUM_BODIES;  // evenly distributed mass
    totalExtraMass += mass;                      // keep track of mass
    // add the body to the vector
    bodies.emplace_back(
        vec3{radius * cos(angle), radius * sin(angle),
             randHeight(gen) - SYSTEM_THICKNESS / 2},              // position
        vec3{velocity * sin(angle), -velocity * cos(angle), 0.0},  // velocity
        vec3{0.0, 0.0, 0.0},  // acceleration
        mass);                // mass
  }

  // return resutl
  return bodies;
}
