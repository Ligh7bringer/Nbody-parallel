#include <cmath>
#include <random>

#include "Body.h"
#include "Timer.h"

Body &Body::update(double dt) {
  // update velocity
  _velocity.x += _accel.x;
  _velocity.y += _accel.y;
  _velocity.z += _accel.z;

  // reset acceleration
  _accel.x = 0.0;
  _accel.y = 0.0;
  _accel.z = 0.0;

  // update position
  _position.x += TIME_STEP * _velocity.x / TO_METERS;
  _position.y += TIME_STEP * _velocity.y / TO_METERS;
  _position.z += TIME_STEP * _velocity.z / TO_METERS;

  return *this;
}

// calculates the effects of an interaction between 2 bodies
void Body::interact(Body &other) {
  vec3 posDiff{};  // position difference between the 2 bodies
  posDiff.x = (_position.x - other.position().x) * TO_METERS;  // calculate it
  posDiff.y = (_position.y - other.position().y) * TO_METERS;
  posDiff.z = (_position.z - other.position().z) * TO_METERS;
  auto dist = posDiff.magnitude();  // the distance between the bodies is the
                                    // length of the vector
  double F = TIME_STEP * (G * _mass * other.mass()) /
             ((dist * dist + SOFTENING * SOFTENING) * dist);  // calculate force

  // set this body's acceleration
  _accel.x -= F * posDiff.x / _mass;
  _accel.y -= F * posDiff.y / _mass;
  _accel.z -= F * posDiff.z / _mass;

  // set the other body's acceleration
  other.acceleration().x += F * posDiff.x / other.mass();
  other.acceleration().y += F * posDiff.y / other.mass();
  other.acceleration().z += F * posDiff.z / other.mass();
}

// generates n random bodies, stores them in a vector and returns it
std::vector<Body> Body::generate(unsigned int n) {
  // Timer t("generate bodies");
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

/* getters and setters */
const vec3 &Body::position() const { return _position; }

const vec3 &Body::velocity() const { return _velocity; }

vec3 &Body::acceleration() { return _accel; }

double Body::mass() const { return _mass; }
