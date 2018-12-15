#ifndef UTIL_H
#define UTIL_H

#include <SFML/Graphics/Color.hpp>
#include "Constants.h"

#include <cmath>
#include <iostream>

/*
 * Some utility functions are stored here
 */

// 3 dimensional vector
struct vec3 {
  double x, y, z;  // positions

  double magnitude()  // returns length of the vector
  {
    return sqrtf(x * x + y * y + z * z);
  }

  // used for debugging
  friend std::ostream& operator<<(std::ostream& str, vec3 const& v) {
    return str << "(" << v.x << ", " << v.y << ", " << v.z << ")";
  }
};

// utility functions
class Util {
 private:
  static double clamp(double x) { return fmax(fmin(x, 1.0), 0.0); }

 public:
  // this might be the same as sfml's mapPixelToCoord
  static double to_pixel_space(double p, int size, float zoom) {
    return (size / 2.0) * (1.0 + p / (SYSTEM_SIZE * zoom));
  }

  // calculates colour for the bodies based on their speed
  static sf::Color get_dot_colour(double x, double y, double vMag) {
    const double velocityMax = MAX_VEL_COLOR;  // 35000
    const double velocityMin =
        sqrt(0.8 * (G * (SOLAR_MASS + EXTRA_MASS * SOLAR_MASS)) /
             (SYSTEM_SIZE * TO_METERS));  // MIN_VEL_COLOR;
    const double vPortion = sqrt((vMag - velocityMin) / velocityMax);

    sf::Color c;
    c.r = clamp(4 * (vPortion - 0.333)) * RGBA_MAX;
    c.g = clamp(fmin(4 * vPortion, 4.0 * (1.0 - vPortion))) * RGBA_MAX;
    c.b = clamp(4 * (0.5 - vPortion)) * RGBA_MAX;
    c.a = RGBA_MAX / 2.f;

    return c;
  }

  unsigned char colour_depth(unsigned char x, unsigned char p, double f) {
    return fmax(fmin((x * f + p), 255), 0);
  }
};

#endif  // !UTIL_H
