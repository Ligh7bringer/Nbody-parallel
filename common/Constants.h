#ifndef CONSTANTS_H_
#define CONSTANTS_H_

/*
 * Some constant values are stored here so that they can be easily changed
 * across all versions of the program.
 */

constexpr unsigned int WIDTH = 1920;             // Window width
constexpr unsigned int HEIGHT = 1080;            // Window height
constexpr unsigned int NUM_BODIES = (1024 * 4);  // Number of bodies
constexpr double PI = 3.14159265358979323846;    // pi
constexpr double TO_METERS = 1.496e11;           // Meters in an AU
constexpr double SYSTEM_SIZE = 3.5;              // Farthest particles in AU
constexpr double SYSTEM_THICKNESS = 0.08;        // Thickness in AU
constexpr double INNER_BOUND = 0.3;  // Closest bodies to center in AU
constexpr double SOFTENING =
    (0.015 * TO_METERS);  // Softens body interactions at close distances
constexpr double SOLAR_MASS = 2.0e30;  // in kg
constexpr double BINARY_SEPARATION =
    0.07;  // AU (only applies when binary code uncommented)
constexpr double EXTRA_MASS =
    1.5;  // 0.02 Disk mask as a portion of center star/black hole massed
constexpr double MAX_DISTANCE =
    0.75;  // 2.0 Barnes-Hut Distance approximation factor
constexpr double G = 6.67408e-11;          // The gravitational constant
constexpr double MAX_VEL_COLOR = 40000.0;  // Both in km/s
constexpr double MIN_VEL_COLOR = 14000.0;
constexpr float DOT_SIZE = 8;  // Range of pixels to render
constexpr double TIME_STEP =
    (3 * 32 * 1024);  // (1*128*1024) Simulated time between integration steps,
                      // // in seconds
constexpr bool DEBUG_INFO = false;         // Print lots of info to the console
constexpr double RGBA_MAX = 255;           // Maximum color value in RGB
constexpr double PARTICLE_MAX_SIZE = 0.7;  // Maximum particle size

#endif /* CONSTANTS_H_ */
