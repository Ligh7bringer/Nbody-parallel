#include "Constants.h"
#include "Simulation.h"

int main() {
  // create a simulation
  Simulation sim(WIDTH, HEIGHT);
  // start it
  sim.start();

  return EXIT_SUCCESS;
}
