#ifndef SIMULATION_H
#define SIMULATION_H

#include <SFML/Graphics.hpp>
#include "Body.h"

#include <vector>

class Simulation {
 private:
  unsigned int _width, _height;  // size of the window
  float _zoom;                   // zoom factor
  sf::RenderWindow _window;      // the window
  sf::View _view;                // the view
  std::vector<Body> _bodies;

  void update();

  // sfml functions
  void poll_events();
  void render();

 public:
  // default constructor won't be needed
  Simulation() = delete;
  // this one should be used instead
  Simulation(unsigned int width, unsigned int height);
  // destructor
  ~Simulation() = default;

  void start();
};

#endif  // SIMULATION_H
