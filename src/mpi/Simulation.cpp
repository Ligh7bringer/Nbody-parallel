#include "Simulation.h"
#include "Timer.h"
#include "Util.h"

#include <iostream>
#include <random>

// constructor
Simulation::Simulation(unsigned int width, unsigned int height)
    : _width(width), _height(height), _zoom(1.2f) {
  // initialise mpi
  auto result = MPI_Init(nullptr, nullptr);
  if (result != MPI_SUCCESS) {
    std::cout << "Error initialising MPI!" << std::endl;
    std::abort();
  }

  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &comm_sz);
  MPI_Comm_rank(comm, &my_rank);

  // number of bodies should be evenly divisible by number of processes
  loc_n = NUM_BODIES / comm_sz;

  // initialise arrays
  masses = new double[NUM_BODIES * sizeof(double)];
  tmp_data = new vec2[2 * loc_n * sizeof(vec2)];
  loc_forces = new vec2[loc_n * sizeof(vec2)];
  loc_pos = new vec2[loc_n * sizeof(vec2)];
  loc_vel = new vec2[loc_n * sizeof(vec2)];
  if (my_rank == 0) {
    pos = new vec2[NUM_BODIES * sizeof(vec2)];
    vel = new vec2[NUM_BODIES * sizeof(vec2)];
  }

  // initialise mpi datatype
  MPI_Type_contiguous(DIM, MPI_DOUBLE, &vect_mpi_t);
  MPI_Type_commit(&vect_mpi_t);
  create_mpi_type(loc_n);

  // create the bodies
  init_bodies(masses, loc_pos, loc_vel, NUM_BODIES, loc_n);

  // initialise SFML
  if (my_rank == 0) {
    // create the window
    _window.create(sf::VideoMode(_width, _height), "N-body simulation",
                   sf::Style::Default);
    // setup the view
    _view.reset(sf::FloatRect(0, 0, _width, _height));
    _view.zoom(_zoom);
    _view.setViewport(sf::FloatRect(0.f, 0.f, 1.f, 1.f));
    // use it
    _window.setView(_view);
  }
}

// destructor
Simulation::~Simulation() {
  // delete all arrays
  delete[] masses;
  delete[] tmp_data;
  delete[] loc_forces;
  delete[] loc_pos;
  delete[] loc_vel;
  if (my_rank == 0) {
    delete[] pos;
    delete[] vel;
  }

  // free mpi datatype
  MPI_Type_free(&vect_mpi_t);
  MPI_Type_free(&cyclic_mpi_t);
  // shutdown mpi
  MPI_Finalize();

  // close the window if it's still open
  if (_window.isOpen()) _window.close();
  // since the program is not running while the window is open (instead it's
  // based on number of iterations), it should exit when the simulation ends as
  // this is what the other versions do so it has to be manually done
  std::abort();
}

// starts the simulation
void Simulation::start() {
  // changing this to while(_window.isOpen()) results in a weird error...
  while (true) {
    Timer t("update");
    calculate_forces(masses, tmp_data, loc_forces, loc_pos, NUM_BODIES, loc_n);

    for (int loc_bodies = 0; loc_bodies < loc_n; loc_bodies++)
      update_bodies(loc_bodies, masses, loc_forces, loc_pos, loc_vel,
                    NUM_BODIES, loc_n, TIME_STEP);

    t.stop();

    if (my_rank == 0) {
      poll_events();
    }

    render(TIME_STEP, masses, loc_pos, loc_vel, NUM_BODIES, loc_n);
  }
}

// renders the bodies
void Simulation::render(double time, double masses[], vec2 loc_pos[],
                        vec2 loc_vel[], int n, int loc_n) {
  // gather all positions and velocities
  MPI_Gather(loc_pos, loc_n, vect_mpi_t, pos, 1, cyclic_mpi_t, 0, comm);
  MPI_Gather(loc_vel, loc_n, vect_mpi_t, vel, 1, cyclic_mpi_t, 0, comm);

  // if this is the master process
  if (my_rank == 0) {
    // clear screen
    _window.clear(sf::Color::Black);

    // temporary circle shape
    sf::CircleShape star(DOT_SIZE, 50);
    star.setOrigin(sf::Vector2f(DOT_SIZE / 2.0f, DOT_SIZE / 2.0f));

    for (size_t i = 0; i < NUM_BODIES; ++i) {
      // get position and velocity
      sf::Vector2f pos(pos[i][X], pos[i][Y]);
      // std::cout << "[" << i << "]:" << pos.x << ", " << pos.y << std::endl;
      sf::Vector2f velocity(vel[i][X], vel[i][Y]);
      // calculate magnitude of the velocity vector
      auto mag = sqrtf(velocity.x * velocity.x + velocity.y * velocity.y);
      // orthogonal projection
      auto x = static_cast<int>(Util::to_pixel_space(pos.x, WIDTH, _zoom));
      auto y = static_cast<int>(Util::to_pixel_space(pos.y, HEIGHT, _zoom));
      // calculate a suitable colour
      star.setFillColor(Util::get_dot_colour(x, y, mag));
      // set the position of the circle shape
      star.setPosition(sf::Vector2f(x, y));
      star.setScale(sf::Vector2f(PARTICLE_MAX_SIZE, PARTICLE_MAX_SIZE));
      // the sun is stored at index 0
      if (i == 0) {
        // make the sun bigger and red
        star.setScale(sf::Vector2f(2.5f, 2.5f));
        star.setFillColor(sf::Color::Red);
      }
      // render
      _window.draw(star);
    }

    // display
    _window.display();
  }
}

/* -- mpi functions --- */
// creates a cyclic mpi type
void Simulation::create_mpi_type(int loc_n) {
  MPI_Datatype temp_mpi_t;
  MPI_Aint lb, extent;

  MPI_Type_vector(loc_n, 1, comm_sz, vect_mpi_t, &temp_mpi_t);
  MPI_Type_get_extent(vect_mpi_t, &lb, &extent);
  MPI_Type_create_resized(temp_mpi_t, lb, extent, &cyclic_mpi_t);
  MPI_Type_commit(&cyclic_mpi_t);
}

// create all the bodies in the simulation
void Simulation::init_bodies(double masses[], vec2 loc_pos[], vec2 loc_vel[],
                             int n, int loc_n) {
  // random distributions which will be used to generate a number of values
  using std::uniform_real_distribution;
  // random angle
  uniform_real_distribution<double> randAngle(0.0, 200.0 * PI);
  // random radius for the system
  uniform_real_distribution<double> randRadius(INNER_BOUND, SYSTEM_SIZE);
  // random z
  uniform_real_distribution<double> randHeight(0.0, SYSTEM_THICKNESS);
  // random mass
  uniform_real_distribution<double> randMass(10.0, 100.0);
  std::random_device rd;
  std::mt19937 gen(rd());

  // predeclare some variables
  double angle;
  double radius;
  double velocity;
  // velocity = 0.67*sqrt((G*SOLAR_MASS) / (4 * BINARY_SEPARATION*TO_METERS));

  // extra mass
  double totalExtraMass = 0.0;

  if (my_rank == 0) {
    // sun - put a really heavy body at the centre of the universe
    pos[0][X] = 0.0;
    pos[0][Y] = 0.0;
    vel[0][X] = 0.0;
    vel[0][Y] = 0.0;
    masses[0] = SOLAR_MASS;

    // starts at 1 because the sun is already at index 0
    for (int index = 1; index < n; ++index) {
      // generate a random body:
      // get a random angle
      angle = randAngle(gen);
      // get a random radius within the system bounds
      radius = sqrt(SYSTEM_SIZE) * sqrt(randRadius(gen));
      velocity =
          pow(((G * (SOLAR_MASS + ((radius - INNER_BOUND) / SYSTEM_SIZE) *
                                      EXTRA_MASS * SOLAR_MASS)) /
               (radius * TO_METERS)),
              0.5);
      // evenly distributed mass
      auto mass = (EXTRA_MASS * SOLAR_MASS) / NUM_BODIES;
      // keep track of mass
      totalExtraMass += mass;

      // store position, velocity and mass
      pos[index][X] = radius * cos(angle);
      pos[index][Y] = radius * sin(angle);
      vel[index][X] = velocity * sin(angle);
      vel[index][Y] = -velocity * cos(angle);
      masses[index] = mass;
    }
  }

  // broadcast the masses
  MPI_Bcast(masses, n, MPI_DOUBLE, 0, comm);
  // and scatter velocities and positions
  MPI_Scatter(pos, 1, cyclic_mpi_t, loc_pos, loc_n, vect_mpi_t, 0, comm);
  MPI_Scatter(vel, 1, cyclic_mpi_t, loc_vel, loc_n, vect_mpi_t, 0, comm);
}

// checks for sfml events
void Simulation::poll_events() {
  sf::Event event{};

  // poll events
  while (_window.pollEvent(event)) {
    if (event.type == sf::Event::Closed) {  // check if window is closed
      _window.close();
    }
    if (event.type ==
        sf::Event::MouseWheelScrolled)  // check if mouse scroll is used
    {
      // zoom in/out accordingly
      _zoom *= 1.f + (-event.mouseWheelScroll.delta / 10.f);
      _view.zoom(1.f + (-event.mouseWheelScroll.delta / 10.f));
    }
  }

  // move the view if the mouse is in one of the four corners of the window
  if (sf::Mouse::getPosition().x > (_width - 20)) _view.move(2 * _zoom, 0);
  if (sf::Mouse::getPosition().x < (0 + 20)) _view.move(-2 * _zoom, 0);
  if (sf::Mouse::getPosition().y > (_height - 20)) _view.move(0, 2 * _zoom);
  if (sf::Mouse::getPosition().y < (0 + 20)) _view.move(0, -2 * _zoom);

  // don't forget to set the view after modifying it
  _window.setView(_view);
}

void Simulation::calculate_forces(double masses[], vec2 tmp_data[],
                                  vec2 loc_forces[], vec2 loc_pos[], int n,
                                  int loc_n) {
  MPI_Status status;

  // source and destination processes
  auto src = (my_rank + 1) % comm_sz;
  auto dest = (my_rank - 1 + comm_sz) % comm_sz;

  // these functions are unsafe but copying data and filling arrays with 0
  // should be a simple enough operation
  memcpy(tmp_data, loc_pos, loc_n * sizeof(vec2));
  memset(tmp_data + loc_n, 0, loc_n * sizeof(vec2));
  memset(loc_forces, 0, loc_n * sizeof(vec2));

  // std::fill(&tmp_data[0][0], &tmp_data[0][0] + sizeof(vec2), 0);
  // std::fill(&loc_forces[0][0], &loc_forces[0][0] + sizeof(vec2), 0);

  // calculate results of local interactions
  calculate_process_forces(masses, tmp_data, loc_forces, loc_pos, loc_n,
                           my_rank, loc_n, my_rank, n, comm_sz);
  // calculate results of interactions with other processes
  for (int i = 1; i < comm_sz; i++) {
    int other_proc = (my_rank + i) % comm_sz;
    MPI_Sendrecv_replace(tmp_data, 2 * loc_n, vect_mpi_t, dest, 0, src, 0, comm,
                         &status);
    calculate_process_forces(masses, tmp_data, loc_forces, loc_pos, loc_n,
                             my_rank, loc_n, other_proc, n, comm_sz);
  }
  MPI_Sendrecv_replace(tmp_data, 2 * loc_n, vect_mpi_t, dest, 0, src, 0, comm,
                       &status);
  for (int loc_part = 0; loc_part < loc_n; loc_part++) {
    loc_forces[loc_part][X] += tmp_data[loc_n + loc_part][X];
    loc_forces[loc_part][Y] += tmp_data[loc_n + loc_part][Y];
  }
}

// calculates the forces acting on bodies
void Simulation::calculate_process_forces(double masses[], vec2 tmp_data[],
                                          vec2 loc_forces[], vec2 pos1[],
                                          int loc_n1, int rk1, int loc_n2,
                                          int rk2, int n, int p) {
  for (int gbl_body1 = rk1, loc_body1 = 0; loc_body1 < loc_n1;
       loc_body1++, gbl_body1 += p) {
    for (int gbl_body2 = first_index(gbl_body1, rk1, rk2, p),
             loc_body2 = global_to_local(gbl_body2, rk2, p);
         loc_body2 < loc_n2; loc_body2++, gbl_body2 += p) {
      calculate_forces_pair(masses[gbl_body1], masses[gbl_body2],
                            pos1[loc_body1], tmp_data[loc_body2],
                            loc_forces[loc_body1],
                            tmp_data[loc_n2 + loc_body2]);
    }
  }
}

// calculates the resulting forces of the interaction between 2 bodies
void Simulation::calculate_forces_pair(double m1, double m2, vec2 pos1,
                                       vec2 pos2, vec2 force1, vec2 force2) {
  vec2 posDiff{};
  posDiff[X] = (pos1[X] - pos2[X]) * TO_METERS;
  posDiff[Y] = (pos1[Y] - pos2[Y]) * TO_METERS;
  auto len = sqrtf(posDiff[X] * posDiff[X] + posDiff[Y] * posDiff[Y]);
  auto len_3 = (len * len + SOFTENING * SOFTENING) * len;
  auto mg = TIME_STEP * G * m1 * m2;
  double F = mg / len_3;

  // store forces
  force1[X] -= F * posDiff[X] / m1;
  force1[Y] -= F * posDiff[Y] / m1;
  force2[X] += F * posDiff[X] / m2;
  force2[Y] += F * posDiff[Y] / m2;
}

// updates the positions of the bodies based on the forces acting on them
void Simulation::update_bodies(int loc_body, double masses[], vec2 loc_forces[],
                               vec2 loc_pos[], vec2 loc_vel[], int n, int loc_n,
                               double delta_t) {
  int body = my_rank * loc_n + loc_body;
  double fact = delta_t / masses[body];
  loc_vel[loc_body][X] += loc_forces[loc_body][X];
  loc_vel[loc_body][Y] += loc_forces[loc_body][Y];
  loc_pos[loc_body][X] += TIME_STEP * loc_vel[loc_body][X] / TO_METERS;
  loc_pos[loc_body][Y] += TIME_STEP * loc_vel[loc_body][Y] / TO_METERS;
}

int Simulation::local_to_global(int loc_part, int proc_rk, int proc_count) {
  return loc_part * proc_count + proc_rk;
}

int Simulation::global_to_local(int gbl_part, int proc_rk, int proc_count) {
  return (gbl_part - proc_rk) / proc_count;
}

int Simulation::first_index(int gbl1, int rk1, int rk2, int proc_count) {
  if (rk1 < rk2)
    return gbl1 + (rk2 - rk1);
  else
    return gbl1 + (rk2 - rk1) + proc_count;
}
/* --------------------- */
