#ifndef SIMULATION_H
#define SIMULATION_H

#include <SFML/Graphics.hpp>

#include <mpi.h>

class Simulation {
 private:
  const unsigned int NUM_STEPS = 1000;
  // only x and y, no point in using z as in the
  // other versions it doesn't really do anything
  const unsigned int DIM = 2;

  const int X = 0;         // x subscript of vector
  const int Y = 1;         // y subscript of vector
  typedef double vec2[2];  // custom vector which is just an array of doubles

  // size of the window
  unsigned int _width;
  unsigned int _height;
  float _zoom;               // zoom factor
  sf::RenderWindow _window;  // the window
  sf::View _view;            // the view

  // arrays to store required data
  vec2* vel = nullptr;
  vec2* pos = nullptr;
  double* masses;    // array where masses are stored
  vec2* loc_pos;     // positions of bodies
  vec2* tmp_data;    // received positions and forces
  vec2* loc_vel;     // local velocities
  vec2* loc_forces;  // local forces
  unsigned int loc_n;
  int loc_part;

  // mpi variables
  int comm_sz;  // number of processes
  int my_rank;  // rank of process
  int _length;
  char _host_name[MPI_MAX_PROCESSOR_NAME];
  // mpi cyclic type
  MPI_Datatype vect_mpi_t;
  MPI_Datatype cyclic_mpi_t;
  MPI_Comm comm;

  /* ------- */
  void create_mpi_type(int loc_n);
  void init_bodies(double masses[], vec2 loc_pos[], vec2 loc_vel[], int n,
                   int loc_n);
  void calculate_forces(double masses[], vec2 tmp_data[], vec2 loc_forces[],
                        vec2 loc_pos[], int n, int loc_n);
  void calculate_process_forces(double masses[], vec2 tmp_data[],
                                vec2 loc_forces[], vec2 pos1[], int loc_n1,
                                int rk1, int loc_n2, int rk2, int n, int p);
  int local_to_global(int loc_part, int proc_rk, int proc_count);
  int global_to_local(int gbl_part, int proc_rk, int proc_count);
  int first_index(int gbl1, int rk1, int rk2, int proc_count);
  void calculate_forces_pair(double m1, double m2, vec2 pos1, vec2 pos2,
                             vec2 force1, vec2 force2);
  void update_bodies(int loc_part, double masses[], vec2 loc_forces[],
                     vec2 loc_pos[], vec2 loc_vel[], int n, int loc_n,
                     double delta_t);
  /* ------- */

  // sfml functions
  void poll_events();
  void render(double time, double masses[], vec2 loc_pos[], vec2 loc_vel[],
              int n, int loc_n);

 public:
  // default constructor won't be needed
  Simulation() = delete;
  // this one should be used instead
  Simulation(unsigned int width, unsigned int height);
  // destructor
  ~Simulation();

  void start();
};

#endif  // SIMULATION_H
