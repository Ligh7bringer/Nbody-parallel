## Parallelisation of a brute force n-body simulation O(n^2)

An investigation of possible ways of improving the performance of an O(n^2) n-body simulation. Techniques implemented:

- **GPGPU parallelisation** with CUDA
- **Distributed parallelisation** with MPI
- **Multithreaded parallelisation** with OpenMP

## Dependencies
- SFML (included as a submodule)
- OpenMP (supported by most c++ compilers)
- MPI ([Windows](https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi), on **Linux** use OpenMPI/mpich)
- Cuda (v.10)

## Build
1. Clone this repo.
2. In a terminal run ``git submodule update --init --recursive``
3. Use **CMake** to generate a solution or makefiles, e.g. ``cmake -G "Visual Studio 15/Unix Makefiles/etc."``
4. Compile from within your IDE of choice or via a terminal using ``make``

