# require a recent version of cmake
cmake_minimum_required(VERSION 3.10)

# project name
project(n_body)

# require c++ 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /EHsc" CACHE INTERNAL "" FORCE)
elseif(UNIX)
  message("hello linux!")
endif()

# Use OpenMP
find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# Use CUDA
find_package(CUDA 10.0 REQUIRED)

# Use MPI
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

#### Setup Directories ####
# Main output directory
SET(OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/")
# Ouput all DLLs from all libs into main build folder
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIRECTORY})

#### Add External Dependencies ####
# SFML 
add_subdirectory("lib/sfml")
set(SFML_INCS "lib/sfml/include")
link_directories("${CMAKE_BINARY_DIR}/lib/sfml/lib")

# some common utility headers are stored here
include_directories(common)

### Executables: ###
# sequential
file(GLOB_RECURSE SOURCES src/sequential/*.cpp src/sequential/*.h)
add_executable(n_body_seq ${SOURCES})
target_include_directories(n_body_seq SYSTEM PRIVATE ${SFML_INCS})
set(EXECUTABLES ${EXECUTABLES} n_body_seq)

# OpenMP version
file(GLOB_RECURSE SOURCES src/openmp/*.cpp src/openmp/*.h)
add_executable(n_body_omp ${SOURCES})
target_include_directories(n_body_omp SYSTEM PRIVATE ${SFML_INCS})
set(EXECUTABLES ${EXECUTABLES} n_body_omp)

# CUDA version
file(GLOB_RECURSE SOURCES src/cuda/*.cpp src/cuda/*.h)
file(GLOB KERNELS src/cuda/Kernels.cu src/cuda/Kernels.cuh)
cuda_add_executable(n_body_cuda ${SOURCES} ${KERNELS})
target_include_directories(n_body_cuda SYSTEM PRIVATE ${SFML_INCS})
target_compile_features(n_body_cuda PUBLIC cxx_std_11) # use c++11
set(EXECUTABLES ${EXECUTABLES} n_body_cuda)

# MPI version
file(GLOB_RECURSE SOURCES src/mpi/*.cpp src/mpi/*.h)
add_executable(n_body_mpi ${SOURCES})
target_include_directories(n_body_mpi SYSTEM PRIVATE ${SFML_INCS})
target_link_libraries(n_body_mpi ${MPI_LIBRARIES})
set(EXECUTABLES ${EXECUTABLES} n_body_mpi)

#### Link Dependencies ####
foreach (exe ${EXECUTABLES})
  # Set working directory to build dir
  set_target_properties(${exe} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(Configuration)
  )

  # link each executable with sfml
  target_link_libraries(${exe}
    optimized sfml-graphics debug sfml-graphics-d
    optimized sfml-audio debug sfml-audio-d
    optimized sfml-window debug sfml-window-d
    optimized sfml-system debug sfml-system-d
    optimized sfml-graphics debug sfml-graphics-d
  )
  
  add_dependencies(${exe} sfml-graphics sfml-audio)
endforeach ()