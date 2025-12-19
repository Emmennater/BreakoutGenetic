cmake_minimum_required(VERSION 3.16)
project(env LANGUAGES CXX)

# Add pybind11
add_subdirectory(pybind11)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Create pybind11 module
pybind11_add_module(env env.cpp)

# Link OpenMP (MSVC will use /openmp)
target_link_libraries(env PRIVATE OpenMP::OpenMP_CXX)

# Optional: enforce C++ standard
set_target_properties(env PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
