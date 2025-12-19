add_subdirectory(pybind11)
pybind11_add_module(module_name env.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
