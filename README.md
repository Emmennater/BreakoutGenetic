# Breakout Genetic Algorithm
## Project Setup
1. Install cmake
2. Clone pybind11
```shell
git clone https://github.com/pybind/pybind11.git
```
3. Build the environment cpp file
```
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release ..
cd ..
.\build.bat
```
4. Rename the .pyd file to env.pyd and move it to the root directory.

The reason this file is big is because it is statically linked.
This means that all of the dependencies are baked inside of the file.