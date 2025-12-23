@echo off

SET eigen=C:\Libraries\eigen-5.0.0
SET python=C:\Users\emmet\AppData\Local\Programs\Python\Python312\include
SET pybind=C:\Libraries\pybind11-3.0.1\include
SET python_lib=C:\Users\emmet\AppData\Local\Programs\Python\Python312\libs

g++ -fopenmp -O3 -shared -std=c++17 -fPIC ^
  -I%eigen% -I%python% -I%pybind% -Iinclude ^
  *.cpp ^
  -o link.pyd ^
  -L%python_lib% ^
  -static ^
  -lpython312 ^
  -static-libgcc -static-libstdc++ ^
  -Wl,--exclude-all-symbols
