@echo off

SET eigen=C:\Libraries\eigen-5.0.0

g++ -fopenmp -I%eigen% -Iinclude ^
  breakout.cpp env.cpp evolution.cpp net.cpp -std=c++17 -O2 -Wall -Wextra -o breakout.exe
breakout.exe
del breakout.exe