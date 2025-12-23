# Breakout Evolutionary AI
Uses a genetic algorithm to train a policy to play breakout.
## Project Set Up
### Compiler
- Make sure you have the `C/C++ Extension Pack` installed.
- Set the path to your compiler in `.vscode\c_cpp_properties.json` (compilerPath).
- Make sure you have that configuration selected (C/C++: Select a Configuration)
### Libaries
- Download [eigen](https://github.com/PX4/eigen/tree/master)
- Unzip and put it somewhere (i.e. C:\Libraries\eigen-5.0.0). Make sure you don't have an extra folder (i.e. eigen-5.0.0\eigen-5.0.0\Eigen).
- In `run.bat` set `eigen` to the place you put the library.
## Run the Project
- Run the vscode task named `Run`.