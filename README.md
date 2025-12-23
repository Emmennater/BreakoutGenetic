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
- In `run.bat` and `build.bat` set `eigen` to the place you put the library.
- Same for the rest of the libraries...
- Download [pybind11](https://github.com/pybind/pybind11/releases)
- For the python headers and libraries I recommend install python with debug binaries etc. 
- Download the [Python Installer](https://www.python.org/downloads/windows)
- Click custom install and select the following:
  - Download debugging symbols
  - Download debug binaries
- Once installed, link the include and libs folder in `build.bat` like before.
- I have mine set up how I described so it should not be very different.
## Build the Project
- For training, run the vscode task named `Build`.
## Run the Project
- For training, run the vscode task named `Run`.
- For testing, run the python file `test.py`.
  - Make sure you have the same python installation selected from the set up process.
