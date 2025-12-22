@echo off
SET pythonVersion=312
cd build
cmake --build . --config Release
ren "Release\env.cp%pythonVersion%-win_amd64.pyd" "env.pyd"
move /Y "Release\env.pyd" "..\env.pyd"
