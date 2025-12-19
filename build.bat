@echo off
cd build
cmake --build . --config Release
ren "Release\env.cp313-win_amd64.pyd" "env.pyd"
move /Y "Release\env.pyd" "..\env.pyd"
