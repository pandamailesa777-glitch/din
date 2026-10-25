@echo off
cd /d "%~dp0"
if exist "dist\VerticalDynamics\VerticalDynamics.exe" (
  start "" "dist\VerticalDynamics\VerticalDynamics.exe"
  exit /b 0
)
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Follow README.md to create .venv and install dependencies.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m vertical_dynamics --gui
if errorlevel 1 pause
