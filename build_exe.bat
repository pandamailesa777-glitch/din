@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --onedir --name VerticalDynamics --collect-data matplotlib --collect-data reportlab --collect-submodules scipy._external.array_api_compat.numpy --hidden-import PIL._tkinter_finder --exclude-module pytest desktop_entry.py
if errorlevel 1 (
  pause
  exit /b 1
)
if not exist "dist\VerticalDynamics\examples" mkdir "dist\VerticalDynamics\examples"
copy /Y "examples\*.json" "dist\VerticalDynamics\examples\" >nul
copy /Y "examples\*.csv" "dist\VerticalDynamics\examples\" >nul
copy /Y "START_HERE.txt" "dist\VerticalDynamics\START_HERE.txt" >nul
echo Built: dist\VerticalDynamics\VerticalDynamics.exe
pause
