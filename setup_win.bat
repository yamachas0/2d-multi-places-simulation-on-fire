@echo off
REM Setup script for LLM Multi-Agent 2D Simulation (Windows)

echo Setting up virtual environment...

REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, run:
echo   deactivate

pause

