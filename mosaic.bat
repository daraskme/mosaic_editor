@echo off
setlocal

:: Change to the script's directory
cd /d "%~dp0"

:: Create venv if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt

:: Run the application with arguments
echo Starting Mosaic Editor...
python mosaic.py %*

endlocal
