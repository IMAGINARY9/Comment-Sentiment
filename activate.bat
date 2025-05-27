@echo off
REM Comment Sentiment Analysis - Environment Activation Script
REM This script activates the virtual environment for development

echo.
echo ========================================
echo  Comment Sentiment Analysis
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Environment activated successfully!
echo.
echo Available commands:
echo   python scripts\train.py          - Start training
echo   python -m pytest tests\          - Run tests
echo   python scripts\evaluate.py       - Evaluate models
echo   python scripts\predict.py        - Make predictions
echo.
echo To deactivate, type: deactivate
echo.

REM Keep the command prompt open
cmd /k
