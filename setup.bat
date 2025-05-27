@echo off
REM Comment Sentiment Analysis - Windows Batch Setup Script
REM This script sets up the virtual environment and installs all dependencies

echo.
echo ========================================
echo  Comment Sentiment Analysis Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python version:
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo Virtual environment already exists.
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip
)

echo.
echo Installing project dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Installing project in development mode...
pip install -e .
if errorlevel 1 (
    echo WARNING: Failed to install project in development mode
)

echo.
echo Creating required directories...
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\interim" mkdir data\interim
if not exist "models\checkpoints" mkdir models\checkpoints
if not exist "models\final" mkdir models\final
if not exist "logs\training" mkdir logs\training
if not exist "reports\figures" mkdir reports\figures
if not exist "lexicons\custom" mkdir lexicons\custom

echo.
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True); print('NLTK data downloaded successfully!')"
if errorlevel 1 (
    echo WARNING: Failed to download NLTK data
)

echo.
echo ========================================
echo  Setup completed successfully!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo   activate.bat
echo.
echo To start training, run:
echo   python scripts\train.py
echo.
pause
