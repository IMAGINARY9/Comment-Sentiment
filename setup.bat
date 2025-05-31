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

REM Upgrade pip, setuptools, and wheel
python -m pip install --upgrade pip setuptools wheel

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

REM Install Jupyter kernel for this environment
python -m ipykernel install --user --name=comment-sentiment-env --display-name="Python (Comment Sentiment)"

REM Create necessary directories (expanded)
for %%d in (data models logs outputs reports notebooks visualizations cache lexicons) do (
    if not exist %%d (
        mkdir %%d
        echo Created directory: %%d
    )
)

REM Add .pth file for PYTHONPATH
if exist venv\Lib\site-packages (
    for /f %%i in ('cd') do echo %%i > venv\Lib\site-packages\comment_sentiment.pth
    echo Created .pth file for automatic Python path configuration
)

echo.
echo Downloading NLTK data...
python -c "import nltk; [nltk.download(x, quiet=True) for x in ['punkt','stopwords','vader_lexicon','wordnet']]"
if errorlevel 1 (
    echo WARNING: Failed to download NLTK data
)

echo.
echo Downloading pre-trained transformer model (Twitter-RoBERTa)...
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest'); AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')"
if errorlevel 1 (
    echo WARNING: Failed to download pre-trained transformer model
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
