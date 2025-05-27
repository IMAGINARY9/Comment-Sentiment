# Comment Sentiment Analysis - Setup Script
# This script sets up the virtual environment and installs dependencies

Write-Host "Setting up Comment Sentiment Analysis environment..." -ForegroundColor Green

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install dependencies
if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Some dependencies failed to install" -ForegroundColor Yellow
    }
} else {
    Write-Host "Warning: requirements.txt not found" -ForegroundColor Yellow
}

# Install project in development mode
if (Test-Path "setup.py") {
    Write-Host "Installing project in development mode..." -ForegroundColor Yellow
    pip install -e .
}

# Create necessary directories
$directories = @("data", "models", "logs", "reports", "lexicons")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Cyan
    }
}

# Download NLTK data (required for social media preprocessing)
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')" 2>$null

Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "Virtual environment is now active." -ForegroundColor Green
Write-Host "To activate in future sessions, run: venv\Scripts\Activate.ps1" -ForegroundColor Cyan
