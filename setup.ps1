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

# Upgrade pip, setuptools, and wheel
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

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

# Install Jupyter kernel for this environment
Write-Host "Installing Jupyter kernel for this environment..." -ForegroundColor Yellow
python -m ipykernel install --user --name=comment-sentiment-env --display-name="Python (Comment Sentiment)"
Write-Host "Jupyter kernel installed" -ForegroundColor Green

# Create necessary directories (expanded)
$directories = @("data", "models", "logs", "outputs", "reports", "notebooks", "visualizations", "cache", "lexicons")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Cyan
    }
}

# Add .pth file for PYTHONPATH
$sitePackagesDir = Join-Path -Path ".\venv\Lib\site-packages" -ChildPath "comment_sentiment.pth"
$projectPath = (Get-Item -Path ".").FullName
$projectPath | Out-File -FilePath $sitePackagesDir -Encoding ascii
Write-Host "Created .pth file for automatic Python path configuration" -ForegroundColor Green

# Download NLTK data (required for social media preprocessing)
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
$nltkDownloads = @("punkt", "stopwords", "vader_lexicon", "wordnet")
foreach ($corpus in $nltkDownloads) {
    python -c "import nltk; nltk.download('$corpus', quiet=True)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Downloaded NLTK corpus: $corpus" -ForegroundColor Green
    } else {
        Write-Host "Warning: Failed to download NLTK corpus: $corpus" -ForegroundColor Yellow
    }
}

# Download pre-trained transformer model (example: twitter-roberta)
Write-Host "Downloading pre-trained transformer model..." -ForegroundColor Yellow
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest'); AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Downloaded Twitter-RoBERTa model" -ForegroundColor Green
} else {
    Write-Host "Warning: Failed to download Twitter-RoBERTa model" -ForegroundColor Yellow
}

Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "Virtual environment is now active." -ForegroundColor Green
Write-Host "To activate in future sessions, run: venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "For Jupyter, run: jupyter notebook" -ForegroundColor Cyan
Write-Host "If you encounter import errors, ensure the venv is activated and PYTHONPATH is set." -ForegroundColor Yellow
