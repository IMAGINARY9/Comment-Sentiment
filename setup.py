"""
Setup script for comment sentiment analysis project.

This script installs dependencies, downloads models, and prepares the environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description, use_venv=True):
    print(f"\n{description}...")
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    if use_venv and os.path.exists(venv_python):
        if command.startswith("python "):
            command = command.replace("python", f'"{venv_python}"', 1)
        elif command.startswith("pip "):
            command = command.replace("pip", f'"{venv_python}" -m pip', 1)
    try:
        result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        print(f"\u2713 {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\u2717 {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def ensure_venv():
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        result = subprocess.run("python -m venv venv", shell=True)
        if result.returncode != 0:
            print("\u2717 Failed to create virtual environment")
            sys.exit(1)
        print("\u2713 Virtual environment created.")
    else:
        print("\u2713 Virtual environment already exists.")

def install_dependencies():
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    run_command(f'"{venv_python}" -m pip install --upgrade pip setuptools wheel', "Upgrading pip and build tools", use_venv=False)
    return run_command(f'"{venv_python}" -m pip install -r requirements.txt', "Installing dependencies", use_venv=False)

def download_models():
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    commands = [
        f'"{venv_python}" -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained(\'cardiffnlp/twitter-roberta-base-sentiment-latest\'); AutoTokenizer.from_pretrained(\'cardiffnlp/twitter-roberta-base-sentiment-latest\')"',
        f'"{venv_python}" -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained(\'bert-base-uncased\'); AutoTokenizer.from_pretrained(\'bert-base-uncased\')"',
    ]
    success = True
    for cmd in commands:
        success &= run_command(cmd, "Downloading pre-trained models", use_venv=False)
    return success

def setup_directories():
    directories = [
        "data", "models", "logs", "lexicons", "embeddings",
        "reports", "notebooks", "visualizations", "cache"
    ]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"\u2713 Created directory: {directory}")
    return True

def download_nltk_data():
    venv_python = os.path.join("venv", "Scripts", "python.exe")
    commands = [
        f'"{venv_python}" -c "import nltk; nltk.download(\'punkt\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'stopwords\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'twitter_samples\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'vader_lexicon\', quiet=True)"',
        f'"{venv_python}" -c "import nltk; nltk.download(\'wordnet\', quiet=True)"',
    ]
    success = True
    for cmd in commands:
        success &= run_command(cmd, "Downloading NLTK data", use_venv=False)
    return success

def download_embeddings():
    print("\nNote: GloVe embeddings are large (>1GB). Download manually if needed:")
    print("- GloVe Twitter 27B: https://nlp.stanford.edu/projects/glove/")
    print("- Save to: ./embeddings/glove.twitter.27B.300d.txt")
    return True

def main():
    print("Setting up Comment Sentiment Analysis Project")
    print("=" * 50)
    if sys.version_info < (3, 8):
        print("\u2717 Python 3.8 or higher is required")
        sys.exit(1)
    print(f"\u2713 Using Python {sys.version}")
    ensure_venv()
    steps = [
        ("Setting up directories", setup_directories),
        ("Installing dependencies", install_dependencies),
        ("Downloading NLTK data", download_nltk_data),
        ("Downloading pre-trained models", download_models),
        ("Checking embeddings", download_embeddings),
    ]
    success_count = 0
    for description, func in steps:
        if func():
            success_count += 1
        else:
            print(f"\u26a0 Warning: {description} failed, but continuing...")
    print(f"\nSetup completed: {success_count}/{len(steps)} steps successful")
    if success_count >= len(steps) - 1:
        print("\n\U0001F389 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Prepare your data: python scripts/prepare_data.py")
        print("2. Train transformer model: python scripts/train.py --config configs/twitter_roberta.yaml")
        print("3. Train BiLSTM model: python scripts/train.py --config configs/bilstm_glove.yaml")
        print("4. Evaluate ensemble: python scripts/evaluate.py --ensemble")
    else:
        print("\n\u26a0 Setup completed with some warnings. Check the output above.")

if __name__ == "__main__":
    main()
