#!/usr/bin/env python3
"""
Setup script for Glioma Detection System
Installs all dependencies and prepares the environment
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def run_command(command, description):
    """Run a command and show progress"""
    print(f"→ {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✓ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description} - Failed")
        print(f"  Error: {e.stderr}")
        return False

def main():
    print_header("GLIOMA DETECTION SYSTEM - SETUP")
    
    print("Python version:", sys.version)
    print("Platform:", platform.system())
    print()
    
    # Step 1: Create virtual environment
    print("[1/5] Creating virtual environment...")
    if not os.path.exists("venv"):
        if run_command(f"{sys.executable} -m venv venv", "Create venv"):
            print("  ✓ Virtual environment created")
    else:
        print("  ✓ Virtual environment already exists")
    
    # Determine pip path
    if platform.system() == "Windows":
        pip = "venv\\Scripts\\pip.exe"
        python = "venv\\Scripts\\python.exe"
    else:
        pip = "venv/bin/pip"
        python = "venv/bin/python"
    
    # Step 2: Upgrade pip
    print("\n[2/5] Upgrading pip...")
    run_command(f"{pip} install --upgrade pip", "Upgrade pip")
    
    # Step 3: Install dependencies
    print("\n[3/5] Installing dependencies...")
    run_command(f"{pip} install -r requirements.txt", "Install packages")
    
    # Step 4: Create directories
    print("\n[4/5] Creating directories...")
    directories = [
        "data/uploads",
        "data/temp",
        "data/dataset",
        "logs",
        "trained_models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ Created: {directory}")
    
    # Step 5: Create .env if not exists
    print("\n[5/5] Setting up environment...")
    if not os.path.exists(".env"):
        with open(".env.example" if os.path.exists(".env.example") else ".env", "w") as f:
            f.write("""DEBUG=True
HOST=0.0.0.0
PORT=8000
SECRET_KEY=glioma-secret-key-change-in-production
KAGGLE_USERNAME=
KAGGLE_KEY=
UPLOAD_DIR=./data/uploads
DATASET_DIR=./data/dataset
MODEL_DIR=./trained_models
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]
""")
        print("  ✓ Created .env file")
    else:
        print("  ✓ .env file already exists")
    
    # Success message
    print_header("SETUP COMPLETE!")
    
    print("Next steps:")
    print()
    print("1. Download dataset (optional):")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\python.exe download_dataset.py")
    else:
        print("   venv/bin/python download_dataset.py")
    print()
    print("2. Start the server:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\python.exe main.py")
    else:
        print("   venv/bin/python main.py")
    print()
    print("3. Open in browser:")
    print("   http://localhost:8000")
    print()
    print("4. API Documentation:")
    print("   http://localhost:8000/docs")
    print()

if __name__ == "__main__":
    main()