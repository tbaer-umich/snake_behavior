#!/bin/bash

# Snake Behavior Classification - Setup Script
# This script sets up the Python environment for the project

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RESET='\033[0m' # No Color

echo "================================"
echo "Snake Behavior Classification"
echo "Environment Setup"
echo " by Thomas M. Baer"
echo "================================"
echo ""

# ------------------------------------------------------
# 1. Auto-Update Section
# ------------------------------------------------------
if [ -d ".git" ]; then
    read -p "Would you like to update to the latest version? [y/n] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "⬇️  Pulling latest changes..."
        git pull
        if [ $? -eq 0 ]; then
             echo -e "${GREEN}[SUCCESS]${RESET}: Update complete."
        else
             echo -e "${RED}[ERROR]${RESET}: Update failed. Continuing with setup..."
        fi
        echo ""
    fi
fi

# ------------------------------------------------------
# 2. Python Version Check
# ------------------------------------------------------
# Check if Python 3.7+ is installed
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${RESET}: Python 3 is not installed. Please install Python 3.7 or higher."
    echo "   Visit https://www.python.org/downloads/ to download Python."
    exit 1
fi

echo -e "${GREEN}[SUCCESS]${RESET}: Found Python version: $python_version"

# ------------------------------------------------------
# 3. macOS Tkinter Check
# ------------------------------------------------------
# Check if we're on macOS and if python-tk might be needed
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "📦 Checking for Tkinter support..."
    python3 -c "import tkinter" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}[WARNING]${RESET}: Tkinter is not available. The labeler GUI requires Tkinter."
        echo ""
        echo "To install Tkinter, you have two options:"
        echo "1. If you have Homebrew, run: brew install python-tk"
        echo "2. Reinstall Python from python.org (includes Tkinter by default)"
        echo ""
        read -p "Do you want to continue without Tkinter? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}[SUCCESS]${RESET}: Tkinter is available"
    fi
fi

# ------------------------------------------------------
# 4. Virtual Environment Setup
# ------------------------------------------------------
# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR]${RESET}: Failed to create virtual environment"
        exit 1
    fi
    echo -e "${GREEN}[SUCCESS]${RESET}: Virtual environment created"
else
    echo -e "${GREEN}[SUCCESS]${RESET}: Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${RESET}: Failed to activate virtual environment"
    exit 1
fi

# ------------------------------------------------------
# 5. Package Installation
# ------------------------------------------------------
# Install requirements
echo ""
echo "📦 Installing required packages..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR]${RESET}: Failed to install packages"
    echo "   Please check your internet connection and try again"
    exit 1
fi

echo ""
echo "================================"
echo -e "${GREEN}[SUCCESS]${RESET}: Setup Complete!"
echo "================================"
echo ""
echo "To start using the project:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the labeler: python python/labeler.py"
echo ""
echo "Remember to run 'source venv/bin/activate' each time you open a new terminal!"
