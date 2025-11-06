#!/bin/bash

# Snake Behavior Classification - Setup Script
# This script sets up the Python environment for the project

echo "================================"
echo "Snake Behavior Classification"
echo "Environment Setup"
echo " by Thomas M. Baer"
echo "================================"
echo ""

# Check if Python 3.7+ is installed
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
if [ $? -ne 0 ]; then
    echo "\[ERROR\]: Python 3 is not installed. Please install Python 3.7 or higher."
    echo "   Visit https://www.python.org/downloads/ to download Python."
    exit 1
fi

echo "\[SUCCESS\]: Found Python version: $python_version"

# Check if we're on macOS and if python-tk might be needed
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "📦 Checking for Tkinter support..."
    python3 -c "import tkinter" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "\[WARNING\]: Tkinter is not available. The labeler GUI requires Tkinter."
        echo ""
        echo "To install Tkinter, you have two options:"
        echo "1. If you have Homebrew, run: brew install python-tk"
        echo "2. Reinstall Python from python.org \(includes Tkinter by default\)"
        echo ""
        read -p "Do you want to continue without Tkinter? \(y/n\): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "\[SUCCESS\]: Tkinter is available"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "\[ERROR\]: Failed to create virtual environment"
        exit 1
    fi
    echo "\[SUCCESS\]: Virtual environment created"
else
    echo "\[SUCCESS\]: Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "\[ERROR\]: Failed to activate virtual environment"
    exit 1
fi

# Install requirements
echo ""
echo "📦 Installing required packages..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "\[ERROR\]: Failed to install packages"
    echo "   Please check your internet connection and try again"
    exit 1
fi

echo ""
echo "================================"
echo "\[SUCCESS\]: Setup Complete!"
echo "================================"
echo ""
echo "To start using the project:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Run the labeler: python python/labeler.py"
echo ""
echo "Remember to run 'source venv/bin/activate' each time you open a new terminal!"
