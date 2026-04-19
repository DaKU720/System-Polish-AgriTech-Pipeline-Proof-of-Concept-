#!/bin/bash
echo "Installing dependencies for Polish AgriTech Pipeline..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install Python 3 and try again."
    exit 1
fi

# Create a virtual environment
echo "Creating a virtual environment (venv)..."
python3 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing pip requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete! To run the application:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Copy .env.example to .env and fill in your OPENAI_API_KEY"
echo "3. Run the script: python main.py"
