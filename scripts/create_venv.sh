#!/bin/bash

# Function to print and log messages
log() {
    echo "$1"
    logger "$1"
}

# Check if virtual environment directory exists and remove it if it does
if [ -d "deeplr_venv" ]; then
    log "Removing existing virtual environment..."
    rm -rf deeplr_venv
fi

# Move up one directory
cd ..

# Create a new virtual environment
log "Creating new virtual environment..."
python3 -m venv deeplr_venv

# Check if the virtual environment was created successfully
if [ $? -ne 0 ]; then
    log "Failed to create virtual environment."
    exit 1
fi

# Activate the virtual environment
log "Activating virtual environment..."
source deeplr_venv/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    log "Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip

# Check if pip upgrade was successful
if [ $? -ne 0 ]; then
    log "Failed to upgrade pip."
    exit 1
fi

# Install requirements
log "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Check if requirements installation was successful
if [ $? -ne 0 ]; then
    log "Failed to install requirements."
    exit 1
fi

log "Virtual environment setup completed successfully."
