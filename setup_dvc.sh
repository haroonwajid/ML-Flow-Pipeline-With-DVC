#!/bin/bash
# DVC Setup Script for Boston Housing Dataset
# This script will help you set up DVC and get the required screenshots

echo "=========================================="
echo "DVC Setup for ML-Flow-Pipeline-With-DVC"
echo "=========================================="
echo ""

# Step 1: Install DVC (if not installed)
echo "Step 1: Checking DVC installation..."
if ! command -v dvc &> /dev/null; then
    echo "DVC not found. Installing DVC..."
    pip install dvc
else
    echo "✓ DVC is already installed"
    dvc --version
fi
echo ""

# Step 2: Initialize DVC
echo "Step 2: Initializing DVC repository..."
if [ ! -d .dvc ]; then
    dvc init
    echo "✓ DVC initialized"
else
    echo "✓ DVC already initialized"
fi
echo ""

# Step 3: Set up remote storage (using local directory as example)
echo "Step 3: Setting up DVC remote storage..."
echo "Choose a remote storage option:"
echo "1. Local directory (for testing)"
echo "2. Google Drive (requires setup)"
echo "3. AWS S3 (requires credentials)"
echo "4. Skip remote setup (you can add it later)"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        REMOTE_PATH="../dvc_remote"
        mkdir -p "$REMOTE_PATH"
        dvc remote add -d myremote "$REMOTE_PATH"
        echo "✓ Local remote storage set up at: $REMOTE_PATH"
        ;;
    2)
        echo "For Google Drive, you'll need to:"
        echo "1. Create a folder in Google Drive"
        echo "2. Get the folder ID from the URL"
        echo "3. Run: dvc remote add -d myremote gdrive://<folder-id>"
        ;;
    3)
        echo "For AWS S3, you'll need:"
        echo "1. AWS credentials configured"
        echo "2. An S3 bucket"
        echo "3. Run: dvc remote add -d myremote s3://<bucket-name>/<path>"
        ;;
    4)
        echo "Skipping remote setup. You can add it later with:"
        echo "  dvc remote add -d myremote <remote-url>"
        ;;
esac
echo ""

# Step 4: Add data file to DVC
echo "Step 4: Adding data file to DVC..."
if [ -f "data/raw/boston_housing.csv" ]; then
    echo "Adding data/raw/boston_housing.csv to DVC..."
    dvc add data/raw/boston_housing.csv
    echo "✓ Data file added to DVC"
    
    # Commit DVC files to git
    git add data/raw/boston_housing.csv.dvc data/raw/.gitignore
    echo "✓ DVC files staged for git commit"
else
    echo "✗ Error: data/raw/boston_housing.csv not found!"
    exit 1
fi
echo ""

# Step 5: Push to remote (if remote is configured)
echo "Step 5: Pushing data to remote storage..."
if dvc remote list | grep -q "myremote"; then
    echo "Pushing to remote..."
    dvc push
    echo "✓ Data pushed to remote"
else
    echo "⚠ No remote configured. Skipping push."
    echo "  Configure a remote first, then run: dvc push"
fi
echo ""

# Step 6: Check status
echo "Step 6: Checking DVC status..."
echo "Running: dvc status"
echo "----------------------------------------"
dvc status
echo "----------------------------------------"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To get your screenshots:"
echo "1. Run: dvc status"
echo "2. Run: dvc push"
echo "3. Take screenshots of the terminal output"
echo ""
echo "Current DVC configuration:"
dvc remote list
echo ""
echo "To view DVC files:"
ls -la data/raw/*.dvc

