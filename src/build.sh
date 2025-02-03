#!/bin/bash
set -e  # Stop script if any command fails

echo "Updating and installing dependencies..."

# Install Java (OpenJDK 17)
apt-get update && apt-get install -y openjdk-17-jdk

# Ensure Python dependencies are installed
pip install --no-cache-dir -r requirements.txt

# Set permissions for R5.jar if needed
chmod +x ./src/R5.jar
