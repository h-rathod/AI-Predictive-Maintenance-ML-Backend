#!/bin/bash

# Create resources directories if they don't exist
mkdir -p src/main/resources/model
mkdir -p src/main/resources/data

# Copy model files
echo "Copying model files..."
cp -v model/* src/main/resources/model/ 2>/dev/null

# Copy data files
echo "Copying data files..."
cp -v data/* src/main/resources/data/ 2>/dev/null

echo "Resources copied successfully!"