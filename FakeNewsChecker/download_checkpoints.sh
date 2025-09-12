#!/usr/bin/env bash
set -e

# Create the Checkpoints folder (if not exists)
mkdir -p Checkpoints

echo "Installing gdown if needed..."
pip install --upgrade gdown

echo "Downloading entire Checkpoints folder from Google Drive..."
gdown "https://drive.google.com/drive/folders/1dnWja7aOCpFaesg7toMU35CqVJFo_Aba" -O Checkpoints --folder

echo "Download done!"
