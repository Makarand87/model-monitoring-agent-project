#!/bin/bash
# Lines that start like this are shell comments
# Read project's current directory with $PWD

echo "Running command from: $PWD"

# Navigate to the directory (optional if already there)
cd "$PWD"

# Stage all changes
git add .

# Prompt the user for input
echo "Enter commit message: "
read -r commitMessage

# Commit with the saved message
git commit -m "$commitMessage"

# Push to the remote repository
git push
