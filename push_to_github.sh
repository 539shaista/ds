#!/bin/bash
set -e

REPO_URL="https://github.com/539shaista/ds.git"
USER_EMAIL="shaistafalak1997@gmail.com"
USER_NAME="539shaista"

echo "=================================================="
echo "🚀 Preparing to push project to GitHub"
echo "Repository: $REPO_URL"
echo "Email:      $USER_EMAIL"
echo "=================================================="

# Check if git is available
if ! command -v git &> /dev/null || git --version 2>&1 | grep -q "xcode-select"; then
    echo "⚠️  Git is not yet ready on your system."
    echo "Please click 'Install' on the macOS prompt that popped up to finish installing developer tools,"
    echo "or run: xcode-select --install"
    exit 1
fi

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
else
    echo "📦 Existing git repository found."
fi

# Configure author details
git config user.name "$USER_NAME"
git config user.email "$USER_EMAIL"

# Stage all files
echo "📝 Staging files..."
git add -A

# Commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit."
else
    echo "💾 Creating commit..."
    git commit -m "Initial commit: Probability distribution simulator and dataset predictor"
fi

# Set main branch
git branch -M main

# Set remote origin
if git remote | grep -q "^origin$"; then
    git remote set-url origin "$REPO_URL"
else
    git remote add origin "$REPO_URL"
fi

echo ""
echo "=================================================="
echo "🔑 Push to GitHub:"
echo "GitHub requires a Personal Access Token (PAT) for authentication."
echo "If you already have a token, enter it below."
echo "If you need to create one:"
echo "  1. Go to: https://github.com/settings/tokens"
echo "  2. Click 'Generate new token (classic)'"
echo "  3. Select scope: 'repo'"
echo "  4. Copy your token (starts with ghp_...)"
echo "=================================================="
echo ""

read -s -p "Enter your GitHub Personal Access Token: " GITHUB_TOKEN
echo ""

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Pushing using standard git credentials prompt..."
    git push -u origin main
else
    echo "Pushing using provided token..."
    git push -u "https://${USER_NAME}:${GITHUB_TOKEN}@github.com/539shaista/ds.git" main
fi

echo ""
echo "✅ Successfully pushed to https://github.com/539shaista/ds.git !"
