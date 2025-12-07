#!/bin/bash

# Clinical Analytics Platform - Streamlit Launcher
# This script runs the Streamlit application

set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "🏥 Clinical Analytics Platform"
echo "================================"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run 'uv sync' first."
    exit 1
fi

# Activate virtual environment and run streamlit
echo "🚀 Starting Streamlit application..."
echo ""

# Run streamlit with the app
source .venv/bin/activate
streamlit run src/clinical_analytics/ui/app.py
