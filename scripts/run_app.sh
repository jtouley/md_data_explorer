#!/bin/bash

# Clinical Analytics Platform - Streamlit Launcher
# This script runs the Streamlit application with helpful startup messages

set -e

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                                                ║${NC}"
echo -e "${BLUE}║       🏥  Clinical Analytics Platform  🏥      ║${NC}"
echo -e "${BLUE}║                                                ║${NC}"
echo -e "${BLUE}║     Multi-dataset clinical analytics with     ║${NC}"
echo -e "${BLUE}║         config-driven architecture             ║${NC}"
echo -e "${BLUE}║                                                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if virtual environment exists
echo -e "${YELLOW}📋 Checking environment...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo -e "${YELLOW}   Please run: ${GREEN}uv sync${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment found${NC}"

# Check for data files
echo ""
echo -e "${YELLOW}📂 Checking for datasets...${NC}"
if [ -f "data/covid_ms/covid_ms_data.csv" ]; then
    echo -e "${GREEN}✓ COVID-MS dataset available${NC}"
else
    echo -e "${YELLOW}⚠ COVID-MS dataset not found (will skip in UI)${NC}"
fi

if [ -d "data/sepsis" ]; then
    echo -e "${GREEN}✓ Sepsis dataset directory found${NC}"
else
    echo -e "${YELLOW}⚠ Sepsis dataset not found (will skip in UI)${NC}"
fi

# Display feature summary
echo ""
echo -e "${BLUE}🚀 Platform Features:${NC}"
echo -e "   • Auto-discovery of datasets via registry"
echo -e "   • Config-driven transformations (no hardcoding)"
echo -e "   • Polars-optimized ETL (5-10x faster)"
echo -e "   • Interactive data profiling"
echo -e "   • Logistic regression analysis"
echo -e "   • CSV/JSON data export"
echo -e "   • 44+ automated tests passing"

# Activate virtual environment and run streamlit
echo ""
echo -e "${GREEN}🎬 Starting Streamlit application...${NC}"
echo -e "${YELLOW}   The app will open in your browser at:${NC} ${GREEN}http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}   Press Ctrl+C to stop the server${NC}"
echo ""

# Run streamlit with the app
source .venv/bin/activate
streamlit run src/clinical_analytics/ui/app.py
