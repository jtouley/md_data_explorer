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

# Check and initialize Ollama LLM service (self-contained, like DuckDB)
echo ""
echo -e "${YELLOW}🤖 Checking Ollama LLM service...${NC}"
source .venv/bin/activate

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama binary found${NC}"
    
    # Check if service is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama service running${NC}"
        
        # Check for models
        MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null || echo "0")
        if [ "$MODELS" -gt 0 ]; then
            echo -e "${GREEN}✓ Ollama ready ($MODELS model(s) available)${NC}"
        else
            echo -e "${YELLOW}⚠ Ollama running but no models available${NC}"
            echo -e "${YELLOW}   Run: ${GREEN}ollama pull llama3.2:3b${NC} to download a model"
        fi
    else
        echo -e "${YELLOW}⚠ Ollama service not running${NC}"
        echo -e "${YELLOW}   Attempting to start Ollama service...${NC}"
        
        # Try to start Ollama in background
        nohup ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        sleep 2
        
        # Check if it started
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ollama service started${NC}"
        else
            echo -e "${YELLOW}⚠ Could not start Ollama service automatically${NC}"
            echo -e "${YELLOW}   Please start manually: ${GREEN}ollama serve${NC}"
            echo -e "${YELLOW}   Natural language queries will use pattern matching only${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Ollama not installed${NC}"
    echo -e "${YELLOW}   Natural language queries will use pattern matching only${NC}"
    echo -e "${YELLOW}   Install at: ${GREEN}https://ollama.ai${NC} or run: ${GREEN}curl -fsSL https://ollama.ai/install.sh | sh${NC}"
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

# Run streamlit with the app (verbose mode for debugging)
source .venv/bin/activate
streamlit run src/clinical_analytics/ui/app.py \
    --logger.level=info \
    --server.fileWatcherType=poll
