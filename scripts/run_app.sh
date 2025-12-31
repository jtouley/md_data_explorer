#!/bin/bash

# Clinical Analytics Platform - Streamlit Launcher
# This script automatically installs dependencies and runs the Streamlit application
# Designed for easy use by doctors - everything is automated!

# Don't exit on error - we want to handle installation failures gracefully
set +e

# Colors for better output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install Homebrew if not present
install_homebrew() {
    if command_exists brew; then
        return 0
    fi
    
    echo -e "${YELLOW}📦 Homebrew not found. Installing Homebrew...${NC}"
    echo -e "${CYAN}   This may take a few minutes and will prompt for your password.${NC}"
    echo ""
    
    # Install Homebrew
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    if [ $? -eq 0 ]; then
        # Add Homebrew to PATH for Apple Silicon Macs
        if [ -f /opt/homebrew/bin/brew ]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [ -f /usr/local/bin/brew ]; then
            eval "$(/usr/local/bin/brew shellenv)"
        fi
        
        echo -e "${GREEN}✓ Homebrew installed successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to install Homebrew${NC}"
        echo -e "${YELLOW}   Please install manually from: https://brew.sh${NC}"
        return 1
    fi
}

# Function to install Ollama via Homebrew
install_ollama() {
    if command_exists ollama; then
        return 0
    fi
    
    echo -e "${YELLOW}🤖 Ollama not found. Installing via Homebrew...${NC}"
    echo -e "${CYAN}   This may take a few minutes.${NC}"
    echo ""
    
    # Ensure Homebrew is available
    if ! command_exists brew; then
        install_homebrew
        if [ $? -ne 0 ]; then
            return 1
        fi
    fi
    
    # Install Ollama
    brew install ollama
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Ollama installed successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to install Ollama${NC}"
        echo -e "${YELLOW}   You can install manually from: https://ollama.ai${NC}"
        return 1
    fi
}

# Function to start Ollama service
start_ollama_service() {
    # Check if service is already running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        return 0
    fi
    
    echo -e "${YELLOW}🚀 Starting Ollama service...${NC}"
    
    # Start Ollama in background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for service to start (max 10 seconds)
    for i in {1..10}; do
        sleep 1
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ollama service started${NC}"
            return 0
        fi
    done
    
    echo -e "${YELLOW}⚠ Ollama service is starting in the background${NC}"
    echo -e "${CYAN}   It may take a moment to be ready.${NC}"
    return 0
}

# Function to ensure Ollama has a model
ensure_ollama_model() {
    # Check if any models are available
    MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null || echo "0")
    
    if [ "$MODELS" -gt 0 ]; then
        return 0
    fi
    
    echo -e "${YELLOW}📥 No Ollama models found. Downloading default model (llama3.2:3b)...${NC}"
    echo -e "${CYAN}   This is a one-time download (~2GB) and may take several minutes.${NC}"
    echo -e "${CYAN}   You can continue using the app - queries will use pattern matching until the model is ready.${NC}"
    echo ""
    
    # Download model in background
    ollama pull llama3.2:3b > /tmp/ollama_pull.log 2>&1 &
    
    echo -e "${GREEN}✓ Model download started in background${NC}"
    return 0
}

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

# Check persistent storage (DuckDB and user uploads)
echo ""
echo -e "${YELLOW}💾 Checking persistent storage...${NC}"

# Ensure data directory exists
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/data/uploads/raw"
mkdir -p "$PROJECT_ROOT/data/uploads/metadata"
mkdir -p "$PROJECT_ROOT/data/parquet"

# Check persistent DuckDB
DB_PATH="$PROJECT_ROOT/data/analytics.duckdb"
if [ -f "$DB_PATH" ]; then
    # Get database size
    DB_SIZE=$(du -h "$DB_PATH" 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "${GREEN}✓ Persistent DuckDB found (${DB_SIZE})${NC}"
else
    echo -e "${YELLOW}⚠ Persistent DuckDB not found (will be created on first upload)${NC}"
fi

# Check user uploads
UPLOADS_DIR="$PROJECT_ROOT/data/uploads"
METADATA_DIR="$UPLOADS_DIR/metadata"
if [ -d "$METADATA_DIR" ]; then
    # Count metadata files (each represents an upload)
    UPLOAD_COUNT=$(find "$METADATA_DIR" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$UPLOAD_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Found ${UPLOAD_COUNT} user upload(s)${NC}"
    else
        echo -e "${CYAN}ℹ No user uploads yet${NC}"
    fi
else
    echo -e "${CYAN}ℹ Uploads directory ready (no uploads yet)${NC}"
fi

# Check Parquet exports
PARQUET_DIR="$PROJECT_ROOT/data/parquet"
if [ -d "$PARQUET_DIR" ]; then
    PARQUET_COUNT=$(find "$PARQUET_DIR" -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$PARQUET_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✓ Found ${PARQUET_COUNT} Parquet export(s)${NC}"
    fi
fi

echo -e "${CYAN}   Storage directories ready${NC}"

# Check and install Ollama LLM service (self-contained, like DuckDB)
echo ""
echo -e "${YELLOW}🤖 Setting up Ollama LLM service...${NC}"
source .venv/bin/activate

# Check if Ollama is installed, install if not
if ! command_exists ollama; then
    install_ollama
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠ Ollama installation failed - continuing without LLM support${NC}"
        echo -e "${CYAN}   Natural language queries will use pattern matching only${NC}"
        OLLAMA_AVAILABLE=false
    else
        OLLAMA_AVAILABLE=true
    fi
else
    echo -e "${GREEN}✓ Ollama binary found${NC}"
    OLLAMA_AVAILABLE=true
fi

# Start Ollama service if available
if [ "$OLLAMA_AVAILABLE" = true ]; then
    start_ollama_service
    
    # Wait a moment for service to be ready
    sleep 2
    
    # Check if service is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ollama service running${NC}"
        
        # Check for models and download if needed
        MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null || echo "0")
        if [ "$MODELS" -gt 0 ]; then
            echo -e "${GREEN}✓ Ollama ready ($MODELS model(s) available)${NC}"
        else
            ensure_ollama_model
        fi
    else
        echo -e "${YELLOW}⚠ Ollama service not responding yet${NC}"
        echo -e "${CYAN}   It may still be starting. Queries will use pattern matching until ready.${NC}"
    fi
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

# Re-enable exit on error for the actual app run
set -e

# Run streamlit with the app (verbose mode for debugging)
source .venv/bin/activate
streamlit run src/clinical_analytics/ui/app.py \
    --logger.level=info \
    --server.fileWatcherType=poll
