#!/bin/bash

# Clinical Analytics Platform - Desktop Launcher
# This script installs dependencies, manages Ollama, starts the FastAPI backend,
# and opens the Electron desktop app (or leaves the API running if Node is absent)
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

# Global variables for cleanup
OLLAMA_PID=""
API_PID=""
STOP_OLLAMA_ON_EXIT="${STOP_OLLAMA_ON_EXIT:-true}"

# Cleanup function - called on script exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down...${NC}"

    if [ -n "$API_PID" ]; then
        echo -e "${YELLOW}   Stopping FastAPI backend (PID: $API_PID)...${NC}"
        kill "$API_PID" 2>/dev/null || true
    fi

    # Stop Ollama if we started it AND flag is set
    if [ -n "$OLLAMA_PID" ] && [ "$STOP_OLLAMA_ON_EXIT" = "true" ]; then
        echo -e "${YELLOW}   Stopping Ollama service (PID: $OLLAMA_PID)...${NC}"
        kill "$OLLAMA_PID" 2>/dev/null || true

        # Also stop any running models gracefully
        if command -v ollama >/dev/null 2>&1; then
            # Get list of running models and stop them
            ollama ps --format json 2>/dev/null | \
                python3 -c "import sys, json; models=json.load(sys.stdin).get('models', []); [print(m['name']) for m in models]" 2>/dev/null | \
                while read -r model; do
                    if [ -n "$model" ]; then
                        echo -e "${CYAN}     Stopping model: $model${NC}"
                        ollama stop "$model" 2>/dev/null || true
                    fi
                done
        fi

        echo -e "${GREEN}✓ Ollama stopped${NC}"
    elif [ -n "$OLLAMA_PID" ]; then
        echo -e "${CYAN}ℹ Keeping Ollama service running (PID: $OLLAMA_PID)${NC}"
    fi

    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit 0
}

# Register cleanup on script exit
trap cleanup EXIT INT TERM

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

    # Only macOS installs through Homebrew; elsewhere never run a remote installer for the user.
    if [ "$(uname -s)" != "Darwin" ]; then
        echo -e "${YELLOW}🤖 Ollama not found. Install it from https://ollama.com/download, then re-run.${NC}"
        echo -e "${CYAN}   Simple questions still work without it.${NC}"
        return 1
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
        echo -e "${GREEN}✓ Ollama service already running${NC}"
        # Don't set OLLAMA_PID - we didn't start it
        return 0
    fi

    echo -e "${YELLOW}🚀 Starting Ollama service...${NC}"

    # Start Ollama in background (remove nohup to allow cleanup)
    ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!

    # Wait for service to start (max 10 seconds)
    for i in {1..10}; do
        sleep 1
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Ollama service started (PID: $OLLAMA_PID)${NC}"
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
    echo -e "${YELLOW}   Please run: ${GREEN}make install-dev${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Virtual environment found${NC}"

# Runtime data (uploads, DuckDB, session DB) lives in the per-user data home, never in the
# checkout. Report where it is so users know where patient data is stored.
echo ""
echo -e "${YELLOW}💾 Checking data home...${NC}"
DATA_HOME=$(uv run --no-sync python -c "from clinical_analytics.core.data_home import data_home; print(data_home())" 2>/dev/null || true)
if [ -n "$DATA_HOME" ]; then
    echo -e "${GREEN}✓ Data is stored in: ${DATA_HOME}${NC}"
    echo -e "${CYAN}  (set CLINICAL_ANALYTICS_DATA_HOME to change it; never copy patient files into this repo)${NC}"
else
    echo -e "${YELLOW}⚠ Could not resolve the data home; run 'make install-dev' first${NC}"
fi
if [ -d "$PROJECT_ROOT/data/uploads/metadata" ] && [ -n "$(find "$PROJECT_ROOT/data/uploads/metadata" -name '*.json' 2>/dev/null | head -1)" ]; then
    echo -e "${RED}⚠ Found uploads inside the repo (data/uploads). Move them to the data home and delete them here.${NC}"
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

# Start the FastAPI backend, then the Electron desktop app
API_URL="http://127.0.0.1:8000"
echo ""
echo -e "${GREEN}🎬 Starting FastAPI backend at ${API_URL} ...${NC}"
source .venv/bin/activate
# Refuse to start if something already answers on the port: the app would talk to it instead.
# Check IPv6 loopback too: a listener on [::1]:8000 could otherwise sit in front of the backend
# for any client that resolves "localhost" to ::1.
if curl -s --max-time 2 "${API_URL}/health" > /dev/null 2>&1 || curl -s -g --max-time 2 "http://[::1]:8000/health" > /dev/null 2>&1; then
    echo -e "${RED}✗ Another program is already using ${API_URL}.${NC}"
    echo -e "${YELLOW}   Close it (or the other copy of this app) and run this script again.${NC}"
    exit 1
fi
# Per-launch API token: the backend rejects requests without it (except /health), and the
# Electron app gets the same value from this environment. A new one is made every launch.
CLINICAL_ANALYTICS_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
if [ -z "$CLINICAL_ANALYTICS_API_TOKEN" ]; then
    echo -e "${RED}✗ Could not generate the API token${NC}"
    exit 1
fi
export CLINICAL_ANALYTICS_API_TOKEN
# Ollama was already handled above, so the API's own auto-init stays off here.
# serve.py runs uvicorn with log_config=None so its access/error logs go through redaction.
python -m clinical_analytics.api.serve --host 127.0.0.1 --port 8000 --log-level info &
API_PID=$!

for _ in $(seq 1 60); do
    if curl -s "${API_URL}/health" > /dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo -e "${RED}✗ FastAPI backend exited during startup (see log above)${NC}"
        exit 1
    fi
    sleep 1
done
if ! curl -s "${API_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}✗ FastAPI backend did not become healthy within 60s${NC}"
    exit 1
fi
# The answer must come from the backend started above, and it must know this launch's token.
if ! kill -0 "$API_PID" 2>/dev/null; then
    echo -e "${RED}✗ FastAPI backend exited during startup; another program may hold the port${NC}"
    exit 1
fi
if ! python -m clinical_analytics.api.security verify-health "${API_URL}"; then
    echo -e "${RED}✗ The server on ${API_URL} is not this launch's backend; not opening the app${NC}"
    exit 1
fi
echo -e "${GREEN}✓ FastAPI backend ready${NC}"

echo -e "${YELLOW}   Press Ctrl+C to stop${NC}"
echo ""

if command -v npm >/dev/null 2>&1 && [ -d electron/node_modules ]; then
    echo -e "${GREEN}🖥  Opening the desktop app (Electron)...${NC}"
    # This script already started the backend with this launch's token; tell the app to use it
    # instead of starting its own sidecar (which it does when launched on its own).
    CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1 npm --prefix electron start
else
    echo -e "${YELLOW}⚠ Electron dependencies not installed - desktop window not opened.${NC}"
    echo -e "${CYAN}   Install once with: cd electron && npm ci   then re-run this script.${NC}"
    echo -e "${CYAN}   The API stays up at ${API_URL} but requires this launch's token.${NC}"
    echo -e "${CYAN}   For direct API access during development use: make run-api (no token, dev mode).${NC}"
    wait "$API_PID"
fi
