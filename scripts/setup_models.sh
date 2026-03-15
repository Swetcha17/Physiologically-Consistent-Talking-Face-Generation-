#!/bin/bash
# Setup: Clone SadTalker + download face parsing weights
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== rPPG Talking-Face Model Setup ==="

# 1. SadTalker
if [ ! -d "SadTalker" ]; then
    echo "[1/2] Cloning SadTalker..."
    git clone https://github.com/OpenTalker/SadTalker.git
    cd SadTalker
    pip install -r requirements.txt
    bash scripts/download_models.sh
    cd "$PROJECT_ROOT"
else
    echo "[1/2] SadTalker already exists."
fi

# 2. Face Parsing weights
if [ ! -f "models/face_parsing_79999_iter.pth" ]; then
    echo "[2/2] Downloading BiSeNet face parsing weights..."
    mkdir -p models
    if command -v gdown &> /dev/null; then
        gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face_parsing_79999_iter.pth
    else
        echo "  Install gdown: pip install gdown"
        echo "  Then run: gdown 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O models/face_parsing_79999_iter.pth"
        echo "  Or download from: https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812"
    fi
else
    echo "[2/2] Face parsing weights already exist."
fi

echo ""
echo "=== Verification ==="
[ -d "SadTalker" ] && echo "[OK] SadTalker" || echo "[MISSING] SadTalker"
[ -f "models/face_parsing_79999_iter.pth" ] && echo "[OK] Face parsing weights" || echo "[MISSING] Face parsing weights"
echo "Done."
