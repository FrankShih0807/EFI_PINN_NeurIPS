#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Cleaning caches/artifacts in: $ROOT_DIR"

# Python caches
find . -type d -name "__pycache__" -prune -print -exec rm -rf {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -print -delete

# macOS metadata
find . -type f -name ".DS_Store" -print -delete

# Build artifacts
rm -rf build dist *.egg-info 2>/dev/null || true

echo "Done."


