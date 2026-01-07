#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT="${1:-neurips_submission.zip}"

echo "Creating submission zip: $OUT"

# Ensure we don't accidentally include a previous archive
rm -f "$OUT"

# Zip the working tree while excluding git history and common junk.
zip -rq "$OUT" . \
  -x ".git/*" \
  -x "**/__pycache__/*" \
  -x "**/*.pyc" -x "**/*.pyo" \
  -x "**/.DS_Store" \
  -x "build/*" -x "dist/*" -x "**/*.egg-info/*" \
  -x "$OUT"

echo "Done: $(du -h "$OUT" | awk '{print $1}')  $OUT"


