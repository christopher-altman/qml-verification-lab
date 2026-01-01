#!/usr/bin/env bash
# Generate curated demo artifacts for qml-verification-lab
# This creates a minimal, stable artifact set for documentation purposes

set -e

DEMO_DIR="artifacts_demo"
EXPERIMENT_ID="toy_identifiability_demo"

echo "Generating demo artifacts..."
echo "Output directory: $DEMO_DIR/$EXPERIMENT_ID"

# Clean existing demo
rm -rf "$DEMO_DIR/$EXPERIMENT_ID"

# Run a small sweep showing the identifiability phenomenon
# This generates ~10 parameter points across noise dimensions
python -m qvl sweep \
  --config examples/toy_smoke.yaml \
  --output-dir "$DEMO_DIR" \
  --seeds 0,1 2>&1 | head -20

echo ""
echo "Demo artifacts generated successfully."
echo "Location: $DEMO_DIR/$EXPERIMENT_ID/"
echo ""
echo "Contents:"
find "$DEMO_DIR/$EXPERIMENT_ID" -type f | head -20
