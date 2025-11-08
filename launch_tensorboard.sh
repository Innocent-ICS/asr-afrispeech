#!/bin/bash

# Launch TensorBoard Script
# This script starts TensorBoard to view experiment logs

echo "=================================="
echo "Launching TensorBoard"
echo "=================================="
echo ""
echo "TensorBoard will be available at: http://localhost:6006"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

# Activate conda environment if needed
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using conda environment: $CONDA_DEFAULT_ENV"
else
    echo "Note: Make sure tensorboard is installed in your environment"
fi

# Launch TensorBoard
tensorboard --logdir=logs/tensorboard --port=6006

