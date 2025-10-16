#!/usr/bin/env bash
set -e
echo "Running flake8..."
flake8 src tests || true
echo "Lint completed."