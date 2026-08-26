#!/bin/bash
if [ "$1" = "web" ]; then
    echo "Starting Web UI on port 8000..."
    exec uvicorn src.api:app --host 0.0.0.0 --port 8000
else
    # Run the standard CLI pipeline
    exec python src/main.py "$@"
fi
