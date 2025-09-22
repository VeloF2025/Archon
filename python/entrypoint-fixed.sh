#!/bin/bash

# Install missing dependencies
pip install email-validator>=2.0.0

# Start the server
python -m uvicorn src.server.main:app --host 0.0.0.0 --port ${ARCHON_SERVER_PORT:-8181} --reload