#!/bin/bash

# Start nginx in background
nginx -g "daemon on;" &

# Start backend
cd /backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
