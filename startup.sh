#!/bin/bash
cd /home/site/wwwroot
# Force fresh deployment - Dec 15 2025
python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}