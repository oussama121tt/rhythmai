#!/usr/bin/env python
"""Simple startup script for the backend"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Force SHA-256 hashing due to bcrypt compatibility issues
os.environ["FORCE_SIMPLE_HASH"] = "1"
subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--reload", "--port", "8000"])
