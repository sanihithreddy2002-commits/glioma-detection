#!/usr/bin/env python3
"""
Run script for Glioma Detection System
Starts the FastAPI server
"""

import uvicorn
import os
import sys

def main():
    print("="*70)
    print("GLIOMA DETECTION SYSTEM".center(70))
    print("="*70)
    print()
    print("Starting server...")
    print()
    print("  API Server:  http://localhost:8000")
    print("  UI:          http://localhost:8000")
    print("  API Docs:    http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop")
    print("="*70)
    print()
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()