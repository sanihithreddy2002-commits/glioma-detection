import os
import sys
import uvicorn

def main():
    print("=" * 70)
    print("GLIOMA DETECTION SYSTEM".center(70))
    print("=" * 70)
    print()
    print("Starting server...")
    print()
    print(" API Server : http://localhost")
    print(" API Docs   : /docs")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()

    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8000)),
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
