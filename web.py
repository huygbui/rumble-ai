"""Compatibility entrypoint for the FastAPI app.

Run with:
  uvicorn web:app
  python web.py
"""

from app.main import app, main


if __name__ == "__main__":
    main()
