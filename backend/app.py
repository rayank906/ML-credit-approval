"""
Compatibility entrypoint: `python app.py` runs the FastAPI dev server.
Implementation lives in `asgi.py` (not `main.py`, which would collide with the repo root data script).
"""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "asgi.py"), run_name="__main__")
