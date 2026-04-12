"""Run the FastAPI dev server from the repo root: `python3 app.py` (implementation in backend/asgi.py)."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "backend" / "asgi.py"), run_name="__main__")
