FROM python:3.11-slim

WORKDIR /app

# Install python dependencies (use repo root requirements.txt)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the FastAPI app needs at runtime
COPY backend ./backend
COPY model ./model
COPY credit_score.py ./credit_score.py

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "backend.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
