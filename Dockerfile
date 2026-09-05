FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY . .

WORKDIR /app/services/api

RUN uv sync --locked --no-dev --no-editable

CMD ["sh", "-c", "uv run uvicorn risk_api.main:app --host 0.0.0.0 --port ${PORT}"]
