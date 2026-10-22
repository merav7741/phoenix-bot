FROM node:22-slim AS frontend-build

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm install --global npm@11.21.1 && npm ci

COPY frontend/ .
RUN npm run build

FROM python:3.12-slim AS app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-build /frontend/dist ./frontend/dist

EXPOSE 10000

CMD ["sh", "-c", "python scripts/prepare_index.py && exec uvicorn server.api:app --host 0.0.0.0 --port ${PORT:-10000}"]
