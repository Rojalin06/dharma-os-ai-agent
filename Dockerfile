FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir openai fastapi uvicorn pydantic
EXPOSE 7860
CMD ["python", "-m", "server.app"]
