FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Hugging Face Docker Spaces route to port 7860 by default.
EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
