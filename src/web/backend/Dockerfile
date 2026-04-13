FROM python:3.11-slim

WORKDIR /app

# Enable caching and install system dependencies for soundfile and librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies directly (simpler configuration)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    librosa \
    soundfile \
    psycopg2-binary \
    pgvector \
    pydantic \
    python-multipart \
    numpy

# Copy source code
COPY . /app/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
