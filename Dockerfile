# Dockerfile for AI-Powered FAQ System
# Multi-stage build for optimized image size

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc g++ make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); \
    nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.10-slim

# Non-root user for security
RUN useradd -m -u 1000 -s /bin/bash faquser && \
    mkdir -p /app /data /models /home/faquser/nltk_data && \
    chown -R faquser:faquser /app /data /models /home/faquser/nltk_data

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy Python packages and NLTK data from builder
COPY --from=builder /root/.local /home/faquser/.local
COPY --from=builder /root/nltk_data /home/faquser/nltk_data

ENV PATH=/home/faquser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NLTK_DATA=/home/faquser/nltk_data

# Copy application code
COPY --chown=faquser:faquser app/       /app/app/
COPY --chown=faquser:faquser ml/        /app/ml/
COPY --chown=faquser:faquser data/      /app/data/
COPY --chown=faquser:faquser saved_models/ /app/saved_models/

RUN mkdir -p /app/logs && chown -R faquser:faquser /app/logs

USER faquser
WORKDIR /app

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
