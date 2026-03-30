# =============================================================================
# Multi-stage Dockerfile — Flight Delay Prediction API
# =============================================================================
# Stage 1: build / install deps
# Stage 2: lean runtime image

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed by some packages (e.g. lightgbm, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system deps (libgomp for LightGBM/XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin \
                    /usr/local/bin

# Copy project source
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 appuser \
    && mkdir -p final_model prediction_output monitoring Logs \
    && chown -R appuser:appuser /app

USER appuser

# Copy and register entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["api"]
