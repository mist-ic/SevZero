FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first for cache efficiency
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# --- Runtime stage ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY models.py .
COPY server/ server/
COPY inference.py .
COPY openenv.yaml .
COPY pyproject.toml .
COPY README.md .

# Use the venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Non-root user
RUN useradd -m appuser
USER appuser

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
