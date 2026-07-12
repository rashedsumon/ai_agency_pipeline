# ==========================================
# Stage 1: Build & Dependency Installation
# ==========================================
FROM python:3.11-slim-bookworm AS builder

# Install uv system-wide
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy project configuration files first to leverage Docker layer caching
COPY pyproject.toml README.md /app/

# Synchronize dependencies (uv will automatically create a .venv here)
# --no-install-project prevents it from looking for 'src' code before we copy it
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ==========================================
# Stage 2: Final Runtime Image
# ==========================================
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy your actual application code into the image
COPY src /app/src

# Set up environment variables
# 1. Place the virtual environment's bin folder at the front of PATH
#    This makes the 'fastapi' and 'streamlit' commands work natively.
ENV PATH="/app/.venv/bin:$PATH"
# 2. Prevent Python from writing .pyc files to disk (reduces image clutter)
ENV PYTHONDONTWRITEBYTECODE=1
# 3. Force stdout/stderr streams to be unbuffered for clean real-time logging
ENV PYTHONUNBUFFERED=1

# Expose ports for both your FastAPI backend and Streamlit frontend
EXPOSE 8000
EXPOSE 8501

# The default command runs your FastAPI app using the correct syntax
CMD ["fastapi", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]
