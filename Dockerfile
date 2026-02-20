# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (e.g., for building packages or common tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy the project files into the container
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY README.md ./
# Copy other necessary files
COPY Makefile ./

# Sync dependencies using uv
# --frozen ensures we use the exact versions in uv.lock
RUN uv sync --frozen

# Set environment variables
# Ensure python output is sent directly to terminal
ENV PYTHONUNBUFFERED=1
# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Default command (can be overridden)
CMD ["python", "src/main.py", "--help"]
