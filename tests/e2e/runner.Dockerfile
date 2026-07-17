# Test-runner image: Python 3.9 + the exact repository dependency lock.
FROM ghcr.io/astral-sh/uv:python3.9-bookworm-slim

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Install dependencies from the lock first so code changes reuse this layer.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --group dev

COPY . .

CMD ["uv", "run", "--frozen", "pytest", "tests/e2e", "-m", "e2e", "-v"]
