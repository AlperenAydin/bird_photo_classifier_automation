FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy dependency files first for caching
COPY pyproject.toml uv.lock* ./

# Install project dependencies
RUN uv sync --frozen --no-install-project --no-dev

# Copy application source code
COPY . .

# Run application via uv
CMD ["uv", "run", "python", "src/immich_exif_tagger.py"]