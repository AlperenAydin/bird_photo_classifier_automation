FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime system packages:
# - exiftool: required by pyexiftool
# - libglib2.0-0, libgomp1: required by OpenCV headless and PyTorch/C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    exiftool \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-install-project --no-dev

COPY . .

CMD ["uv", "run", "python", "src/immich_exif_tagger.py"]