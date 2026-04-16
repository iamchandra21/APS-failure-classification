FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    ca-certificates \
 && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
 && unzip awscliv2.zip \
 && ./aws/install \
 && rm -rf aws awscliv2.zip \
 && rm -rf /var/lib/apt/lists/*


# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first — this layer is cached until deps change
COPY pyproject.toml requirements.txt ./

# Install all dependencies into the system Python (no venv needed inside Docker)
RUN uv pip install --system -r requirements.txt

# Copy the rest of the source code
COPY . /app

# Install the local sensor package (editable-style, system-wide)
RUN uv pip install --system -e .

CMD ["python3", "main.py"]