# docker build -t ghcr.io/ir-datasets/local-ir-datasets:0.0.1-dev .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# ir-datasets v2 branch, pinned to a commit for reproducible rebuilds.
RUN pip install --no-cache-dir "git+https://github.com/ir-datasets/ir-datasets.git@07a9ad7eddbfb5c701c716ed815fd0b7b4003708"

# hf: / hf-local: datasets need these at resolve time.
RUN pip install --no-cache-dir "huggingface_hub" "datasets"

# Kept in sync with pyproject.toml's [project.optional-dependencies].test.
RUN pip install --no-cache-dir "coverage[toml]>=7.6" "pytest>=8.3" "tira"

WORKDIR /workspaces/local-ir-datasets
