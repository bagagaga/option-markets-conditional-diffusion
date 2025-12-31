FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    build-essential \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir poetry==1.7.1
COPY . .

RUN pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
ENV HYDRA_FULL_ERROR=1

CMD ["bash"]
