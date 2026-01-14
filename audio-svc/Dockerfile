# ===============================
# Base Image
# ===============================
FROM nvidia/cuda:12.6.1-devel-ubuntu20.04

# ===============================
# Environment Variables
# ===============================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ===============================
# Install System Dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Install Python 3.10.0 from source
# ===============================
WORKDIR /tmp

RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip && \
    cd / && rm -rf /tmp/Python-3.10.0*

# ===============================
# Upgrade pip
# ===============================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ===============================
# Set Workdir
# ===============================
WORKDIR /app

# ===============================
# Install Python Dependencies
# ===============================
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ===============================
# Copy Application Code
# ===============================
COPY . /app

# ===============================
# Expose Port (按需修改)
# ===============================
EXPOSE 8000

# ===============================
# Run API
# ===============================
CMD ["python", "run_api.py"]
