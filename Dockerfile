# --- Stage 1: 基础镜像 ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# --- Stage 2: 环境配置 ---
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# --- Stage 3: 系统及工具安装 ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        build-essential \
        libcudnn9-cuda-12 \
        python3-dev \
        libgl1-mesa-glx \
        libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置 pip 全局使用清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# --- Stage 4: 项目配置与依赖安装 ---
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 使用清华镜像源安装依赖
RUN pip install --no-cache-dir \
    -r requirements.txt

# 复制项目文件
COPY . .

# 赋予启动脚本执行权限
RUN chmod +x start.sh

# --- Stage 5: 容器运行配置 ---
EXPOSE 12020
EXPOSE 12021

CMD ["./start.sh"]    