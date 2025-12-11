# 使用 Python 3.10 基础镜像 (PaddleOCR 兼容性较好)
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (PaddleOCR 和 OpenCV 需要)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
# 注意：使用清华源加速，并单独安装 paddlepaddle
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 如果 requirements.txt 里没有 paddlepaddle，需要取消下面这行的注释
# RUN pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目代码
COPY . .

# 创建输出目录
RUN mkdir -p docs

# 设置环境变量 (如果有 ERNIE Token，建议通过 docker run -e 传入，不要写死)
# ENV ERNIE_ACCESS_TOKEN="your_token_here"

# 运行命令
CMD ["python", "src/main.py"]