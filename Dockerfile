FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# OpenVoice 설치 (GitHub에서)
RUN pip install git+https://github.com/myshell-ai/OpenVoice.git

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p data logs models checkpoints configs

# 포트 노출
EXPOSE 8000

# 환경 변수 설정
ENV PYTHONPATH="${PYTHONPATH}:/app"

# 애플리케이션 실행
CMD ["python", "main.py"]