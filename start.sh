#!/bin/bash

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OPENAI_API_KEY="your_api_key_here"
export DATABASE_URL="postgresql://user:password@localhost/english_learning"
export REDIS_URL="redis://localhost:6379"

# 가상환경 활성화
source venv/bin/activate

# 데이터베이스 마이그레이션
alembic upgrade head

# 서버 시작
python main.py