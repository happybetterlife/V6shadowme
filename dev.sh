#!/bin/bash

# Voice Shadow - Development Mode
echo "🔧 Starting Voice Shadow in Development Mode..."

# 환경 변수 설정 (개발용)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DEBUG=True
export LOG_LEVEL=DEBUG

# 개발 모드 환경변수
export OPENAI_API_KEY="sk-test-key"  # 테스트용
export VOICE_SAMPLE_RATE=22050
export WHISPER_MODEL=base

# 가상환경 활성화
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Please run ./run.sh first to setup environment"
    exit 1
fi

# 개발용 의존성 설치
echo "📦 Installing development dependencies..."
pip install pytest pytest-asyncio black flake8

# 코드 포맷팅
echo "🎨 Formatting code..."
black . --exclude venv

# 린트 체크
echo "🔍 Running linter..."
flake8 . --exclude=venv --max-line-length=100 --ignore=E203,W503

# 테스트 실행 (있다면)
if [ -d "tests" ]; then
    echo "🧪 Running tests..."
    pytest tests/
fi

# Hot reload로 서버 시작
echo "🔥 Starting development server with hot reload..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 --log-level debug