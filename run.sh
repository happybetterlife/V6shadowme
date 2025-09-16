#!/bin/bash

# Voice Shadow - AI English Learning App 실행 스크립트
echo "🎯 Starting Voice Shadow AI English Learning App..."

# 현재 디렉토리를 Python 경로에 추가
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# .env 파일이 있는지 확인
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys"
fi

# 가상환경 확인 및 활성화
if [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# 필요한 디렉토리 생성
mkdir -p models data logs checkpoints configs

# 데이터베이스 초기화 (SQLite)
if [ ! -f "app.db" ]; then
    echo "🗄️  Initializing database..."
    python -c "
import asyncio
from database.manager import DatabaseManager

async def init_db():
    db = DatabaseManager()
    await db.initialize()
    print('✅ Database initialized')

asyncio.run(init_db())
"
fi

echo "🚀 Starting server..."
python main.py