#!/bin/bash

# Docker Compose 실행 스크립트
echo "🐳 Voice Shadow - Docker Deployment"

# 환경 변수 체크
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY is not set. Please set it in your environment or .env file."
    echo "   Example: export OPENAI_API_KEY='sk-your-key-here'"
    exit 1
fi

# 실행 모드 선택
MODE=${1:-prod}

case $MODE in
    "dev")
        echo "🔧 Starting in Development Mode..."
        docker-compose -f docker-compose.dev.yml up --build
        ;;
    "prod")
        echo "🚀 Starting in Production Mode..."
        
        # 필요한 디렉토리 생성
        mkdir -p data logs ssl
        
        # SSL 디렉토리가 비어있으면 self-signed 인증서 생성 (개발용)
        if [ ! -f ssl/cert.pem ]; then
            echo "🔒 Generating self-signed SSL certificate..."
            openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes \
                -subj "/C=US/ST=State/L=City/O=Organization/OU=OrgUnit/CN=localhost"
        fi
        
        docker-compose up --build -d
        
        echo "✅ Services started!"
        echo "📊 Check status: docker-compose ps"
        echo "📋 View logs: docker-compose logs -f"
        echo "🌐 Access app: http://localhost"
        echo "🔍 Health check: http://localhost/health"
        ;;
    "stop")
        echo "⏹️  Stopping services..."
        docker-compose down
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        ;;
    *)
        echo "Usage: $0 [dev|prod|stop|clean]"
        echo "  dev   - Development mode with hot reload"
        echo "  prod  - Production mode with nginx (default)"
        echo "  stop  - Stop all services"
        echo "  clean - Stop and remove all containers, volumes, and images"
        exit 1
        ;;
esac