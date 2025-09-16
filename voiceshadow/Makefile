# VoiceShadow Makefile
.PHONY: help install setup dev build deploy clean test lint format

# Default target
help:
	@echo "VoiceShadow - AI Voice Cloning & Shadowing Platform"
	@echo ""
	@echo "Available commands:"
	@echo "  install     - Install all dependencies"
	@echo "  setup       - Setup development environment"
	@echo "  dev         - Start development servers"
	@echo "  build       - Build for production"
	@echo "  deploy      - Deploy to production"
	@echo "  clean       - Clean build artifacts"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting"
	@echo "  format      - Format code"
	@echo "  docker      - Build and run with Docker"
	@echo "  firebase    - Deploy Firebase configuration"

# Install dependencies
install:
	@echo "Installing dependencies..."
	cd backend && pip install -r requirements.txt
	cd frontend && flutter pub get
	@echo "Dependencies installed successfully!"

# Setup development environment
setup: install
	@echo "Setting up development environment..."
	cp backend/env.example backend/.env
	cp frontend/android/app/google-services.json.example frontend/android/app/google-services.json
	cp frontend/ios/Runner/GoogleService-Info.plist.example frontend/ios/Runner/GoogleService-Info.plist
	@echo "Please update the configuration files with your actual values"
	@echo "Setup completed!"

# Start development servers
dev:
	@echo "Starting ShadowME development servers..."
	@echo "Backend will be available at http://localhost:8000"
	@echo "Frontend will be available at http://localhost:3000"
	@echo "Firebase emulators will be available at http://localhost:4000"
	@echo ""
	@echo "Starting Firebase emulators..."
	firebase emulators:start --only auth,firestore,storage &
	@echo "Starting backend server..."
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	@echo "Starting frontend server..."
	cd frontend && flutter run -d web-server --web-port 3000 --web-hostname 0.0.0.0 &
	@echo "All servers started!"

# Build for production
build:
	@echo "Building for production..."
	cd frontend && flutter build web --release
	@echo "Build completed!"

# Deploy to production
deploy: build
	@echo "Deploying to production..."
	firebase deploy --only hosting,firestore,storage
	@echo "Deployment completed!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	cd frontend && flutter clean
	cd backend && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	cd backend && find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean completed!"

# Run tests
test:
	@echo "Running tests..."
	cd backend && python -m pytest tests/ -v
	cd frontend && flutter test
	@echo "Tests completed!"

# Run linting
lint:
	@echo "Running linting..."
	cd backend && flake8 app/ --max-line-length=100
	cd backend && mypy app/ --ignore-missing-imports
	cd frontend && flutter analyze
	@echo "Linting completed!"

# Format code
format:
	@echo "Formatting code..."
	cd backend && black app/ --line-length=100
	cd backend && isort app/ --profile black
	cd frontend && dart format .
	@echo "Formatting completed!"

# Docker commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting Docker containers..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose logs -f

docker: docker-build docker-up

# Firebase commands
firebase-init:
	@echo "Initializing Firebase..."
	firebase init

firebase-deploy:
	@echo "Deploying Firebase configuration..."
	firebase deploy --only firestore,storage

firebase-emulators:
	@echo "Starting Firebase emulators..."
	firebase emulators:start

firebase: firebase-deploy

# Development shortcuts
backend:
	@echo "Starting backend server..."
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	@echo "Starting frontend server..."
	cd frontend && flutter run -d web-server --web-port 3000

# Database commands
db-migrate:
	@echo "Running database migrations..."
	cd backend && alembic upgrade head

db-reset:
	@echo "Resetting database..."
	cd backend && alembic downgrade base && alembic upgrade head

# Utility commands
logs:
	@echo "Showing application logs..."
	tail -f logs/app.log

status:
	@echo "Checking service status..."
	@curl -s http://localhost:8000/health || echo "Backend not running"
	@curl -s http://localhost:3000 || echo "Frontend not running"
	@curl -s http://localhost:4000 || echo "Firebase emulators not running"
