# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install only essential Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn[standard] pydantic pydantic-settings python-jose[cryptography] passlib[bcrypt] python-multipart httpx requests firebase-admin google-cloud-firestore sqlalchemy alembic python-dotenv aiofiles structlog pytrends feedparser aiohttp apscheduler beautifulsoup4 lxml aiosqlite

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p logs temp uploads models

# Expose port
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]