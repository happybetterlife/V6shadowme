# Multi-stage Dockerfile for VoiceShadow
FROM python:3.11-slim as backend

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ .

# Create necessary directories
RUN mkdir -p uploads models logs temp

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Flutter frontend stage
FROM ubuntu:22.04 as flutter-base

# Install Flutter dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    xz-utils \
    zip \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Install Flutter
ENV FLUTTER_VERSION=3.16.0
RUN git clone https://github.com/flutter/flutter.git -b ${FLUTTER_VERSION} /flutter
ENV PATH="/flutter/bin:/flutter/bin/cache/dart-sdk/bin:${PATH}"

# Enable Flutter web
RUN flutter config --enable-web
RUN flutter doctor

# Frontend build stage
FROM flutter-base as frontend

WORKDIR /app

# Copy Flutter project
COPY frontend/ .

# Get dependencies
RUN flutter pub get

# Build for web
RUN flutter build web --release

# Production stage
FROM nginx:alpine as production

# Copy built Flutter web app
COPY --from=frontend /app/build/web /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy backend
COPY --from=backend /app /backend

# Install Python in nginx image for backend
RUN apk add --no-cache python3 py3-pip

# Expose ports
EXPOSE 80 8000

# Start both nginx and backend
COPY start.sh /start.sh
RUN chmod +x /start.sh
CMD ["/start.sh"]
