# ShadowME - AI Voice Shadowing Platform

A modern AI-powered voice shadowing application built with Flutter frontend and Python FastAPI backend, designed based on Figma V5-ShadowME design system.

## 🎯 Features

### Core Features
- **3-Step Onboarding**: Welcome, Voice Setup, and Practice introduction
- **Voice Setup**: 30-second voice recording for AI voice model creation
- **Shadowing Practice**: Interactive pronunciation practice with real-time feedback
- **Progress Analytics**: Detailed performance tracking and insights
- **Responsive Design**: Optimized for both mobile and tablet devices

### Technical Features
- **Flutter 3.x**: Latest Flutter framework with Material Design 3
- **Riverpod State Management**: Modern state management solution
- **Firebase Integration**: Authentication, Firestore, and Cloud Storage
- **Python FastAPI**: High-performance backend with ML integration
- **Real-time Audio Processing**: Advanced audio recording and analysis
- **Responsive UI**: Adaptive design for different screen sizes

## 🏗️ Architecture

```
ShadowME Platform
├── Frontend (Flutter)           ├── Backend (FastAPI)
│   ├── Design System            │   ├── API Endpoints
│   ├── Onboarding (3 steps)     │   ├── Voice Cloning Service
│   ├── Voice Setup              │   ├── Speech Analysis
│   ├── Dashboard                │   └── ML Model Integration
│   ├── Shadowing Practice       │
│   └── Progress Analytics       ├── Firebase Services
├── Shared Configuration         │   ├── Authentication
└── Docker & Deployment          │   ├── Firestore Database
                                 │   └── Cloud Storage
```

## 🚀 Getting Started

### Prerequisites
- Flutter SDK 3.10+
- Python 3.8+
- Firebase project setup
- Docker (optional)

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd voiceshadow

# Setup development environment
make setup

# Start development servers
make dev
```

### Manual Setup

#### Frontend Setup
```bash
cd frontend
flutter pub get
flutter run -d web-server --web-port 3000
```

#### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Firebase Setup
```bash
# Install Firebase CLI
npm install -g firebase-tools

# Login and initialize
firebase login
firebase init

# Start emulators
firebase emulators:start
```

## 📱 App Flow

1. **Authentication**: Sign up/Sign in with Firebase Auth
2. **Onboarding**: 3-step introduction to ShadowME features
3. **Voice Setup**: Record 30 seconds for AI voice model creation
4. **Dashboard**: Main hub with progress overview and quick actions
5. **Shadowing Practice**: Interactive pronunciation exercises
6. **Progress Analytics**: Detailed performance insights and achievements

## 🎨 Design System

### ShadowME V5 Design System
- **Colors**: Primary (Indigo), Secondary (Purple), Accent (Cyan/Emerald)
- **Typography**: Inter font family with consistent sizing scale
- **Components**: Custom ShadowME buttons, cards, and progress indicators
- **Animations**: Smooth transitions and micro-interactions
- **Responsive**: Mobile-first design with tablet optimizations

### Key Components
- `ShadowMEButton`: Custom button with multiple variants
- `ShadowMECard`: Flexible card component with different types
- `ShadowMEProgressCard`: Progress tracking component
- `ResponsiveUtils`: Utility for responsive design

## 🔧 Development

### Available Commands
```bash
make install      # Install dependencies
make setup        # Setup development environment
make dev          # Start development servers
make build        # Build for production
make deploy       # Deploy to Firebase
make clean        # Clean build artifacts
make test         # Run tests
make lint         # Run linting
make format       # Format code
```

### Project Structure
```
frontend/
├── lib/
│   ├── core/                    # Core utilities and theme
│   ├── features/                # Feature-based modules
│   │   ├── auth/               # Authentication
│   │   ├── onboarding/         # 3-step onboarding
│   │   ├── voice_setup/        # Voice recording setup
│   │   ├── dashboard/          # Main dashboard
│   │   ├── shadowing/          # Practice sessions
│   │   └── analytics/          # Progress tracking
│   ├── shared/                 # Shared widgets and components
│   └── services/               # API and Firebase services
```

## 🔐 Configuration

### Environment Variables
Copy and update the example configuration files:
```bash
cp backend/env.example backend/.env
cp frontend/android/app/google-services.json.example frontend/android/app/google-services.json
cp frontend/ios/Runner/GoogleService-Info.plist.example frontend/ios/Runner/GoogleService-Info.plist
```

### Firebase Configuration
1. Create a Firebase project
2. Enable Authentication, Firestore, and Storage
3. Download configuration files
4. Update the example files with your actual values

## 🐳 Docker Deployment

```bash
# Build and run with Docker
make docker

# Or manually
docker-compose up --build
```

## 📊 Performance

- **Frontend**: Optimized Flutter web build with lazy loading
- **Backend**: Async FastAPI with background task processing
- **Database**: Efficient Firestore queries with proper indexing
- **Storage**: Optimized audio file handling and streaming
- **Caching**: Redis integration for improved performance

## 🤝 Contributing

1. Follow the established architecture patterns
2. Maintain feature-based organization
3. Write comprehensive tests
4. Follow code formatting standards (Black, isort, dart format)
5. Update documentation as needed

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Design based on Figma V5-ShadowME specifications
- Built with Flutter and Python FastAPI
- Powered by Firebase and AI/ML technologies
