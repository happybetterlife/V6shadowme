# VoiceShadow Architecture

## Overview

VoiceShadow is an AI-powered voice cloning and speech shadowing platform built with Flutter frontend and Python FastAPI backend. The application enables users to create voice models, generate speech, and practice pronunciation through shadowing exercises.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VoiceShadow Platform                     │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Flutter)          │  Backend (FastAPI)              │
│  ┌─────────────────────────┐ │  ┌─────────────────────────────┐ │
│  │ Authentication          │ │  │ API Endpoints              │ │
│  │ Onboarding              │ │  │ Voice Cloning Service      │ │
│  │ Voice Cloning           │ │  │ Speech Analysis Service    │ │
│  │ Shadowing Exercises     │ │  │ ML Model Management        │ │
│  │ Analytics & Progress    │ │  │ Firebase Integration       │ │
│  └─────────────────────────┘ │  └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Firebase Services                            │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ Authentication  │ │ Firestore DB    │ │ Cloud Storage   │   │
│  │ User Management │ │ Voice Models    │ │ Audio Files     │   │
│  │ Session Mgmt    │ │ User Progress   │ │ Generated Audio │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Frontend (Flutter)
- **Framework**: Flutter 3.10+
- **State Management**: Riverpod
- **UI Components**: Material Design 3
- **Audio**: Record, AudioPlayers
- **Firebase**: Core, Auth, Firestore, Storage
- **HTTP**: Dio
- **Architecture**: Feature-based with clean architecture

### Backend (Python FastAPI)
- **Framework**: FastAPI 0.104+
- **Database**: Firebase Firestore (primary), PostgreSQL (optional)
- **Authentication**: Firebase Auth + JWT
- **ML/AI**: PyTorch, TTS, ESPnet
- **Audio Processing**: Librosa, SoundFile, PyDub
- **Caching**: Redis
- **Deployment**: Docker, Nginx

### Infrastructure
- **Database**: Firebase Firestore
- **Storage**: Firebase Cloud Storage
- **Authentication**: Firebase Authentication
- **Hosting**: Firebase Hosting
- **CDN**: Firebase Hosting
- **Monitoring**: Firebase Analytics

## Project Structure

```
voiceshadow/
├── frontend/                 # Flutter App
│   ├── lib/
│   │   ├── main.dart
│   │   ├── core/             # Core utilities, constants, themes
│   │   ├── features/         # Feature-based modules
│   │   │   ├── auth/         # Authentication
│   │   │   ├── onboarding/   # User onboarding
│   │   │   ├── voice_cloning/# Voice model creation
│   │   │   ├── shadowing/    # Speech shadowing
│   │   │   └── analytics/    # Progress tracking
│   │   ├── shared/           # Shared widgets and providers
│   │   └── services/         # API and Firebase services
│   └── pubspec.yaml
│
├── backend/                  # Python FastAPI
│   ├── app/
│   │   ├── main.py          # Application entry point
│   │   ├── api/endpoints/   # API route handlers
│   │   ├── core/            # Configuration and security
│   │   ├── models/          # Data models
│   │   ├── services/        # Business logic
│   │   └── ml_models/       # ML model integration
│   └── requirements.txt
│
└── shared/                   # Shared configuration
    ├── firebase.json        # Firebase configuration
    ├── firestore.rules      # Database security rules
    └── storage.rules        # Storage security rules
```

## Key Features

### 1. Authentication & User Management
- Firebase Authentication integration
- Email/password and social login
- User profile management
- Secure session handling

### 2. Voice Cloning
- AI-powered voice model creation
- Multiple audio sample processing
- Voice quality assessment
- Real-time processing status

### 3. Speech Shadowing
- Interactive pronunciation practice
- Real-time speech analysis
- Progress tracking and feedback
- Personalized learning paths

### 4. Analytics & Progress
- User performance metrics
- Learning progress visualization
- Achievement tracking
- Detailed analytics dashboard

## API Endpoints

### Authentication
- `POST /api/v1/auth/signup` - User registration
- `POST /api/v1/auth/signin` - User login
- `POST /api/v1/auth/signout` - User logout
- `POST /api/v1/auth/reset-password` - Password reset

### Voice Cloning
- `POST /api/v1/voice-cloning/` - Create voice model
- `GET /api/v1/voice-cloning/` - Get user's voice models
- `GET /api/v1/voice-cloning/{model_id}` - Get specific model
- `POST /api/v1/voice-cloning/{model_id}/generate` - Generate speech
- `DELETE /api/v1/voice-cloning/{model_id}` - Delete model

### Speech Analysis
- `POST /api/v1/speech-analysis/` - Analyze speech
- `GET /api/v1/speech-analysis/{session_id}` - Get analysis results

### User Progress
- `GET /api/v1/user-progress/` - Get user progress
- `POST /api/v1/user-progress/` - Update progress

## Security

### Firebase Security Rules
- User-based data access control
- Secure file upload/download
- Authentication-based API access
- Rate limiting and abuse prevention

### Backend Security
- JWT token validation
- Firebase token verification
- Input validation and sanitization
- CORS configuration
- Rate limiting middleware

## Deployment

### Development
```bash
# Install dependencies
make install

# Setup environment
make setup

# Start development servers
make dev
```

### Production
```bash
# Build and deploy with Docker
make docker

# Deploy to Firebase
make deploy
```

### Docker Compose
- Multi-service container orchestration
- Nginx reverse proxy
- Redis caching
- PostgreSQL database (optional)
- Volume management for persistent data

## Performance Considerations

### Frontend
- Lazy loading of features
- Efficient state management with Riverpod
- Optimized audio processing
- Responsive UI design

### Backend
- Async/await for non-blocking operations
- Redis caching for frequently accessed data
- Background task processing
- Efficient file handling and streaming

### Infrastructure
- Firebase CDN for static assets
- Optimized Firestore queries
- Efficient audio file storage
- Real-time data synchronization

## Monitoring & Analytics

### Application Monitoring
- Firebase Analytics integration
- Error tracking and logging
- Performance metrics
- User engagement analytics

### Infrastructure Monitoring
- Health check endpoints
- Log aggregation
- Performance monitoring
- Resource utilization tracking

## Future Enhancements

### Planned Features
- Real-time collaboration
- Advanced voice effects
- Multi-language support
- Mobile app development
- API rate limiting improvements
- Advanced ML model integration

### Scalability
- Microservices architecture
- Load balancing
- Database sharding
- CDN optimization
- Auto-scaling capabilities

## Getting Started

1. **Clone the repository**
2. **Install dependencies**: `make install`
3. **Setup environment**: `make setup`
4. **Configure Firebase**: Update configuration files
5. **Start development**: `make dev`
6. **Access application**: http://localhost:3000

## Contributing

1. Follow the established architecture patterns
2. Maintain feature-based organization
3. Write comprehensive tests
4. Follow code formatting standards
5. Update documentation as needed

## License

This project is licensed under the MIT License.
