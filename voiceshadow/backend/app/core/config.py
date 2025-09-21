from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    # App Configuration
    APP_NAME: str = "VoiceShadow API"
    DEBUG: bool = False
    VERSION: str = "1.0.0"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Firebase Configuration
    FIREBASE_PROJECT_ID: str = "v7-shadowme"
    FIREBASE_PRIVATE_KEY_ID: str = ""
    FIREBASE_PRIVATE_KEY: str = ""
    FIREBASE_CLIENT_EMAIL: str = ""
    FIREBASE_CLIENT_ID: str = ""
    FIREBASE_AUTH_URI: str = "https://accounts.google.com/o/oauth2/auth"
    FIREBASE_TOKEN_URI: str = "https://oauth2.googleapis.com/token"
    FIREBASE_AUTH_PROVIDER_X509_CERT_URL: str = ""
    FIREBASE_CLIENT_X509_CERT_URL: str = ""
    FIREBASE_CREDENTIALS_PATH: str = "./firebase-credentials.json"

    # Environment Settings
    APP_ENV: str = "development"

    # Database Configuration
    USE_SQLITE_DB: bool = True
    OFFLINE_MODE: bool = False

    # Feature Flags
    ENABLE_TREND_COLLECTION: bool = True
    ENABLE_SCHEDULER: bool = True
    ENABLE_GOOGLE_TRENDS: bool = False
    ENABLE_NEWS_API: bool = True
    ENABLE_REDDIT_API: bool = False
    USE_MOCK_DATA: bool = False

    # Trend Collection Settings
    TREND_COLLECTION_INTERVAL: int = 2
    TRENDS_CACHE_DURATION: int = 3600
    
    # Storage Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_AUDIO_FORMATS: List[str] = [".wav", ".mp3", ".flac", ".m4a"]
    
    # ML Model Configuration
    MODEL_DIR: str = "models"
    VOICE_CLONING_MODEL: str = "tacotron2"
    SPEECH_RECOGNITION_MODEL: str = "whisper"
    MAX_AUDIO_DURATION: int = 300  # 5 minutes
    
    # Redis Configuration (for caching)
    REDIS_URL: Optional[str] = None
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Database Configuration
    DATABASE_URL: Optional[str] = None
    
    # External API Keys
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_CLOUD_API_KEY: Optional[str] = None
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.UPLOAD_DIR,
            self.MODEL_DIR,
            "logs",
            "temp"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def firebase_credentials(self) -> dict:
        """Get Firebase credentials as a dictionary"""
        return {
            "type": "service_account",
            "project_id": self.FIREBASE_PROJECT_ID,
            "private_key_id": self.FIREBASE_PRIVATE_KEY_ID,
            "private_key": self.FIREBASE_PRIVATE_KEY.replace("\\n", "\n"),
            "client_email": self.FIREBASE_CLIENT_EMAIL,
            "client_id": self.FIREBASE_CLIENT_ID,
            "auth_uri": self.FIREBASE_AUTH_URI,
            "token_uri": self.FIREBASE_TOKEN_URI,
            "auth_provider_x509_cert_url": self.FIREBASE_AUTH_PROVIDER_X509_CERT_URL,
            "client_x509_cert_url": self.FIREBASE_CLIENT_X509_CERT_URL,
        }

# Create settings instance
settings = Settings()

# Validate settings
def validate_settings():
    """Validate critical settings"""
    required_settings = [
        "SECRET_KEY",
    ]
    
    # Firebase settings are optional for development
    optional_firebase_settings = [
        "FIREBASE_PROJECT_ID",
        "FIREBASE_PRIVATE_KEY",
        "FIREBASE_CLIENT_EMAIL",
    ]
    
    missing_settings = []
    for setting in required_settings:
        if not getattr(settings, setting):
            missing_settings.append(setting)
    
    # Only require Firebase settings if project ID is provided
    if settings.FIREBASE_PROJECT_ID:
        for setting in optional_firebase_settings:
            if not getattr(settings, setting):
                missing_settings.append(setting)
    
    if missing_settings:
        print(f"Warning: Missing optional settings: {', '.join(missing_settings)}")
        # Don't raise error for development, just warn
    
    return True

# Validate on import
validate_settings()
