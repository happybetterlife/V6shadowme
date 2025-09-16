"""
애플리케이션 설정
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    
    # Database
    database_url: str = "sqlite:///./app.db"
    
    # Voice Processing
    voice_sample_rate: int = 22050
    voice_max_duration: int = 30  # seconds
    
    # Learning Parameters
    default_difficulty: str = "intermediate"
    max_corrections_per_session: int = 5
    shadowing_sentences_per_level: int = 10
    
    # Audio Processing
    bone_conduction_enabled: bool = True
    audio_chunk_size: int = 1024
    
    # Session Management
    session_timeout_minutes: int = 60
    max_active_sessions: int = 100
    
    # Development
    debug: bool = False
    log_level: str = "INFO"
    
    # CORS
    allowed_origins: list = ["*"]
    
    # File Paths
    models_path: str = "./models"
    data_path: str = "./data"
    logs_path: str = "./logs"
    
    # OpenVoice Settings
    openvoice_checkpoint: str = "checkpoints/openvoice_v2.pth"
    openvoice_config: str = "configs/openvoice_v2.json"
    
    # Whisper Settings
    whisper_model: str = "base"
    
    class Config:
        env_file = ".env"
        case_sensitive = False