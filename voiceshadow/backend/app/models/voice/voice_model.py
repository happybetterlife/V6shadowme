from sqlalchemy import Column, String, DateTime, Text, JSON, Float, Integer, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum as PyEnum

Base = declarative_base()

class VoiceStatus(PyEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

class VoiceModel(Base):
    """Voice model database table"""
    __tablename__ = "voice_models"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(Enum(VoiceStatus), default=VoiceStatus.PENDING)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Voice quality metrics
    clarity_score = Column(Float, default=0.0)
    naturalness_score = Column(Float, default=0.0)
    similarity_score = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)
    
    # Model metadata
    sample_files = Column(JSON)  # List of file paths
    generated_model_url = Column(String(500))
    model_config = Column(JSON)  # Model configuration parameters
    
    # Audio metrics
    total_samples = Column(Integer, default=0)
    total_duration = Column(Float, default=0.0)  # in seconds
    average_sample_length = Column(Float, default=0.0)  # in seconds
    primary_language = Column(String(10), default="en")
    average_volume = Column(Float, default=0.0)
    average_pitch = Column(Float, default=0.0)
    
    # Processing information
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value if self.status else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "quality": {
                "clarity": self.clarity_score,
                "naturalness": self.naturalness_score,
                "similarity": self.similarity_score,
                "overall": self.overall_score,
            },
            "sample_files": self.sample_files or [],
            "generated_model_url": self.generated_model_url,
            "model_config": self.model_config or {},
            "metrics": {
                "total_samples": self.total_samples,
                "total_duration": self.total_duration,
                "average_sample_length": self.average_sample_length,
                "primary_language": self.primary_language,
                "average_volume": self.average_volume,
                "average_pitch": self.average_pitch,
            },
            "processing": {
                "started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
                "completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
                "error_message": self.error_message,
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceModel":
        """Create model from dictionary"""
        model = cls()
        model.id = data.get("id")
        model.user_id = data.get("user_id")
        model.name = data.get("name")
        model.description = data.get("description")
        
        status_str = data.get("status")
        if status_str:
            model.status = VoiceStatus(status_str)
        
        # Quality metrics
        quality = data.get("quality", {})
        model.clarity_score = quality.get("clarity", 0.0)
        model.naturalness_score = quality.get("naturalness", 0.0)
        model.similarity_score = quality.get("similarity", 0.0)
        model.overall_score = quality.get("overall", 0.0)
        
        # Metadata
        model.sample_files = data.get("sample_files", [])
        model.generated_model_url = data.get("generated_model_url")
        model.model_config = data.get("model_config", {})
        
        # Audio metrics
        metrics = data.get("metrics", {})
        model.total_samples = metrics.get("total_samples", 0)
        model.total_duration = metrics.get("total_duration", 0.0)
        model.average_sample_length = metrics.get("average_sample_length", 0.0)
        model.primary_language = metrics.get("primary_language", "en")
        model.average_volume = metrics.get("average_volume", 0.0)
        model.average_pitch = metrics.get("average_pitch", 0.0)
        
        # Processing information
        processing = data.get("processing", {})
        if processing.get("started_at"):
            model.processing_started_at = datetime.fromisoformat(processing["started_at"])
        if processing.get("completed_at"):
            model.processing_completed_at = datetime.fromisoformat(processing["completed_at"])
        model.error_message = processing.get("error_message")
        
        return model

class VoiceSample(Base):
    """Voice sample database table"""
    __tablename__ = "voice_samples"
    
    id = Column(String, primary_key=True, index=True)
    voice_model_id = Column(String, nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer)
    duration = Column(Float)  # in seconds
    sample_rate = Column(Integer)
    channels = Column(Integer)
    bit_rate = Column(Integer)
    
    # Audio analysis
    volume_level = Column(Float)
    pitch_level = Column(Float)
    noise_level = Column(Float)
    clarity_score = Column(Float)
    
    # Metadata
    language = Column(String(10), default="en")
    transcription = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary"""
        return {
            "id": self.id,
            "voice_model_id": self.voice_model_id,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_rate": self.bit_rate,
            "audio_analysis": {
                "volume_level": self.volume_level,
                "pitch_level": self.pitch_level,
                "noise_level": self.noise_level,
                "clarity_score": self.clarity_score,
            },
            "language": self.language,
            "transcription": self.transcription,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

class VoiceGeneration(Base):
    """Voice generation history table"""
    __tablename__ = "voice_generations"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    voice_model_id = Column(String, nullable=False, index=True)
    input_text = Column(Text, nullable=False)
    output_file_path = Column(String(500))
    
    # Generation parameters
    speed = Column(Float, default=1.0)
    pitch = Column(Float, default=0.0)
    emotion = Column(String(50))
    
    # Generation status
    status = Column(Enum(VoiceStatus), default=VoiceStatus.PENDING)
    processing_time = Column(Float)  # in seconds
    error_message = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert generation to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "voice_model_id": self.voice_model_id,
            "input_text": self.input_text,
            "output_file_path": self.output_file_path,
            "parameters": {
                "speed": self.speed,
                "pitch": self.pitch,
                "emotion": self.emotion,
            },
            "status": self.status.value if self.status else None,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
