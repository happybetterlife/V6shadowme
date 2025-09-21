from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import uuid
import logging
from datetime import datetime

from app.core.security import get_current_user
from app.services.voice_cloning.voice_cloning_service import VoiceCloningService
from app.services.firebase.firebase_service import FirebaseService
from app.services.prompt.prompt_service import PromptService

# Optional import for voice cloning service
try:
    from app.services.openvoice_service import get_openvoice_service
    OPENVOICE_AVAILABLE = True
except ImportError:
    get_openvoice_service = None
    OPENVOICE_AVAILABLE = False

try:
    from app.models.voice.voice_model import VoiceModel, VoiceStatus
except ImportError:
    VoiceModel = None
    # Create a mock VoiceStatus for when the module is not available
    class VoiceStatus:
        PENDING = type('', (), {'value': 'pending'})()
        PROCESSING = type('', (), {'value': 'processing'})()
        COMPLETED = type('', (), {'value': 'completed'})()
        FAILED = type('', (), {'value': 'failed'})()
        CANCELLED = type('', (), {'value': 'cancelled'})()

logger = logging.getLogger(__name__)
router = APIRouter()
prompt_service = PromptService()

@router.post("/", response_model=Dict[str, Any])
async def create_voice_model(
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    name: str = Form(...),
    description: str = Form(""),
    audio_files: List[UploadFile] = File(...)
):
    """Create a new voice model"""
    # Check if OpenVoice is available
    if not OPENVOICE_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Voice cloning service is currently unavailable. OpenVoice module is not installed."
        )

    try:
        # Validate input
        if not name or len(name.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Voice model name must be at least 2 characters long"
            )
        
        if not audio_files or len(audio_files) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 3 audio files are required"
            )
        
        if len(audio_files) > 20:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 20 audio files allowed"
            )
        
        # Validate audio files
        for file in audio_files:
            if not file.content_type or not file.content_type.startswith('audio/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a valid audio file"
                )
        
        # Create voice model record
        model_id = str(uuid.uuid4())
        user_id = current_user["uid"]
        
        voice_model_data = {
            "id": model_id,
            "user_id": user_id,
            "name": name.strip(),
            "description": description.strip(),
            "status": VoiceStatus.PENDING.value,
            "created_at": datetime.utcnow().isoformat(),
            "quality": {
                "clarity": 0.0,
                "naturalness": 0.0,
                "similarity": 0.0,
                "overall": 0.0,
            },
            "sample_files": [],
            "model_config": {},
            "metrics": {
                "total_samples": 0,
                "total_duration": 0.0,
                "average_sample_length": 0.0,
                "primary_language": "en",
                "average_volume": 0.0,
                "average_pitch": 0.0,
            },
            "processing": {
                "started_at": None,
                "completed_at": None,
                "error_message": None,
            }
        }
        
        # Save to Firestore
        firebase_service = FirebaseService()
        await firebase_service.create_voice_model(voice_model_data)
        
        # Start background processing
        background_tasks.add_task(
            process_voice_model_background,
            model_id,
            user_id,
            audio_files
        )
        
        return {
            "message": "Voice model creation started",
            "model_id": model_id,
            "status": "pending",
            "estimated_processing_time": "5-10 minutes"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create voice model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create voice model"
        )

@router.get("/", response_model=List[Dict[str, Any]])
async def get_voice_models(
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0,
    status_filter: Optional[str] = None
):
    """Get user's voice models"""
    try:
        user_id = current_user["uid"]
        firebase_service = FirebaseService()
        
        voice_models = await firebase_service.get_user_voice_models(
            user_id=user_id,
            limit=limit,
            offset=offset,
            status_filter=status_filter
        )

        return voice_models
        
    except Exception as e:
        logger.error(f"Failed to get voice models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve voice models"
        )

@router.get("/{model_id}", response_model=Dict[str, Any])
async def get_voice_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get specific voice model"""
    try:
        user_id = current_user["uid"]
        firebase_service = FirebaseService()
        
        voice_model = await firebase_service.get_voice_model(model_id)
        
        if not voice_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice model not found"
            )
        
        # Check ownership
        if voice_model["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return voice_model
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voice model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve voice model"
        )

@router.post("/{model_id}/generate", response_model=Dict[str, Any])
async def generate_speech(
    model_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    text: str = Form(...),
    speed: float = Form(1.0),
    pitch: float = Form(0.0),
    emotion: Optional[str] = Form(None)
):
    """Generate speech using voice model"""
    try:
        # Validate input
        if not text or len(text.strip()) < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text is required"
            )
        
        if len(text) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text too long (max 1000 characters)"
            )
        
        if not 0.5 <= speed <= 2.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Speed must be between 0.5 and 2.0"
            )
        
        if not -12.0 <= pitch <= 12.0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Pitch must be between -12.0 and 12.0"
            )
        
        user_id = current_user["uid"]
        firebase_service = FirebaseService()
        
        # Check if model exists and user has access
        voice_model = await firebase_service.get_voice_model(model_id)
        if not voice_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice model not found"
            )
        
        if voice_model["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        if voice_model["status"] != VoiceStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Voice model is not ready for generation"
            )
        
        # Create generation record
        generation_id = str(uuid.uuid4())
        generation_data = {
            "id": generation_id,
            "user_id": user_id,
            "voice_model_id": model_id,
            "input_text": text.strip(),
            "output_file_path": None,
            "parameters": {
                "speed": speed,
                "pitch": pitch,
                "emotion": emotion,
            },
            "status": VoiceStatus.PENDING.value,
            "processing_time": None,
            "error_message": None,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }
        
        await firebase_service.create_voice_generation(generation_data)
        
        # Start background generation
        background_tasks.add_task(
            generate_speech_background,
            generation_id,
            model_id,
            text.strip(),
            speed,
            pitch,
            emotion
        )

        return {
            "message": "Speech generation started",
            "generation_id": generation_id,
            "status": "pending",
            "estimated_processing_time": "30-60 seconds"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate speech: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate speech"
        )

@router.delete("/{model_id}")
async def delete_voice_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete voice model"""
    try:
        user_id = current_user["uid"]
        firebase_service = FirebaseService()
        
        # Check if model exists and user has access
        voice_model = await firebase_service.get_voice_model(model_id)
        if not voice_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice model not found"
            )
        
        if voice_model["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Delete model
        await firebase_service.delete_voice_model(model_id)
        
        return {"message": "Voice model deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete voice model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete voice model"
        )

@router.get("/{model_id}/generations", response_model=List[Dict[str, Any]])
async def get_voice_generations(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    limit: int = 20,
    offset: int = 0
):
    """Get voice generation history for a model"""
    try:
        user_id = current_user["uid"]
        firebase_service = FirebaseService()
        
        # Check if model exists and user has access
        voice_model = await firebase_service.get_voice_model(model_id)
        if not voice_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice model not found"
            )
        
        if voice_model["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        generations = await firebase_service.get_voice_generations(
            model_id=model_id,
            limit=limit,
            offset=offset
        )
        
        return generations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get voice generations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve voice generations"
        )


@router.get("/prompt", response_model=Dict[str, Any])
async def get_recording_prompt(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Return a random 30-second practice prompt."""
    prompt = prompt_service.get_random_prompt()
    return prompt.to_dict()


@router.post("/recordings", response_model=Dict[str, Any])
async def submit_prompt_recording(
    current_user: Dict[str, Any] = Depends(get_current_user),
    audio_file: UploadFile = File(...),
    prompt_id: str = Form(...),
):
    """Accept a user recording and start voice cloning via OpenVoice."""

    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be an audio file",
        )

    prompt = prompt_service.get_prompt_by_id(prompt_id)
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prompt not found",
        )

    model_id = str(uuid.uuid4())
    user_id = current_user["uid"]

    firebase_service = FirebaseService()
    voice_model_data = {
        "id": model_id,
        "user_id": user_id,
        "name": f"Prompt recording {prompt.topic}",
        "description": prompt.text,
        "status": VoiceStatus.PROCESSING.value,
        "created_at": datetime.utcnow().isoformat(),
        "prompt": prompt.to_dict(),
        "quality": {
            "clarity": 0.0,
            "naturalness": 0.0,
            "similarity": 0.0,
            "overall": 0.0,
        },
        "sample_files": [],
        "model_config": {},
        "metrics": {
            "total_samples": 1,
            "total_duration": prompt.duration_seconds,
            "average_sample_length": prompt.duration_seconds,
            "primary_language": "en",
            "average_volume": 0.0,
            "average_pitch": 0.0,
        },
        "processing": {
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "error_message": None,
        },
    }

    await firebase_service.create_voice_model(voice_model_data)

    # Use OpenVoice service directly for recording processing
    if OPENVOICE_AVAILABLE and get_openvoice_service:
        openvoice_service = get_openvoice_service()
        if openvoice_service.is_available():
            # Save uploaded audio file temporarily
            import tempfile
            import os

            temp_dir = tempfile.mkdtemp()
            temp_audio_path = os.path.join(temp_dir, f"{model_id}_recording.wav")

            with open(temp_audio_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)

            # Create voice model using OpenVoice
            result = openvoice_service.create_voice_model(
                audio_files=[temp_audio_path],
                model_name=model_id
            )

            # Cleanup temp file
            os.unlink(temp_audio_path)
            os.rmdir(temp_dir)

            # Update voice model with results
            if result["success"]:
                voice_model_data["status"] = VoiceStatus.COMPLETED.value
                voice_model_data["processing"]["completed_at"] = datetime.utcnow().isoformat()
                voice_model_data["model_config"] = {
                    "openvoice_model_id": result["model_id"],
                    "embedding_path": result.get("embedding_path"),
                    "num_samples": result.get("num_samples", 1),
                    "num_valid_embeddings": result.get("num_valid_embeddings", 1)
                }
            else:
                voice_model_data["status"] = VoiceStatus.FAILED.value
                voice_model_data["processing"]["error_message"] = result.get("error", "Unknown error")
    else:
        # Fallback to mock processing if OpenVoice not available
        voice_model_data["status"] = VoiceStatus.COMPLETED.value
        voice_model_data["processing"]["completed_at"] = datetime.utcnow().isoformat()
        result = {"success": True, "model_id": model_id}

    # Update the voice model in Firebase
    await firebase_service.update_voice_model(model_id, voice_model_data)
    updated_model = await firebase_service.get_voice_model(model_id)

    return {
        "message": "Recording processed successfully",
        "model_id": model_id,
        "status": voice_model_data["status"],
        "prompt": prompt.to_dict(),
        "voice_model": updated_model,
        "openvoice_available": openvoice_service.is_available(),
    }

# Background tasks
async def process_voice_model_background(
    model_id: str,
    user_id: str,
    audio_files: List[UploadFile]
):
    """Background task to process voice model"""
    try:
        logger.info(f"Starting voice model processing for {model_id}")
        
        # Update status to processing
        firebase_service = FirebaseService()
        await firebase_service.update_voice_model_status(
            model_id,
            VoiceStatus.PROCESSING.value,
            processing_started_at=datetime.utcnow().isoformat()
        )
        
        # Process with voice cloning service
        voice_cloning_service = VoiceCloningService()
        result = await voice_cloning_service.create_voice_model(
            model_id=model_id,
            user_id=user_id,
            audio_files=audio_files
        )
        
        # Update model with results
        await firebase_service.update_voice_model_processing_result(
            model_id,
            result
        )
        
        logger.info(f"Voice model processing completed for {model_id}")
        
    except Exception as e:
        logger.error(f"Voice model processing failed for {model_id}: {e}")
        
        # Update status to failed
        try:
            firebase_service = FirebaseService()
            await firebase_service.update_voice_model_status(
                model_id,
                VoiceStatus.FAILED.value,
                error_message=str(e)
            )
        except Exception as update_error:
            logger.error(f"Failed to update error status: {update_error}")

async def generate_speech_background(
    generation_id: str,
    model_id: str,
    text: str,
    speed: float,
    pitch: float,
    emotion: Optional[str]
):
    """Background task to generate speech"""
    try:
        logger.info(f"Starting speech generation for {generation_id}")
        
        # Update status to processing
        firebase_service = FirebaseService()
        await firebase_service.update_voice_generation_status(
            generation_id,
            VoiceStatus.PROCESSING.value
        )
        
        # Generate speech
        voice_cloning_service = VoiceCloningService()
        result = await voice_cloning_service.generate_speech(
            model_id=model_id,
            text=text,
            speed=speed,
            pitch=pitch,
            emotion=emotion
        )
        
        # Update generation with results
        await firebase_service.update_voice_generation_result(
            generation_id,
            result
        )
        
        logger.info(f"Speech generation completed for {generation_id}")
        
    except Exception as e:
        logger.error(f"Speech generation failed for {generation_id}: {e}")
        
        # Update status to failed
        try:
            firebase_service = FirebaseService()
            await firebase_service.update_voice_generation_status(
                generation_id,
                VoiceStatus.FAILED.value,
                error_message=str(e)
            )
        except Exception as update_error:
            logger.error(f"Failed to update error status: {update_error}")
