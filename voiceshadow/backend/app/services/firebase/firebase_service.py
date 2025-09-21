"""In-memory FirebaseService stub used for local development.

This module replaces the production Firebase integration so that the
FastAPI application can start without external dependencies. The stub keeps
state in simple dictionaries and mimics the async interface expected by the
rest of the codebase.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from app.models.voice.voice_model import VoiceStatus

logger = logging.getLogger(__name__)


class FirebaseService:
    """Minimal stand-in for the Firebase service layer."""

    _initialized: bool = False
    _voice_models: Dict[str, Dict[str, Any]] = {}
    _voice_generations: Dict[str, Dict[str, Any]] = {}

    def __init__(self) -> None:
        if not self.__class__._initialized:
            logger.warning(
                "FirebaseService used before initialize(); calling initialize() automatically"
            )
            self.__class__.initialize()

    @classmethod
    def initialize(cls) -> None:
        """Initialise the stub service."""
        if not cls._initialized:
            logger.info("FirebaseService stub initialised (in-memory store)")
            cls._initialized = True

    # Voice model helpers -------------------------------------------------
    async def create_voice_model(self, data: Dict[str, Any]) -> None:
        self._voice_models[data["id"]] = data

    async def get_user_voice_models(
        self,
        *,
        user_id: str,
        limit: int,
        offset: int,
        status_filter: Optional[str],
    ) -> List[Dict[str, Any]]:
        models = [
            model
            for model in self._voice_models.values()
            if model.get("user_id") == user_id
        ]
        if status_filter:
            models = [
                model
                for model in models
                if model.get("status") == status_filter
            ]
        return models[offset : offset + limit]

    async def get_voice_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self._voice_models.get(model_id)

    async def delete_voice_model(self, model_id: str) -> None:
        self._voice_models.pop(model_id, None)
        # Remove related generations as well
        self._voice_generations = {
            key: value
            for key, value in self._voice_generations.items()
            if value.get("voice_model_id") != model_id
        }

    async def update_voice_model_status(
        self,
        model_id: str,
        status: str,
        *,
        processing_started_at: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        model = self._voice_models.get(model_id)
        if not model:
            return
        model["status"] = status
        processing = model.setdefault("processing", {})
        if processing_started_at:
            processing["started_at"] = processing_started_at
        if error_message:
            processing["error_message"] = error_message
        if status == VoiceStatus.FAILED.value:
            processing["completed_at"] = datetime.utcnow().isoformat()

    async def update_voice_model_processing_result(
        self,
        model_id: str,
        result: Dict[str, Any],
    ) -> None:
        model = self._voice_models.get(model_id)
        if not model:
            return
        model.update(result)
        model["status"] = VoiceStatus.COMPLETED.value
        processing = model.setdefault("processing", {})
        processing["completed_at"] = datetime.utcnow().isoformat()
        processing["error_message"] = None

    # Voice generation helpers -------------------------------------------
    async def create_voice_generation(self, data: Dict[str, Any]) -> None:
        self._voice_generations[data["id"]] = data

    async def get_voice_generations(
        self,
        *,
        model_id: str,
        limit: int,
        offset: int,
    ) -> List[Dict[str, Any]]:
        generations = [
            generation
            for generation in self._voice_generations.values()
            if generation.get("voice_model_id") == model_id
        ]
        return generations[offset : offset + limit]

    async def update_voice_generation_status(
        self,
        generation_id: str,
        status: str,
        *,
        error_message: Optional[str] = None,
    ) -> None:
        generation = self._voice_generations.get(generation_id)
        if not generation:
            return
        generation["status"] = status
        if error_message:
            generation["error_message"] = error_message
        if status in {VoiceStatus.COMPLETED.value, VoiceStatus.FAILED.value}:
            generation["completed_at"] = datetime.utcnow().isoformat()

    async def update_voice_generation_result(
        self,
        generation_id: str,
        result: Dict[str, Any],
    ) -> None:
        generation = self._voice_generations.get(generation_id)
        if not generation:
            return
        generation.update(result)
        generation["status"] = VoiceStatus.COMPLETED.value
        generation["completed_at"] = datetime.utcnow().isoformat()
        generation["error_message"] = None
