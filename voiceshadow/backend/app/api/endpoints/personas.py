"""Endpoints for managing voice personas."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.security import get_current_user
from app.services.persona_service import persona_service

router = APIRouter()


class PersonaSelectionRequest(BaseModel):
    persona_id: str = Field(..., description="Identifier of the persona to select")


class PersonaRecommendationRequest(BaseModel):
    context: str = Field("", description="Conversation context or topic")
    user_level: str = Field("intermediate", description="Learner proficiency level")
    learning_goals: List[str] = Field(
        default_factory=list,
        description="List of learning goals (e.g., pronunciation, business english)",
    )


@router.get("/", response_model=Dict[str, Any])
async def list_personas(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Return available personas along with system status information."""
    personas = persona_service.get_available_personas()
    status_info = persona_service.status()
    return {
        "status": status_info,
        "personas": personas,
    }


@router.post("/select", response_model=Dict[str, Any])
async def select_persona(
    request: PersonaSelectionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Select a persona for the current user."""
    if not persona_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persona system is unavailable in this environment",
        )

    try:
        persona_info = await persona_service.select_persona(current_user["uid"], request.persona_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    return {
        "message": "Persona selected successfully",
        "persona": persona_info,
    }


@router.get("/current", response_model=Dict[str, Any])
async def get_current_persona(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get the currently selected persona for the user."""
    if not persona_service.is_available():
        return {
            "persona": None,
            "status": persona_service.status(),
        }

    persona_info = await persona_service.get_current_persona(current_user["uid"])
    return {
        "persona": persona_info,
    }


@router.post("/recommend", response_model=Dict[str, Any])
async def recommend_persona(
    request: PersonaRecommendationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Recommend a persona based on learner context and goals."""
    if not persona_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persona system is unavailable in this environment",
        )

    recommendation = await persona_service.recommend_persona(
        current_user["uid"],
        request.context,
        request.user_level,
        request.learning_goals,
    )

    return recommendation


@router.get("/samples/{context}", response_model=Dict[str, Any])
async def persona_sample_phrases(
    context: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Return sample phrases for the selected persona in a given context."""
    if not persona_service.is_available():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Persona system is unavailable in this environment",
        )

    phrases = await persona_service.get_persona_sample_phrases(current_user["uid"], context)
    return {
        "context": context,
        "phrases": phrases,
    }
