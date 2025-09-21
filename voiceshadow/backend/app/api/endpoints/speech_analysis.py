from fastapi import APIRouter, Depends
from typing import Dict, Any

from app.core.security import get_current_user

router = APIRouter()


@router.post("/analyze")
async def analyze_speech_stub(
    payload: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    return {
        "message": "Speech analysis stubbed response",
        "input": payload,
        "user": current_user.get("uid"),
        "scores": {
            "clarity": 0.6,
            "pronunciation": 0.65,
            "intonation": 0.62,
        },
    }
