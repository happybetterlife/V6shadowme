from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime

from app.core.security import get_current_user

router = APIRouter()


@router.get("/")
async def get_progress_stub(current_user: Dict[str, Any] = Depends(get_current_user)):
    uid = current_user.get("uid")
    return {
        "user": uid,
        "summary": {
            "sessions_this_week": 3,
            "total_minutes": 120,
            "improvement_score": 0.4,
        },
        "last_updated": datetime.utcnow().isoformat(),
    }
