"""Lightweight voice persona service used by the API layer."""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Static persona catalogue sourced from the core persona definitions.
_PERSONA_DATA: Dict[str, Dict[str, Any]] = {
    "emily_us": {
        "metadata": {
            "id": "emily_us",
            "name": "Emily",
            "accent": "american",
            "personality": "friendly",
            "role": "conversation_partner",
            "gender": "female",
            "age_group": "young_adult",
            "specialty_contexts": ["casual_conversation", "daily_life", "hobbies"],
            "sample_voice_url": "/api/voice/sample/emily_us",
            "tagline": "Friendly • Beginners",
            "flag": "us",
            "recommended_for": ["beginners", "casual practice"],
            "avatar": "emily",
            "characteristics": {
                "pitch": 220.0,
                "speed": 1.0,
                "energy": 0.8,
                "clarity": 0.95,
                "warmth": 0.9,
            },
        },
        "sample_phrases": {
            "greeting": [
                "Hey there! How's it going?",
                "Hi! What's new with you today?",
                "Hello! Ready for some English practice?",
            ],
            "encouragement": [
                "You're doing great!",
                "That's awesome progress!",
                "Keep it up, you've got this!",
            ],
            "clarification": [
                "Could you say that again?",
                "I'm not sure I caught that.",
                "Can you rephrase that for me?",
            ],
        },
    },
    "james_uk": {
        "metadata": {
            "id": "james_uk",
            "name": "James",
            "accent": "british",
            "personality": "professional",
            "role": "formal_instructor",
            "gender": "male",
            "age_group": "adult",
            "specialty_contexts": ["business_english", "presentations", "formal_writing"],
            "sample_voice_url": "/api/voice/sample/james_uk",
            "tagline": "Formal • Business",
            "flag": "gb",
            "recommended_for": ["business english", "presentations"],
            "avatar": "james",
            "characteristics": {
                "pitch": 120.0,
                "speed": 0.95,
                "energy": 0.7,
                "clarity": 0.98,
                "warmth": 0.6,
            },
        },
        "sample_phrases": {
            "greeting": [
                "Good day! Shall we begin our session?",
                "Right then, let's get started.",
                "Excellent! I'm delighted to work with you today.",
            ],
            "correction": [
                "I beg your pardon, but there's a small error there.",
                "Actually, the correct form would be...",
                "May I suggest a slight adjustment?",
            ],
        },
    },
    "sarah_au": {
        "metadata": {
            "id": "sarah_au",
            "name": "Sarah",
            "accent": "australian",
            "personality": "energetic",
            "role": "enthusiastic_coach",
            "gender": "female",
            "age_group": "young_adult",
            "specialty_contexts": ["sports", "travel", "adventure", "motivation"],
            "sample_voice_url": "/api/voice/sample/sarah_au",
            "tagline": "Casual • Social",
            "flag": "au",
            "recommended_for": ["social english", "travel", "confidence"],
            "avatar": "sarah",
            "characteristics": {
                "pitch": 210.0,
                "speed": 1.05,
                "energy": 0.9,
                "clarity": 0.93,
                "warmth": 0.95,
            },
        },
        "sample_phrases": {
            "motivation": [
                "No worries, mate! You'll get it!",
                "Fair dinkum, that was brilliant!",
                "You're a real champion!",
            ],
            "encouragement": [
                "Keep pushing — you're smashing it!",
                "That answer was spot on!",
                "Let's give it another go together!",
            ],
        },
    },
    "michael_ca": {
        "metadata": {
            "id": "michael_ca",
            "name": "Michael",
            "accent": "canadian",
            "personality": "calm",
            "role": "academic_coach",
            "gender": "male",
            "age_group": "adult",
            "specialty_contexts": ["academic_english", "study_skills", "exam_prep"],
            "sample_voice_url": "/api/voice/sample/michael_ca",
            "tagline": "Academic • Study",
            "flag": "ca",
            "recommended_for": ["test prep", "academic writing"],
            "avatar": "michael",
            "characteristics": {
                "pitch": 130.0,
                "speed": 0.9,
                "energy": 0.65,
                "clarity": 0.96,
                "warmth": 0.7,
            },
        },
        "sample_phrases": {
            "study": [
                "Let's break this concept down together.",
                "I'll guide you through the academic phrasing.",
                "Take your time—we can review it step by step.",
            ],
            "feedback": [
                "That's a strong argument; let's support it with evidence.",
                "Consider refining the thesis to make it more precise.",
            ],
        },
    },
}


class PersonaService:
    """Manages persona metadata and per-user selections."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._personas = deepcopy(_PERSONA_DATA)
        self._user_selection: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return True

    def status(self) -> Dict[str, Optional[str]]:
        return {"available": True, "error": None}

    def get_available_personas(self) -> List[Dict[str, Any]]:
        return [deepcopy(data["metadata"]) for data in self._personas.values()]

    async def get_current_persona(self, user_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            persona_id = self._user_selection.get(user_id)
        if not persona_id:
            return None
        persona = self._personas.get(persona_id)
        return deepcopy(persona["metadata"]) if persona else None

    async def get_persona_sample_phrases(self, user_id: str, context: str) -> List[str]:
        async with self._lock:
            persona_id = self._user_selection.get(user_id)
        if not persona_id:
            return []
        persona = self._personas.get(persona_id)
        if not persona:
            return []
        phrases = persona["sample_phrases"].get(context, [])
        return list(phrases)

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    async def select_persona(self, user_id: str, persona_id: str) -> Dict[str, Any]:
        if persona_id not in self._personas:
            raise ValueError(f"Persona '{persona_id}' not found")
        async with self._lock:
            self._user_selection[user_id] = persona_id
        return deepcopy(self._personas[persona_id]["metadata"])

    async def recommend_persona(
        self,
        user_id: str,
        context: str,
        user_level: str,
        learning_goals: List[str],
    ) -> Dict[str, Any]:
        persona_id = self._score_personas(context, user_level, learning_goals)
        persona_meta = await self.select_persona(user_id, persona_id)
        return {
            "persona_id": persona_id,
            "persona": persona_meta,
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _score_personas(self, context: str, user_level: str, learning_goals: List[str]) -> str:
        context_lower = context.lower()
        goals_lower = [goal.lower() for goal in learning_goals]

        best_score = -1
        best_persona = next(iter(self._personas.keys()))

        for persona_id, data in self._personas.items():
            meta = data["metadata"]
            score = 0

            # Level alignment
            level = user_level.lower()
            if level == "beginner" and "beginners" in meta["recommended_for"]:
                score += 10
            elif level == "intermediate" and any(
                key in meta["recommended_for"] for key in ["conversation", "casual practice", "confidence"]
            ):
                score += 8
            elif level == "advanced" and any(
                key in meta["recommended_for"] for key in ["business english", "test prep", "academic writing"]
            ):
                score += 9

            # Context matching
            if context_lower:
                for speciality in meta["specialty_contexts"]:
                    if speciality in context_lower:
                        score += 5

            # Learning goals matching
            for goal in goals_lower:
                if goal in meta["recommended_for"] or goal in meta["specialty_contexts"]:
                    score += 3

            if score > best_score:
                best_score = score
                best_persona = persona_id

        return best_persona


# Global singleton instance
persona_service = PersonaService()
