"""Prompt service providing sample 30-second reading passages."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str
    duration_seconds: int
    topic: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "duration_seconds": self.duration_seconds,
            "topic": self.topic,
        }


class PromptService:
    """In-memory prompt catalogue used for practice recordings."""

    _PROMPTS: List[Prompt] = [
        Prompt(
            id="prompt-1",
            duration_seconds=30,
            topic="daily_routine",
            text=(
                "Every morning I brew a cup of coffee before the sun rises, letting the aroma fill the kitchen while the clock ticks quietly. "
                "As the kettle hums I stretch by the window, describe the dawn sky, and list three intentions for the day. "
                "By the time the first sip reaches my lips I am grounded, calm, and ready to speak with confidence."
            ),
        ),
        Prompt(
            id="prompt-2",
            duration_seconds=30,
            topic="travel",
            text=(
                "Imagine wandering through Lisbon's narrow alleys where the smell of warm pastries floats above the cobblestones. "
                "Describe the sound of soulful fado rising from cafés while the Atlantic breeze brushes bright ceramic tiles. "
                "Share how the city invites curious travelers to slow down, listen closely, and capture every detail."
            ),
        ),
        Prompt(
            id="prompt-3",
            duration_seconds=30,
            topic="technology",
            text=(
                "Technology moves fastest when teammates collaborate with curiosity and celebrate small wins together. "
                "Picture a Friday demo where prototypes spark debate, lessons from tiny failures are shared aloud, and bold ideas are recorded. "
                "Explain how those conversations shape the products that eventually delight real customers."
            ),
        ),
    ]

    def get_random_prompt(self) -> Prompt:
        return random.choice(self._PROMPTS)

    def get_prompt_by_id(self, prompt_id: str) -> Optional[Prompt]:
        for prompt in self._PROMPTS:
            if prompt.id == prompt_id:
                return prompt
        return None


__all__ = ["Prompt", "PromptService"]
