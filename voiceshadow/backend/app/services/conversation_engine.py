"""Conversation engine - integrates Cornell DB, PersonaChat, and trends."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import openai  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    openai = None  # type: ignore

try:  # pragma: no cover
    from database.cornell_db import CornellDatabase  # type: ignore
    from database.personachat_db import PersonaChatDatabase  # type: ignore
    from database.trends_db import TrendsDatabase  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - stub replacements
    class CornellDatabase:  # type: ignore
        async def find_related_patterns(self, keywords: List[str]) -> List[Dict[str, str]]:
            await asyncio.sleep(0)
            return [{"dialogue": f"Sample dialogue about {', '.join(keywords)}"}]

    class PersonaChatDatabase:  # type: ignore
        async def match_persona(self, profile: Dict[str, str]) -> Dict[str, str]:
            await asyncio.sleep(0)
            return {"persona": "Friendly traveler", "style": "casual"}

    class TrendsDatabase:  # type: ignore
        async def initialize(self) -> None:
            await asyncio.sleep(0)


class ConversationEngine:
    def __init__(self) -> None:
        self.cornell_db = CornellDatabase()
        self.personachat_db = PersonaChatDatabase()
        self.trends_db = TrendsDatabase()
        self.sessions: Dict[str, Dict[str, object]] = {}

    async def initialize_session(self, user_id: str) -> Dict[str, object]:
        user_profile = await self.load_user_profile(user_id)
        await self.trends_db.initialize()
        trending_topic = await self.select_trending_topic(user_profile)

        cornell_patterns = await self.cornell_db.find_related_patterns(trending_topic["keywords"])
        persona_context = await self.personachat_db.match_persona(user_profile)

        session = {
            "id": self.generate_session_id(),
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "topic": trending_topic,
            "cornell_patterns": cornell_patterns,
            "persona_context": persona_context,
            "conversation_history": [],
            "error_log": [],
            "learning_points": [],
        }

        self.sessions[session["id"]] = session
        return session

    async def generate_response(self, user_input: str, session_id: str) -> Dict[str, object]:
        session = self.sessions[session_id]
        context = self.build_context(session)

        prompt = (
            f"You are an English teacher discussing: {session['topic']['title']}\n\n"
            f"Context from movie dialogues:\n{json.dumps(session['cornell_patterns'][:3], indent=2)}\n\n"
            f"Persona context:\n{json.dumps(session['persona_context'], indent=2)}\n\n"
            f"Conversation history:\n{self.format_history(session['conversation_history'][-5:])}\n\n"
            f"User said: {user_input}\n\n"
            "Generate a natural, educational response that introduces relevant vocabulary, gently corrects errors, and matches the user's level."
        )

        if openai is None:
            logger.warning("openai module unavailable; returning stub response")
            ai_response = "Let's continue discussing this topic in English!"
        else:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a friendly English teacher."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            ai_response = response.choices[0].message.content  # type: ignore

        session["conversation_history"].append(
            {"user": user_input, "ai": ai_response, "timestamp": datetime.utcnow().isoformat()}
        )

        return {
            "text": ai_response,
            "session_id": session_id,
            "topic_reference": session["topic"]["title"],
        }

    async def select_trending_topic(self, user_profile: Dict[str, object]) -> Dict[str, object]:
        topic = await self.trends_db.get_todays_topic(user_profile)
        topic.setdefault("title", topic.get("name", "Trending topic"))
        topic.setdefault("keywords", self.extract_keywords(topic["title"]))
        topic.setdefault("difficulty", self.assess_difficulty(topic["title"]))
        topic.setdefault("relevance_score", topic.get("popularity_score", 0.5))
        return topic

    # ------------------------------------------------------------------
    # Helper methods ---------------------------------------------------
    # ------------------------------------------------------------------

    async def load_user_profile(self, user_id: str) -> Dict[str, object]:
        await asyncio.sleep(0)
        return {"id": user_id, "level": "intermediate", "interests": ["travel", "technology"]}

    def generate_session_id(self) -> str:
        return f"conv-{int(datetime.utcnow().timestamp())}"

    def build_context(self, session: Dict[str, object]) -> str:
        history = session["conversation_history"]
        context = f"Topic: {session['topic']['title']}\nUser Level: {session.get('user_level', 'intermediate')}\n"
        if history:
            context += "Recent conversation:\n"
            for turn in history[-3:]:
                context += f"User: {turn['user']}\nAI: {turn['ai']}\n"
        return context

    def format_history(self, history: List[Dict[str, object]]) -> str:
        lines = []
        for turn in history:
            lines.append(f"User: {turn['user']}")
            lines.append(f"AI: {turn['ai']}")
        return "\n".join(lines)

    async def calculate_relevance_score(self, trend: str, interests: List[str]) -> float:
        await asyncio.sleep(0)
        score = sum(1 for interest in interests if interest.lower() in trend.lower())
        return float(score)

    def extract_keywords(self, trend: str) -> List[str]:
        return [part.strip().lower() for part in trend.split() if part.isalpha()]

    def assess_difficulty(self, trend: str) -> str:
        length = len(trend.split())
        if length <= 3:
            return "easy"
        if length <= 6:
            return "intermediate"
        return "advanced"


__all__ = ["ConversationEngine"]
