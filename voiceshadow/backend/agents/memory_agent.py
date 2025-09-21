"""
Memory Agent

Manages conversation history, user context, and learning preferences.
This agent maintains persistent memory of user interactions and preferences.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path


class MemoryAgent:
    """Agent for managing conversation memory and user context"""

    def __init__(self):
        self.user_contexts = {}
        self.conversation_history = {}
        self.user_preferences = {}
        self.context_data_file = Path("data/user_contexts.json")
        self.history_data_file = Path("data/conversation_history.json")
        self.preferences_data_file = Path("data/user_preferences.json")
        self.max_history_length = 50  # Maximum conversations to keep per user
        self.context_expire_days = 30  # Days before context expires

    async def initialize(self) -> None:
        """Initialize the memory agent"""
        await self.load_all_data()
        print("Memory Agent initialized")

    async def load_all_data(self) -> None:
        """Load all persistent data from files"""
        await asyncio.gather(
            self.load_contexts(),
            self.load_conversation_history(),
            self.load_preferences()
        )

    async def load_contexts(self) -> None:
        """Load user contexts from file"""
        try:
            if self.context_data_file.exists():
                with open(self.context_data_file, 'r') as f:
                    self.user_contexts = json.load(f)
                await self.clean_expired_contexts()
            else:
                self.context_data_file.parent.mkdir(parents=True, exist_ok=True)
                self.user_contexts = {}
        except Exception as e:
            print(f"Error loading user contexts: {e}")
            self.user_contexts = {}

    async def load_conversation_history(self) -> None:
        """Load conversation history from file"""
        try:
            if self.history_data_file.exists():
                with open(self.history_data_file, 'r') as f:
                    self.conversation_history = json.load(f)
                await self.clean_old_history()
            else:
                self.conversation_history = {}
        except Exception as e:
            print(f"Error loading conversation history: {e}")
            self.conversation_history = {}

    async def load_preferences(self) -> None:
        """Load user preferences from file"""
        try:
            if self.preferences_data_file.exists():
                with open(self.preferences_data_file, 'r') as f:
                    self.user_preferences = json.load(f)
            else:
                self.user_preferences = {}
        except Exception as e:
            print(f"Error loading user preferences: {e}")
            self.user_preferences = {}

    async def save_contexts(self) -> None:
        """Save user contexts to file"""
        try:
            with open(self.context_data_file, 'w') as f:
                json.dump(self.user_contexts, f, indent=2)
        except Exception as e:
            print(f"Error saving user contexts: {e}")

    async def save_history(self) -> None:
        """Save conversation history to file"""
        try:
            with open(self.history_data_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation history: {e}")

    async def save_preferences(self) -> None:
        """Save user preferences to file"""
        try:
            with open(self.preferences_data_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving user preferences: {e}")

    async def load_context(self, user_id: str) -> Dict[str, Any]:
        """Load context for a specific user"""
        if user_id not in self.user_contexts:
            await self.initialize_user_context(user_id)

        context = self.user_contexts[user_id].copy()

        # Add recent conversation history
        recent_history = await self.get_recent_conversations(user_id, limit=10)
        context["recent_history"] = recent_history

        # Add user preferences
        preferences = self.user_preferences.get(user_id, {})
        context["preferences"] = preferences

        # Update last accessed time
        context["last_accessed"] = datetime.now().isoformat()
        self.user_contexts[user_id] = context

        return context

    async def initialize_user_context(self, user_id: str) -> None:
        """Initialize context for a new user"""
        self.user_contexts[user_id] = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "conversation_count": 0,
            "current_topics": [],
            "interests": [],
            "learning_goals": [],
            "difficulty_preferences": "adaptive",
            "preferred_conversation_style": "casual",
            "vocabulary_focus_areas": [],
            "grammar_focus_areas": [],
            "cultural_background": "unknown",
            "timezone": "UTC",
            "session_data": {}
        }

        # Initialize empty history and preferences
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = self.get_default_preferences()

        await self.save_contexts()

    def get_default_preferences(self) -> Dict[str, Any]:
        """Get default user preferences"""
        return {
            "conversation_length": "medium",  # short, medium, long
            "feedback_frequency": "moderate",  # low, moderate, high
            "error_correction_style": "gentle",  # strict, moderate, gentle
            "topic_preferences": ["general", "current_events"],
            "avoid_topics": [],
            "preferred_practice_areas": ["speaking", "listening"],
            "native_language": "unknown",
            "learning_style": "conversational",  # formal, conversational, mixed
            "personality_match": "friendly",  # professional, friendly, casual
            "challenge_level": "moderate"  # easy, moderate, challenging
        }

    async def store_conversation(
        self,
        user_id: str,
        user_input: str,
        response: Dict[str, Any],
        errors: List[Dict[str, Any]]
    ) -> None:
        """Store a conversation in memory"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response,
            "errors": errors,
            "session_id": response.get("session_id", "unknown"),
            "conversation_id": len(self.conversation_history[user_id]) + 1,
            "metadata": {
                "response_time": response.get("response_time", 0),
                "confidence_score": response.get("confidence", 0.5),
                "topic": response.get("topic", "general"),
                "difficulty_level": response.get("difficulty_level", "intermediate")
            }
        }

        self.conversation_history[user_id].append(conversation_entry)

        # Limit history length
        if len(self.conversation_history[user_id]) > self.max_history_length:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history_length:]

        # Update user context
        await self.update_context_from_conversation(user_id, user_input, response, errors)

        # Save data
        await asyncio.gather(
            self.save_history(),
            self.save_contexts()
        )

    async def update_context_from_conversation(
        self,
        user_id: str,
        user_input: str,
        response: Dict[str, Any],
        errors: List[Dict[str, Any]]
    ) -> None:
        """Update user context based on conversation"""
        if user_id not in self.user_contexts:
            await self.initialize_user_context(user_id)

        context = self.user_contexts[user_id]
        context["conversation_count"] += 1
        context["last_accessed"] = datetime.now().isoformat()

        # Extract and update topics
        topic = response.get("topic", "general")
        if topic not in context["current_topics"]:
            context["current_topics"].append(topic)
            # Keep only recent topics (max 10)
            context["current_topics"] = context["current_topics"][-10:]

        # Update vocabulary focus based on errors
        for error in errors:
            if error.get("type") == "vocabulary":
                word = error.get("word", "")
                if word and word not in context["vocabulary_focus_areas"]:
                    context["vocabulary_focus_areas"].append(word)
                    # Keep only recent focus areas (max 20)
                    context["vocabulary_focus_areas"] = context["vocabulary_focus_areas"][-20:]

        # Update grammar focus based on errors
        for error in errors:
            if error.get("type") == "grammar":
                pattern = error.get("pattern", "")
                if pattern and pattern not in context["grammar_focus_areas"]:
                    context["grammar_focus_areas"].append(pattern)
                    # Keep only recent focus areas (max 15)
                    context["grammar_focus_areas"] = context["grammar_focus_areas"][-15:]

        # Update interests based on user input analysis
        interests = await self.extract_interests_from_input(user_input)
        for interest in interests:
            if interest not in context["interests"]:
                context["interests"].append(interest)
                # Keep only recent interests (max 15)
                context["interests"] = context["interests"][-15:]

    async def extract_interests_from_input(self, user_input: str) -> List[str]:
        """Extract potential interests from user input"""
        interests = []
        input_lower = user_input.lower()

        # Simple keyword-based interest extraction
        interest_keywords = {
            "technology": ["computer", "software", "app", "tech", "digital", "ai", "robot"],
            "sports": ["football", "soccer", "basketball", "tennis", "gym", "exercise", "run"],
            "music": ["music", "song", "band", "concert", "guitar", "piano", "sing"],
            "travel": ["travel", "trip", "vacation", "country", "city", "airport", "hotel"],
            "food": ["food", "restaurant", "cook", "recipe", "meal", "eat", "dinner"],
            "movies": ["movie", "film", "cinema", "actor", "series", "tv", "watch"],
            "books": ["book", "read", "novel", "author", "library", "story"],
            "nature": ["nature", "park", "hiking", "mountain", "beach", "tree", "animal"],
            "art": ["art", "paint", "draw", "museum", "gallery", "design", "creative"],
            "science": ["science", "research", "study", "experiment", "discovery", "lab"]
        }

        for interest, keywords in interest_keywords.items():
            if any(keyword in input_lower for keyword in keywords):
                interests.append(interest)

        return interests

    async def get_recent_conversations(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations for a user"""
        if user_id not in self.conversation_history:
            return []

        history = self.conversation_history[user_id]
        return history[-limit:] if history else []

    async def update_preferences(self, user_id: str, feedback: Dict[str, Any]) -> None:
        """Update user preferences based on feedback"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = self.get_default_preferences()

        preferences = self.user_preferences[user_id]

        # Update preferences based on feedback
        if "conversation_length_preference" in feedback:
            preferences["conversation_length"] = feedback["conversation_length_preference"]

        if "error_correction_preference" in feedback:
            preferences["error_correction_style"] = feedback["error_correction_preference"]

        if "topic_interests" in feedback:
            new_topics = feedback["topic_interests"]
            for topic in new_topics:
                if topic not in preferences["topic_preferences"]:
                    preferences["topic_preferences"].append(topic)

        if "difficulty_preference" in feedback:
            preferences["challenge_level"] = feedback["difficulty_preference"]

        if "learning_style_preference" in feedback:
            preferences["learning_style"] = feedback["learning_style_preference"]

        if "personality_preference" in feedback:
            preferences["personality_match"] = feedback["personality_preference"]

        preferences["last_updated"] = datetime.now().isoformat()
        await self.save_preferences()

    async def get_conversation_context(self, user_id: str, conversation_count: int = 5) -> Dict[str, Any]:
        """Get contextual information for generating responses"""
        context = await self.load_context(user_id)
        recent_conversations = await self.get_recent_conversations(user_id, conversation_count)

        # Build conversation context
        conversation_context = {
            "user_profile": {
                "interests": context["interests"],
                "learning_goals": context["learning_goals"],
                "vocabulary_focus": context["vocabulary_focus_areas"],
                "grammar_focus": context["grammar_focus_areas"],
                "conversation_style": context["preferred_conversation_style"]
            },
            "recent_topics": context["current_topics"],
            "conversation_history": [
                {
                    "user_said": conv["user_input"],
                    "system_responded": conv["response"].get("text", ""),
                    "timestamp": conv["timestamp"],
                    "topic": conv["metadata"].get("topic", "general")
                }
                for conv in recent_conversations
            ],
            "preferences": context["preferences"],
            "session_data": context.get("session_data", {})
        }

        return conversation_context

    async def clean_expired_contexts(self) -> None:
        """Clean up expired user contexts"""
        cutoff_date = datetime.now() - timedelta(days=self.context_expire_days)
        expired_users = []

        for user_id, context in self.user_contexts.items():
            try:
                last_accessed = datetime.fromisoformat(context.get("last_accessed", ""))
                if last_accessed < cutoff_date:
                    expired_users.append(user_id)
            except (ValueError, TypeError):
                # Invalid date format, mark for cleanup
                expired_users.append(user_id)

        for user_id in expired_users:
            del self.user_contexts[user_id]
            print(f"Cleaned expired context for user: {user_id}")

        if expired_users:
            await self.save_contexts()

    async def clean_old_history(self) -> None:
        """Clean up old conversation history"""
        cutoff_date = datetime.now() - timedelta(days=self.context_expire_days * 2)  # Keep history longer

        for user_id in list(self.conversation_history.keys()):
            conversations = self.conversation_history[user_id]
            filtered_conversations = []

            for conv in conversations:
                try:
                    conv_date = datetime.fromisoformat(conv["timestamp"])
                    if conv_date >= cutoff_date:
                        filtered_conversations.append(conv)
                except (ValueError, TypeError, KeyError):
                    # Invalid or missing timestamp, keep recent ones
                    filtered_conversations.append(conv)

            if len(filtered_conversations) != len(conversations):
                self.conversation_history[user_id] = filtered_conversations
                print(f"Cleaned old conversations for user: {user_id}")

        await self.save_history()

    async def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get statistics for a specific user"""
        if user_id not in self.user_contexts:
            return {"status": "no_data"}

        context = self.user_contexts[user_id]
        history = self.conversation_history.get(user_id, [])

        # Calculate statistics
        total_conversations = len(history)
        if total_conversations == 0:
            return {"status": "no_conversations"}

        recent_conversations = history[-10:]  # Last 10 conversations
        total_errors = sum(len(conv.get("errors", [])) for conv in recent_conversations)
        avg_errors_per_conversation = total_errors / len(recent_conversations) if recent_conversations else 0

        # Topic distribution
        topics = [conv["metadata"].get("topic", "general") for conv in recent_conversations]
        topic_counts = {}
        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        return {
            "total_conversations": total_conversations,
            "recent_conversations": len(recent_conversations),
            "average_errors": avg_errors_per_conversation,
            "active_interests": context["interests"],
            "current_topics": context["current_topics"],
            "vocabulary_focus_count": len(context["vocabulary_focus_areas"]),
            "grammar_focus_count": len(context["grammar_focus_areas"]),
            "topic_distribution": topic_counts,
            "member_since": context["created_at"],
            "last_active": context["last_accessed"]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall memory agent statistics"""
        total_users = len(self.user_contexts)
        total_conversations = sum(len(history) for history in self.conversation_history.values())

        # Active users (accessed in last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        active_users = 0

        for context in self.user_contexts.values():
            try:
                last_accessed = datetime.fromisoformat(context.get("last_accessed", ""))
                if last_accessed >= week_ago:
                    active_users += 1
            except (ValueError, TypeError):
                pass

        return {
            "total_users": total_users,
            "active_users_week": active_users,
            "total_conversations": total_conversations,
            "average_conversations_per_user": total_conversations / total_users if total_users > 0 else 0,
            "memory_data_files": {
                "contexts": self.context_data_file.exists(),
                "history": self.history_data_file.exists(),
                "preferences": self.preferences_data_file.exists()
            }
        }