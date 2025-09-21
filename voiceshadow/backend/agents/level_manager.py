"""
Level Manager Agent

Manages user language proficiency levels and adapts conversation difficulty accordingly.
This agent tracks user progress and determines appropriate conversation complexity.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path


class LevelManagerAgent:
    """Agent for managing user language proficiency levels"""

    def __init__(self):
        self.user_levels = {}
        self.level_criteria = {
            "beginner": {
                "min_score": 0.0,
                "max_score": 0.3,
                "vocabulary_size": 500,
                "grammar_complexity": "basic",
                "description": "Basic vocabulary and simple sentence structures"
            },
            "elementary": {
                "min_score": 0.3,
                "max_score": 0.5,
                "vocabulary_size": 1000,
                "grammar_complexity": "simple",
                "description": "Elementary vocabulary with present/past tense"
            },
            "intermediate": {
                "min_score": 0.5,
                "max_score": 0.7,
                "vocabulary_size": 2000,
                "grammar_complexity": "moderate",
                "description": "Moderate vocabulary with complex tenses"
            },
            "upper_intermediate": {
                "min_score": 0.7,
                "max_score": 0.85,
                "vocabulary_size": 3500,
                "grammar_complexity": "advanced",
                "description": "Advanced vocabulary with subjunctive and conditionals"
            },
            "advanced": {
                "min_score": 0.85,
                "max_score": 1.0,
                "vocabulary_size": 5000,
                "grammar_complexity": "native",
                "description": "Near-native proficiency with idiomatic expressions"
            }
        }
        self.user_data_file = Path("data/user_levels.json")

    async def initialize(self) -> None:
        """Initialize the level manager agent"""
        await self.load_user_data()
        print("Level Manager Agent initialized")

    async def load_user_data(self) -> None:
        """Load user level data from file"""
        try:
            if self.user_data_file.exists():
                with open(self.user_data_file, 'r') as f:
                    self.user_levels = json.load(f)
            else:
                # Create directory if it doesn't exist
                self.user_data_file.parent.mkdir(parents=True, exist_ok=True)
                self.user_levels = {}
        except Exception as e:
            print(f"Error loading user level data: {e}")
            self.user_levels = {}

    async def save_user_data(self) -> None:
        """Save user level data to file"""
        try:
            with open(self.user_data_file, 'w') as f:
                json.dump(self.user_levels, f, indent=2)
        except Exception as e:
            print(f"Error saving user level data: {e}")

    async def get_level(self, user_id: str) -> str:
        """Get the current proficiency level for a user"""
        user_data = self.user_levels.get(user_id, {})

        if not user_data:
            # New user - start with assessment
            await self.initialize_user(user_id)
            return "intermediate"  # Default starting level

        return user_data.get("current_level", "intermediate")

    async def initialize_user(self, user_id: str) -> None:
        """Initialize a new user with default level data"""
        self.user_levels[user_id] = {
            "current_level": "intermediate",
            "level_score": 0.6,
            "conversation_count": 0,
            "correct_responses": 0,
            "errors_count": 0,
            "vocabulary_used": set(),
            "grammar_patterns": [],
            "last_assessment": datetime.now().isoformat(),
            "progress_history": [
                {
                    "date": datetime.now().isoformat(),
                    "level": "intermediate",
                    "score": 0.6,
                    "reason": "initial_assessment"
                }
            ]
        }
        await self.save_user_data()

    async def update_level(self, user_id: str, feedback: Dict[str, Any]) -> None:
        """Update user level based on performance feedback"""
        if user_id not in self.user_levels:
            await self.initialize_user(user_id)

        user_data = self.user_levels[user_id]

        # Update conversation statistics
        user_data["conversation_count"] += 1

        # Process feedback
        errors = feedback.get("errors", [])
        response_quality = feedback.get("response_quality", 0.5)
        grammar_accuracy = feedback.get("grammar_accuracy", 0.5)
        vocabulary_complexity = feedback.get("vocabulary_complexity", 0.5)

        # Calculate performance metrics
        error_rate = len(errors) / max(1, user_data["conversation_count"])
        user_data["errors_count"] += len(errors)

        if response_quality > 0.7:
            user_data["correct_responses"] += 1

        # Update vocabulary tracking
        new_vocabulary = feedback.get("vocabulary_used", [])
        if "vocabulary_used" not in user_data:
            user_data["vocabulary_used"] = []
        user_data["vocabulary_used"].extend(new_vocabulary)
        user_data["vocabulary_used"] = list(set(user_data["vocabulary_used"]))

        # Calculate new level score
        accuracy_score = user_data["correct_responses"] / max(1, user_data["conversation_count"])
        vocabulary_score = min(1.0, len(user_data["vocabulary_used"]) / 2000)
        grammar_score = grammar_accuracy

        new_score = (accuracy_score * 0.4 + vocabulary_score * 0.3 + grammar_score * 0.3)
        user_data["level_score"] = new_score

        # Determine new level
        new_level = self.score_to_level(new_score)
        old_level = user_data["current_level"]

        if new_level != old_level:
            user_data["current_level"] = new_level
            user_data["progress_history"].append({
                "date": datetime.now().isoformat(),
                "level": new_level,
                "score": new_score,
                "reason": "performance_update",
                "from_level": old_level
            })
            print(f"User {user_id} level updated: {old_level} -> {new_level}")

        user_data["last_assessment"] = datetime.now().isoformat()
        await self.save_user_data()

    def score_to_level(self, score: float) -> str:
        """Convert a numeric score to a proficiency level"""
        for level, criteria in self.level_criteria.items():
            if criteria["min_score"] <= score <= criteria["max_score"]:
                return level
        return "intermediate"  # Default fallback

    async def get_level_requirements(self, level: str) -> Dict[str, Any]:
        """Get the requirements and characteristics for a specific level"""
        return self.level_criteria.get(level, self.level_criteria["intermediate"])

    async def get_conversation_parameters(self, user_id: str) -> Dict[str, Any]:
        """Get conversation parameters based on user's current level"""
        level = await self.get_level(user_id)
        level_info = await self.get_level_requirements(level)

        return {
            "level": level,
            "vocabulary_complexity": level_info["grammar_complexity"],
            "max_vocabulary_size": level_info["vocabulary_size"],
            "sentence_complexity": level_info["grammar_complexity"],
            "topic_difficulty": self.get_topic_difficulty(level),
            "error_tolerance": self.get_error_tolerance(level)
        }

    def get_topic_difficulty(self, level: str) -> str:
        """Get appropriate topic difficulty for a level"""
        difficulty_map = {
            "beginner": "basic_daily_life",
            "elementary": "personal_interests",
            "intermediate": "current_events_opinions",
            "upper_intermediate": "abstract_concepts",
            "advanced": "complex_discussions"
        }
        return difficulty_map.get(level, "current_events_opinions")

    def get_error_tolerance(self, level: str) -> float:
        """Get error tolerance threshold for a level"""
        tolerance_map = {
            "beginner": 0.7,      # High tolerance
            "elementary": 0.6,
            "intermediate": 0.4,
            "upper_intermediate": 0.2,
            "advanced": 0.1       # Low tolerance
        }
        return tolerance_map.get(level, 0.4)

    async def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Get detailed progress information for a user"""
        if user_id not in self.user_levels:
            return {"status": "no_data"}

        user_data = self.user_levels[user_id]
        current_level = user_data["current_level"]
        level_info = await self.get_level_requirements(current_level)

        # Calculate progress within current level
        score = user_data["level_score"]
        level_min = level_info["min_score"]
        level_max = level_info["max_score"]
        level_progress = (score - level_min) / (level_max - level_min) if level_max > level_min else 1.0

        return {
            "current_level": current_level,
            "level_score": score,
            "progress_in_level": min(1.0, max(0.0, level_progress)),
            "conversation_count": user_data["conversation_count"],
            "accuracy_rate": user_data["correct_responses"] / max(1, user_data["conversation_count"]),
            "vocabulary_size": len(user_data.get("vocabulary_used", [])),
            "target_vocabulary": level_info["vocabulary_size"],
            "recent_history": user_data.get("progress_history", [])[-5:],  # Last 5 changes
            "next_level": self.get_next_level(current_level),
            "level_description": level_info["description"]
        }

    def get_next_level(self, current_level: str) -> Optional[str]:
        """Get the next level after the current one"""
        levels = ["beginner", "elementary", "intermediate", "upper_intermediate", "advanced"]
        try:
            current_index = levels.index(current_level)
            if current_index < len(levels) - 1:
                return levels[current_index + 1]
        except ValueError:
            pass
        return None

    async def suggest_level_adjustment(self, user_id: str) -> Dict[str, Any]:
        """Suggest level adjustment based on recent performance"""
        if user_id not in self.user_levels:
            return {"suggestion": "no_data"}

        user_data = self.user_levels[user_id]
        recent_conversations = min(10, user_data["conversation_count"])

        if recent_conversations < 5:
            return {"suggestion": "need_more_data", "message": "Need more conversations to assess level"}

        accuracy = user_data["correct_responses"] / user_data["conversation_count"]
        current_level = user_data["current_level"]

        # Suggest adjustment based on performance
        if accuracy > 0.9:
            next_level = self.get_next_level(current_level)
            if next_level:
                return {
                    "suggestion": "level_up",
                    "recommended_level": next_level,
                    "reason": f"High accuracy rate: {accuracy:.2f}",
                    "confidence": 0.8
                }
        elif accuracy < 0.3:
            levels = ["beginner", "elementary", "intermediate", "upper_intermediate", "advanced"]
            try:
                current_index = levels.index(current_level)
                if current_index > 0:
                    return {
                        "suggestion": "level_down",
                        "recommended_level": levels[current_index - 1],
                        "reason": f"Low accuracy rate: {accuracy:.2f}",
                        "confidence": 0.7
                    }
            except ValueError:
                pass

        return {
            "suggestion": "maintain",
            "current_level": current_level,
            "reason": f"Performance is appropriate for level: {accuracy:.2f}",
            "confidence": 0.6
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get overall statistics for the level manager"""
        if not self.user_levels:
            return {"total_users": 0}

        level_distribution = {}
        total_conversations = 0
        total_accuracy = 0

        for user_data in self.user_levels.values():
            level = user_data.get("current_level", "unknown")
            level_distribution[level] = level_distribution.get(level, 0) + 1
            total_conversations += user_data.get("conversation_count", 0)
            if user_data.get("conversation_count", 0) > 0:
                total_accuracy += user_data.get("correct_responses", 0) / user_data["conversation_count"]

        return {
            "total_users": len(self.user_levels),
            "level_distribution": level_distribution,
            "total_conversations": total_conversations,
            "average_accuracy": total_accuracy / len(self.user_levels) if self.user_levels else 0,
            "available_levels": list(self.level_criteria.keys())
        }