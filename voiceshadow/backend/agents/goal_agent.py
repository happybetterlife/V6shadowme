"""
Goal Agent

Manages user learning goals, tracks progress, and provides personalized recommendations.
This agent helps users set and achieve their language learning objectives.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path


class GoalAgent:
    """Agent for managing user learning goals and progress tracking"""

    def __init__(self):
        self.user_goals = {}
        self.goal_templates = {}
        self.goals_data_file = Path("data/user_goals.json")
        self.achievement_thresholds = {
            "beginner": {"conversations": 10, "accuracy": 0.6, "vocabulary": 50},
            "elementary": {"conversations": 20, "accuracy": 0.7, "vocabulary": 100},
            "intermediate": {"conversations": 30, "accuracy": 0.75, "vocabulary": 200},
            "upper_intermediate": {"conversations": 50, "accuracy": 0.8, "vocabulary": 350},
            "advanced": {"conversations": 100, "accuracy": 0.85, "vocabulary": 500}
        }

    async def initialize(self) -> None:
        """Initialize the goal agent"""
        await self.load_goal_data()
        await self.initialize_goal_templates()
        print("Goal Agent initialized")

    async def load_goal_data(self) -> None:
        """Load user goals from file"""
        try:
            if self.goals_data_file.exists():
                with open(self.goals_data_file, 'r') as f:
                    self.user_goals = json.load(f)
            else:
                self.goals_data_file.parent.mkdir(parents=True, exist_ok=True)
                self.user_goals = {}
        except Exception as e:
            print(f"Error loading user goals: {e}")
            self.user_goals = {}

    async def save_goal_data(self) -> None:
        """Save user goals to file"""
        try:
            with open(self.goals_data_file, 'w') as f:
                json.dump(self.user_goals, f, indent=2)
        except Exception as e:
            print(f"Error saving user goals: {e}")

    async def initialize_goal_templates(self) -> None:
        """Initialize predefined goal templates"""
        self.goal_templates = {
            "daily_conversation": {
                "name": "Daily Conversation Practice",
                "description": "Have meaningful conversations every day",
                "type": "recurring",
                "target_metric": "conversations_per_day",
                "target_value": 1,
                "timeframe_days": 30,
                "difficulty": "beginner",
                "milestones": [7, 14, 21, 30]
            },
            "vocabulary_expansion": {
                "name": "Vocabulary Expansion",
                "description": "Learn new vocabulary words through conversation",
                "type": "cumulative",
                "target_metric": "new_vocabulary_words",
                "target_value": 100,
                "timeframe_days": 60,
                "difficulty": "intermediate",
                "milestones": [25, 50, 75, 100]
            },
            "accuracy_improvement": {
                "name": "Improve Speaking Accuracy",
                "description": "Reduce errors and improve grammar accuracy",
                "type": "percentage",
                "target_metric": "accuracy_rate",
                "target_value": 0.85,
                "timeframe_days": 45,
                "difficulty": "intermediate",
                "milestones": [0.7, 0.75, 0.8, 0.85]
            },
            "fluency_building": {
                "name": "Build Conversational Fluency",
                "description": "Engage in longer, more complex conversations",
                "type": "progression",
                "target_metric": "conversation_length",
                "target_value": 10,  # 10 exchanges per conversation
                "timeframe_days": 30,
                "difficulty": "advanced",
                "milestones": [3, 5, 7, 10]
            },
            "topic_mastery": {
                "name": "Topic Mastery",
                "description": "Master conversation in specific topic areas",
                "type": "categorical",
                "target_metric": "topics_mastered",
                "target_value": 5,
                "timeframe_days": 90,
                "difficulty": "upper_intermediate",
                "milestones": [1, 2, 3, 5]
            },
            "consistency_challenge": {
                "name": "30-Day Consistency Challenge",
                "description": "Practice conversation for 30 consecutive days",
                "type": "streak",
                "target_metric": "consecutive_days",
                "target_value": 30,
                "timeframe_days": 30,
                "difficulty": "all_levels",
                "milestones": [7, 14, 21, 30]
            }
        }

    async def initialize_user_goals(self, user_id: str) -> None:
        """Initialize goals for a new user"""
        if user_id not in self.user_goals:
            self.user_goals[user_id] = {
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "active_goals": [],
                "completed_goals": [],
                "goal_history": [],
                "achievements": [],
                "current_streak": 0,
                "longest_streak": 0,
                "last_activity": datetime.now().isoformat(),
                "progress_stats": {
                    "total_conversations": 0,
                    "total_vocabulary_learned": 0,
                    "average_accuracy": 0.0,
                    "topics_practiced": [],
                    "milestones_reached": 0
                }
            }
            await self.save_goal_data()

    async def update_progress(self, user_id: str, user_input: str, response: Dict[str, Any]) -> None:
        """Update user progress based on conversation"""
        if user_id not in self.user_goals:
            await self.initialize_user_goals(user_id)

        user_data = self.user_goals[user_id]

        # Update basic stats
        user_data["progress_stats"]["total_conversations"] += 1
        user_data["last_activity"] = datetime.now().isoformat()

        # Update vocabulary if provided in response
        new_vocabulary = response.get("vocabulary_used", [])
        if new_vocabulary:
            current_vocab = set(user_data["progress_stats"].get("vocabulary_learned", []))
            current_vocab.update(new_vocabulary)
            user_data["progress_stats"]["vocabulary_learned"] = list(current_vocab)
            user_data["progress_stats"]["total_vocabulary_learned"] = len(current_vocab)

        # Update accuracy
        errors = response.get("errors", [])
        conversation_accuracy = max(0.0, 1.0 - (len(errors) / 10))  # Assume max 10 errors

        current_accuracy = user_data["progress_stats"]["average_accuracy"]
        total_conversations = user_data["progress_stats"]["total_conversations"]

        # Running average of accuracy
        new_accuracy = ((current_accuracy * (total_conversations - 1)) + conversation_accuracy) / total_conversations
        user_data["progress_stats"]["average_accuracy"] = new_accuracy

        # Update topics practiced
        topic = response.get("topic", "general")
        if topic not in user_data["progress_stats"]["topics_practiced"]:
            user_data["progress_stats"]["topics_practiced"].append(topic)

        # Update streak
        await self.update_streak(user_id)

        # Check progress on active goals
        for goal in user_data["active_goals"]:
            await self.check_goal_progress(user_id, goal, user_input, response)

        await self.save_goal_data()

    async def update_streak(self, user_id: str) -> None:
        """Update user's conversation streak"""
        user_data = self.user_goals[user_id]
        last_activity = datetime.fromisoformat(user_data["last_activity"])
        today = datetime.now().date()
        last_activity_date = last_activity.date()

        if last_activity_date == today:
            # Same day, streak continues
            pass
        elif last_activity_date == today - timedelta(days=1):
            # Yesterday, increment streak
            user_data["current_streak"] += 1
            if user_data["current_streak"] > user_data["longest_streak"]:
                user_data["longest_streak"] = user_data["current_streak"]
        else:
            # Streak broken
            user_data["current_streak"] = 1

    async def check_goal_progress(self, user_id: str, goal: Dict[str, Any], user_input: str, response: Dict[str, Any]) -> None:
        """Check and update progress on a specific goal"""
        goal_type = goal.get("type", "cumulative")
        target_metric = goal.get("target_metric", "")
        current_progress = goal.get("current_progress", 0)

        # Update progress based on goal type
        if target_metric == "conversations_per_day":
            # Count today's conversations
            today = datetime.now().date()
            goal_start = datetime.fromisoformat(goal["start_date"]).date()
            days_elapsed = (today - goal_start).days + 1

            conversations_today = 1  # This conversation
            goal["current_progress"] = conversations_today
            goal["total_progress"] = goal.get("total_progress", 0) + 1

        elif target_metric == "new_vocabulary_words":
            new_vocab = response.get("vocabulary_used", [])
            if new_vocab:
                goal_vocab = set(goal.get("vocabulary_learned", []))
                new_words = set(new_vocab) - goal_vocab
                if new_words:
                    goal["vocabulary_learned"] = list(goal_vocab.union(new_words))
                    goal["current_progress"] = len(goal["vocabulary_learned"])

        elif target_metric == "accuracy_rate":
            errors = response.get("errors", [])
            conversation_accuracy = max(0.0, 1.0 - (len(errors) / 10))

            # Running average for this goal
            goal_conversations = goal.get("conversations_count", 0) + 1
            current_avg = goal.get("current_progress", 0.0)
            new_avg = ((current_avg * (goal_conversations - 1)) + conversation_accuracy) / goal_conversations

            goal["current_progress"] = new_avg
            goal["conversations_count"] = goal_conversations

        elif target_metric == "conversation_length":
            # Estimate conversation length from response
            response_length = len(response.get("text", "").split())
            input_length = len(user_input.split())
            total_length = response_length + input_length

            # Use exchange count as a proxy (rough estimate)
            estimated_exchanges = max(1, total_length // 20)

            if estimated_exchanges > goal.get("current_progress", 0):
                goal["current_progress"] = estimated_exchanges

        elif target_metric == "topics_mastered":
            topic = response.get("topic", "general")
            mastered_topics = goal.get("mastered_topics", [])

            # Consider a topic mastered after 5 successful conversations
            topic_count = goal.get("topic_counts", {})
            topic_count[topic] = topic_count.get(topic, 0) + 1
            goal["topic_counts"] = topic_count

            if topic_count[topic] >= 5 and topic not in mastered_topics:
                mastered_topics.append(topic)
                goal["mastered_topics"] = mastered_topics
                goal["current_progress"] = len(mastered_topics)

        elif target_metric == "consecutive_days":
            user_data = self.user_goals[user_id]
            goal["current_progress"] = user_data["current_streak"]

        # Check for milestone achievements
        await self.check_milestones(user_id, goal)

        # Check if goal is completed
        if goal["current_progress"] >= goal["target_value"]:
            await self.complete_goal(user_id, goal)

    async def check_milestones(self, user_id: str, goal: Dict[str, Any]) -> None:
        """Check if user has reached any milestones"""
        milestones = goal.get("milestones", [])
        current_progress = goal.get("current_progress", 0)
        reached_milestones = goal.get("reached_milestones", [])

        for milestone in milestones:
            if current_progress >= milestone and milestone not in reached_milestones:
                reached_milestones.append(milestone)
                goal["reached_milestones"] = reached_milestones

                # Add achievement
                await self.add_achievement(user_id, {
                    "type": "milestone",
                    "goal_name": goal["name"],
                    "milestone": milestone,
                    "date": datetime.now().isoformat(),
                    "description": f"Reached {milestone} in {goal['name']}"
                })

                self.user_goals[user_id]["progress_stats"]["milestones_reached"] += 1

    async def complete_goal(self, user_id: str, goal: Dict[str, Any]) -> None:
        """Mark a goal as completed"""
        user_data = self.user_goals[user_id]

        goal["completed_date"] = datetime.now().isoformat()
        goal["status"] = "completed"

        # Move from active to completed
        user_data["active_goals"] = [g for g in user_data["active_goals"] if g["id"] != goal["id"]]
        user_data["completed_goals"].append(goal)

        # Add completion achievement
        await self.add_achievement(user_id, {
            "type": "goal_completion",
            "goal_name": goal["name"],
            "date": datetime.now().isoformat(),
            "description": f"Completed goal: {goal['name']}",
            "difficulty": goal.get("difficulty", "intermediate")
        })

        print(f"User {user_id} completed goal: {goal['name']}")

    async def add_achievement(self, user_id: str, achievement: Dict[str, Any]) -> None:
        """Add an achievement for the user"""
        if user_id not in self.user_goals:
            await self.initialize_user_goals(user_id)

        self.user_goals[user_id]["achievements"].append(achievement)

    async def update_goals(self, user_id: str, feedback: Dict[str, Any]) -> None:
        """Update user goals based on feedback"""
        if user_id not in self.user_goals:
            await self.initialize_user_goals(user_id)

        # Process goal-related feedback
        new_goals = feedback.get("new_goals", [])
        for goal_template_id in new_goals:
            await self.add_goal_from_template(user_id, goal_template_id)

        # Update existing goal preferences
        goal_updates = feedback.get("goal_updates", {})
        for goal_id, updates in goal_updates.items():
            await self.update_goal(user_id, goal_id, updates)

        await self.save_goal_data()

    async def add_goal_from_template(self, user_id: str, template_id: str, custom_params: Optional[Dict[str, Any]] = None) -> bool:
        """Add a new goal from a template"""
        if template_id not in self.goal_templates:
            return False

        template = self.goal_templates[template_id].copy()

        # Apply custom parameters if provided
        if custom_params:
            template.update(custom_params)

        # Create goal instance
        goal = {
            "id": f"{user_id}_{template_id}_{len(self.user_goals[user_id]['active_goals'])}",
            "template_id": template_id,
            "name": template["name"],
            "description": template["description"],
            "type": template["type"],
            "target_metric": template["target_metric"],
            "target_value": template["target_value"],
            "current_progress": 0,
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() + timedelta(days=template["timeframe_days"])).isoformat(),
            "timeframe_days": template["timeframe_days"],
            "difficulty": template["difficulty"],
            "milestones": template["milestones"],
            "reached_milestones": [],
            "status": "active",
            "created_at": datetime.now().isoformat()
        }

        self.user_goals[user_id]["active_goals"].append(goal)
        self.user_goals[user_id]["goal_history"].append({
            "action": "goal_added",
            "goal_id": goal["id"],
            "goal_name": goal["name"],
            "date": datetime.now().isoformat()
        })

        await self.save_goal_data()
        return True

    async def update_goal(self, user_id: str, goal_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing goal"""
        user_data = self.user_goals.get(user_id, {})
        active_goals = user_data.get("active_goals", [])

        for goal in active_goals:
            if goal["id"] == goal_id:
                goal.update(updates)
                goal["last_updated"] = datetime.now().isoformat()
                await self.save_goal_data()
                return True

        return False

    async def get_user_goals(self, user_id: str) -> Dict[str, Any]:
        """Get all goals for a user"""
        if user_id not in self.user_goals:
            await self.initialize_user_goals(user_id)

        user_data = self.user_goals[user_id]

        # Calculate progress percentages
        for goal in user_data["active_goals"]:
            if goal["target_value"] > 0:
                goal["progress_percentage"] = min(100, (goal["current_progress"] / goal["target_value"]) * 100)
            else:
                goal["progress_percentage"] = 0

        return {
            "active_goals": user_data["active_goals"],
            "completed_goals": user_data["completed_goals"][-10:],  # Last 10 completed
            "achievements": user_data["achievements"][-20:],  # Last 20 achievements
            "progress_stats": user_data["progress_stats"],
            "streaks": {
                "current": user_data["current_streak"],
                "longest": user_data["longest_streak"]
            }
        }

    async def suggest_goals(self, user_id: str, user_level: str, user_interests: List[str]) -> List[Dict[str, Any]]:
        """Suggest appropriate goals for a user"""
        suggestions = []

        # Filter templates by difficulty
        suitable_templates = {}
        for template_id, template in self.goal_templates.items():
            template_difficulty = template["difficulty"]
            if template_difficulty == "all_levels" or template_difficulty == user_level:
                suitable_templates[template_id] = template

        # Get user's current goals to avoid duplicates
        current_goal_templates = set()
        if user_id in self.user_goals:
            for goal in self.user_goals[user_id]["active_goals"]:
                current_goal_templates.add(goal.get("template_id"))

        # Create suggestions
        for template_id, template in suitable_templates.items():
            if template_id not in current_goal_templates:
                suggestion = {
                    "template_id": template_id,
                    "name": template["name"],
                    "description": template["description"],
                    "difficulty": template["difficulty"],
                    "timeframe_days": template["timeframe_days"],
                    "target_value": template["target_value"],
                    "estimated_effort": self.estimate_effort(template),
                    "benefits": self.get_goal_benefits(template),
                    "recommended": self.is_goal_recommended(user_level, template, user_interests)
                }
                suggestions.append(suggestion)

        # Sort by recommendation score
        suggestions.sort(key=lambda x: x["recommended"], reverse=True)
        return suggestions[:5]  # Top 5 suggestions

    def estimate_effort(self, template: Dict[str, Any]) -> str:
        """Estimate effort required for a goal"""
        timeframe = template["timeframe_days"]
        target_value = template["target_value"]

        if timeframe <= 7:
            return "low"
        elif timeframe <= 30:
            return "medium"
        else:
            return "high"

    def get_goal_benefits(self, template: Dict[str, Any]) -> List[str]:
        """Get benefits of achieving a goal"""
        benefits_map = {
            "daily_conversation": ["Improved fluency", "Better confidence", "Consistent practice"],
            "vocabulary_expansion": ["Richer expression", "Better comprehension", "Academic improvement"],
            "accuracy_improvement": ["Clearer communication", "Professional skills", "Grammar mastery"],
            "fluency_building": ["Natural conversation", "Reduced hesitation", "Cultural understanding"],
            "topic_mastery": ["Specialized knowledge", "Professional skills", "Cultural awareness"],
            "consistency_challenge": ["Habit formation", "Discipline building", "Long-term improvement"]
        }

        template_id = template.get("name", "").lower().replace(" ", "_")
        return benefits_map.get(template_id, ["Language improvement", "Personal growth"])

    def is_goal_recommended(self, user_level: str, template: Dict[str, Any], user_interests: List[str]) -> float:
        """Calculate recommendation score for a goal"""
        score = 0.5  # Base score

        # Level appropriateness
        template_difficulty = template["difficulty"]
        if template_difficulty == user_level:
            score += 0.3
        elif template_difficulty == "all_levels":
            score += 0.2

        # Interest alignment
        template_name = template["name"].lower()
        for interest in user_interests:
            if interest.lower() in template_name:
                score += 0.1

        # Goal type preferences
        if template["type"] in ["streak", "recurring"]:
            score += 0.1  # Habit-forming goals are generally good

        return min(1.0, score)

    def get_stats(self) -> Dict[str, Any]:
        """Get goal agent statistics"""
        total_users = len(self.user_goals)
        total_active_goals = sum(len(user_data["active_goals"]) for user_data in self.user_goals.values())
        total_completed_goals = sum(len(user_data["completed_goals"]) for user_data in self.user_goals.values())
        total_achievements = sum(len(user_data["achievements"]) for user_data in self.user_goals.values())

        # Goal template distribution
        template_usage = {}
        for user_data in self.user_goals.values():
            for goal in user_data["active_goals"] + user_data["completed_goals"]:
                template_id = goal.get("template_id", "unknown")
                template_usage[template_id] = template_usage.get(template_id, 0) + 1

        return {
            "total_users": total_users,
            "total_active_goals": total_active_goals,
            "total_completed_goals": total_completed_goals,
            "total_achievements": total_achievements,
            "available_templates": len(self.goal_templates),
            "template_usage": template_usage,
            "completion_rate": total_completed_goals / max(1, total_active_goals + total_completed_goals)
        }