"""
Error Agent

Analyzes user input for language errors, tracks error patterns, and provides targeted feedback.
This agent helps users identify and correct common mistakes in their language learning.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


class ErrorAgent:
    """Agent for analyzing language errors and tracking user patterns"""

    def __init__(self):
        self.user_error_patterns = {}
        self.error_rules = {}
        self.error_data_file = Path("data/user_errors.json")
        self.common_errors = {}

    async def initialize(self) -> None:
        """Initialize the error agent"""
        await self.load_error_data()
        await self.initialize_error_rules()
        print("Error Agent initialized")

    async def load_error_data(self) -> None:
        """Load user error patterns from file"""
        try:
            if self.error_data_file.exists():
                with open(self.error_data_file, 'r') as f:
                    self.user_error_patterns = json.load(f)
            else:
                self.error_data_file.parent.mkdir(parents=True, exist_ok=True)
                self.user_error_patterns = {}
        except Exception as e:
            print(f"Error loading user error data: {e}")
            self.user_error_patterns = {}

    async def save_error_data(self) -> None:
        """Save user error patterns to file"""
        try:
            with open(self.error_data_file, 'w') as f:
                json.dump(self.user_error_patterns, f, indent=2)
        except Exception as e:
            print(f"Error saving user error data: {e}")

    async def initialize_error_rules(self) -> None:
        """Initialize error detection rules"""
        self.error_rules = {
            "grammar": {
                "subject_verb_agreement": {
                    "patterns": [
                        (r'\b(I|you|we|they)\s+are\b', "Subject-verb agreement: Use 'am/are' correctly"),
                        (r'\bhe\s+are\b|\bshe\s+are\b|\bit\s+are\b', "Subject-verb agreement: Use 'is' with he/she/it"),
                        (r'\b(I)\s+is\b', "Subject-verb agreement: Use 'I am' not 'I is'"),
                    ],
                    "severity": "high",
                    "category": "basic_grammar"
                },
                "verb_tense": {
                    "patterns": [
                        (r'\byesterday\s+\w+\s+\w*ing\b', "Tense error: Use past tense with 'yesterday'"),
                        (r'\btomorrow\s+\w+ed\b', "Tense error: Use future tense with 'tomorrow'"),
                        (r'\bnow\s+\w+ed\b', "Tense error: Use present tense with 'now'"),
                    ],
                    "severity": "medium",
                    "category": "verb_forms"
                },
                "articles": {
                    "patterns": [
                        (r'\ba\s+[aeiou]\w*\b', "Article error: Use 'an' before vowel sounds"),
                        (r'\ban\s+[^aeiou]\w*\b', "Article error: Use 'a' before consonant sounds"),
                    ],
                    "severity": "low",
                    "category": "articles"
                },
                "prepositions": {
                    "patterns": [
                        (r'\bin\s+monday\b|\bin\s+tuesday\b|\bin\s+wednesday\b|\bin\s+thursday\b|\bin\s+friday\b|\bin\s+saturday\b|\bin\s+sunday\b',
                         "Preposition error: Use 'on' with days of the week"),
                        (r'\bon\s+\d{4}\b|\bon\s+january\b|\bon\s+february\b', "Preposition error: Use 'in' with years and months"),
                    ],
                    "severity": "medium",
                    "category": "prepositions"
                },
                "plurals": {
                    "patterns": [
                        (r'\btwo\s+\w+(?<!s)(?<!es)(?<!ies)\b', "Plural error: Use plural form after numbers > 1"),
                        (r'\bmany\s+\w+(?<!s)(?<!es)(?<!ies)\b', "Plural error: Use plural form after 'many'"),
                    ],
                    "severity": "medium",
                    "category": "number_agreement"
                }
            },
            "vocabulary": {
                "word_choice": {
                    "patterns": [
                        (r'\bmake\s+homework\b', "Word choice: Use 'do homework' not 'make homework'"),
                        (r'\bdo\s+a\s+mistake\b', "Word choice: Use 'make a mistake' not 'do a mistake'"),
                        (r'\bsay\s+me\b', "Word choice: Use 'tell me' not 'say me'"),
                    ],
                    "severity": "medium",
                    "category": "collocation"
                },
                "common_confusions": {
                    "patterns": [
                        (r'\btheir\s+going\b|\btheir\s+here\b', "Confusion: 'their' vs 'they're' - use 'they're' for 'they are'"),
                        (r'\bits\s+raining\b', "Confusion: Use 'it's' (it is) not 'its' (possessive)"),
                        (r'\btoo\s+much\s+books\b|\btoo\s+much\s+people\b', "Confusion: Use 'too many' with countable nouns"),
                    ],
                    "severity": "high",
                    "category": "word_confusion"
                }
            },
            "spelling": {
                "common_misspellings": {
                    "patterns": [
                        (r'\brecieve\b', "Spelling: 'receive' (i before e except after c)"),
                        (r'\boccured\b', "Spelling: 'occurred' (double r)"),
                        (r'\bseperate\b', "Spelling: 'separate' (a not e)"),
                        (r'\bdefinately\b', "Spelling: 'definitely' (finite not finate)"),
                    ],
                    "severity": "low",
                    "category": "spelling"
                }
            },
            "style": {
                "formality": {
                    "patterns": [
                        (r"\bdon't\b|\bcan't\b|\bwon't\b", "Style: Consider using full forms in formal writing"),
                        (r'\bkinda\b|\bsorta\b|\bgonna\b|\bwanna\b', "Style: Use standard forms instead of contractions"),
                    ],
                    "severity": "low",
                    "category": "register"
                }
            }
        }

        # Initialize common error tracking
        self.common_errors = {
            "subject_verb_agreement": {"count": 0, "users": set()},
            "verb_tense": {"count": 0, "users": set()},
            "articles": {"count": 0, "users": set()},
            "prepositions": {"count": 0, "users": set()},
            "plurals": {"count": 0, "users": set()},
            "word_choice": {"count": 0, "users": set()},
            "spelling": {"count": 0, "users": set()},
            "formality": {"count": 0, "users": set()}
        }

    async def analyze(self, user_input: str, user_id: str = "anonymous") -> List[Dict[str, Any]]:
        """Analyze user input for errors"""
        errors = []
        input_lower = user_input.lower()

        # Check against all error rules
        for error_type, error_categories in self.error_rules.items():
            for category, rule_data in error_categories.items():
                patterns = rule_data["patterns"]
                severity = rule_data["severity"]
                error_category = rule_data["category"]

                for pattern, description in patterns:
                    matches = re.finditer(pattern, input_lower, re.IGNORECASE)
                    for match in matches:
                        error = {
                            "type": error_type,
                            "category": error_category,
                            "subcategory": category,
                            "description": description,
                            "severity": severity,
                            "position": match.span(),
                            "matched_text": match.group(),
                            "suggestion": self.get_correction_suggestion(category, match.group()),
                            "confidence": self.calculate_confidence(pattern, match.group()),
                            "timestamp": datetime.now().isoformat()
                        }
                        errors.append(error)

        # Track user-specific error patterns
        if user_id != "anonymous":
            await self.track_user_errors(user_id, errors)

        # Add contextual analysis
        errors = await self.add_contextual_analysis(user_input, errors)

        return errors

    def get_correction_suggestion(self, category: str, matched_text: str) -> str:
        """Get correction suggestions for specific error categories"""
        suggestions = {
            "subject_verb_agreement": {
                "he are": "he is",
                "she are": "she is",
                "it are": "it is",
                "I is": "I am",
                "they is": "they are"
            },
            "articles": {
                "a apple": "an apple",
                "a elephant": "an elephant",
                "an book": "a book",
                "an car": "a car"
            },
            "word_choice": {
                "make homework": "do homework",
                "do a mistake": "make a mistake",
                "say me": "tell me"
            },
            "common_misspellings": {
                "recieve": "receive",
                "occured": "occurred",
                "seperate": "separate",
                "definately": "definitely"
            }
        }

        category_suggestions = suggestions.get(category, {})
        return category_suggestions.get(matched_text.lower(), f"Check '{matched_text}'")

    def calculate_confidence(self, pattern: str, matched_text: str) -> float:
        """Calculate confidence score for error detection"""
        # Simple heuristic based on pattern specificity
        if len(pattern) > 20:  # More specific patterns
            return 0.9
        elif "\\b" in pattern:  # Word boundary patterns
            return 0.8
        else:
            return 0.7

    async def track_user_errors(self, user_id: str, errors: List[Dict[str, Any]]) -> None:
        """Track errors for a specific user"""
        if user_id not in self.user_error_patterns:
            self.user_error_patterns[user_id] = {
                "user_id": user_id,
                "error_history": [],
                "error_counts": {},
                "improvement_areas": [],
                "last_updated": datetime.now().isoformat(),
                "total_errors": 0,
                "recent_trend": "stable"
            }

        user_data = self.user_error_patterns[user_id]

        for error in errors:
            # Add to history
            user_data["error_history"].append({
                "timestamp": error["timestamp"],
                "type": error["type"],
                "category": error["category"],
                "subcategory": error["subcategory"],
                "severity": error["severity"]
            })

            # Update counts
            error_key = f"{error['type']}_{error['category']}"
            user_data["error_counts"][error_key] = user_data["error_counts"].get(error_key, 0) + 1
            user_data["total_errors"] += 1

            # Track in global stats
            if error["subcategory"] in self.common_errors:
                self.common_errors[error["subcategory"]]["count"] += 1
                self.common_errors[error["subcategory"]]["users"].add(user_id)

        # Limit history size
        if len(user_data["error_history"]) > 100:
            user_data["error_history"] = user_data["error_history"][-100:]

        # Update improvement areas
        await self.update_improvement_areas(user_id)

        user_data["last_updated"] = datetime.now().isoformat()
        await self.save_error_data()

    async def update_improvement_areas(self, user_id: str) -> None:
        """Update priority improvement areas for a user"""
        user_data = self.user_error_patterns[user_id]
        error_counts = user_data["error_counts"]

        # Find top error categories
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        top_errors = sorted_errors[:5]  # Top 5 error types

        improvement_areas = []
        for error_key, count in top_errors:
            if count >= 3:  # Only suggest areas with multiple occurrences
                error_type, category = error_key.split("_", 1)
                improvement_areas.append({
                    "category": category,
                    "type": error_type,
                    "frequency": count,
                    "priority": "high" if count >= 10 else "medium" if count >= 5 else "low",
                    "suggestion": self.get_improvement_suggestion(category)
                })

        user_data["improvement_areas"] = improvement_areas

    def get_improvement_suggestion(self, category: str) -> str:
        """Get improvement suggestions for error categories"""
        suggestions = {
            "basic_grammar": "Review basic grammar rules and practice with simple sentences",
            "verb_forms": "Focus on verb tense usage and practice with time expressions",
            "articles": "Study article usage rules and practice with nouns",
            "prepositions": "Learn preposition patterns and practice with time/place expressions",
            "number_agreement": "Practice singular/plural agreement with numbers and quantifiers",
            "collocation": "Study word combinations and common phrases",
            "word_confusion": "Create a list of commonly confused words and practice",
            "spelling": "Use spell check and practice writing commonly misspelled words",
            "register": "Study formal vs informal language and appropriate usage"
        }
        return suggestions.get(category, "Practice this area with focused exercises")

    async def add_contextual_analysis(self, user_input: str, errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add contextual analysis to detected errors"""
        for error in errors:
            # Analyze surrounding context
            position = error["position"]
            start, end = position
            context_start = max(0, start - 20)
            context_end = min(len(user_input), end + 20)
            context = user_input[context_start:context_end]

            error["context"] = context
            error["impact"] = self.assess_error_impact(error, user_input)
            error["learning_tip"] = self.get_learning_tip(error["category"])

        return errors

    def assess_error_impact(self, error: Dict[str, Any], full_text: str) -> str:
        """Assess the impact of an error on communication"""
        severity = error["severity"]
        error_type = error["type"]

        if error_type == "grammar" and severity == "high":
            return "high - may cause confusion"
        elif error_type == "vocabulary" and severity == "medium":
            return "medium - affects clarity"
        elif error_type == "spelling" or severity == "low":
            return "low - minor distraction"
        else:
            return "medium - affects fluency"

    def get_learning_tip(self, category: str) -> str:
        """Get learning tips for specific error categories"""
        tips = {
            "basic_grammar": "Remember: Subject and verb must agree in number (singular/plural)",
            "verb_forms": "Tip: Time words often indicate which tense to use",
            "articles": "Rule: Use 'a' before consonant sounds, 'an' before vowel sounds",
            "prepositions": "Tip: Learn prepositions with common time and place expressions",
            "number_agreement": "Rule: Numbers greater than 1 require plural nouns",
            "collocation": "Tip: Some verbs and nouns always go together - learn these pairs",
            "word_confusion": "Strategy: Create memory devices for commonly confused words",
            "spelling": "Tip: Break long words into smaller parts to spell correctly",
            "register": "Context: Match your language formality to the situation"
        }
        return tips.get(category, "Practice makes perfect!")

    async def get_user_error_patterns(self, user_id: str) -> List[Dict[str, Any]]:
        """Get error patterns for a specific user"""
        if user_id not in self.user_error_patterns:
            return []

        user_data = self.user_error_patterns[user_id]

        # Recent errors (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_errors = []

        for error in user_data["error_history"]:
            try:
                error_date = datetime.fromisoformat(error["timestamp"])
                if error_date >= thirty_days_ago:
                    recent_errors.append(error)
            except (ValueError, KeyError):
                continue

        # Analysis
        error_trends = self.analyze_error_trends(recent_errors)

        return {
            "user_id": user_id,
            "total_errors": user_data["total_errors"],
            "recent_errors_count": len(recent_errors),
            "improvement_areas": user_data["improvement_areas"],
            "error_trends": error_trends,
            "most_common_errors": self.get_most_common_errors(user_data["error_counts"]),
            "progress_indicators": self.calculate_progress_indicators(user_data),
            "last_updated": user_data["last_updated"]
        }

    def analyze_error_trends(self, recent_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in recent errors"""
        if not recent_errors:
            return {"trend": "insufficient_data"}

        # Group by week
        weekly_counts = {}
        for error in recent_errors:
            try:
                error_date = datetime.fromisoformat(error["timestamp"])
                week_key = error_date.strftime("%Y-W%U")
                weekly_counts[week_key] = weekly_counts.get(week_key, 0) + 1
            except (ValueError, KeyError):
                continue

        if len(weekly_counts) < 2:
            return {"trend": "insufficient_data"}

        # Calculate trend
        weeks = sorted(weekly_counts.keys())
        recent_weeks = weeks[-2:]

        if len(recent_weeks) == 2:
            improvement = weekly_counts[recent_weeks[0]] - weekly_counts[recent_weeks[1]]
            if improvement > 0:
                trend = "improving"
            elif improvement < 0:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "weekly_counts": weekly_counts,
            "recent_average": sum(weekly_counts.values()) / len(weekly_counts)
        }

    def get_most_common_errors(self, error_counts: Dict[str, int]) -> List[Dict[str, Any]]:
        """Get most common error types for a user"""
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)

        common_errors = []
        for error_key, count in sorted_errors[:5]:
            error_type, category = error_key.split("_", 1)
            common_errors.append({
                "type": error_type,
                "category": category,
                "count": count,
                "suggestion": self.get_improvement_suggestion(category)
            })

        return common_errors

    def calculate_progress_indicators(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate progress indicators for a user"""
        total_errors = user_data["total_errors"]
        error_history = user_data["error_history"]

        if total_errors == 0:
            return {"status": "no_errors_yet"}

        # Calculate recent improvement
        recent_30_days = []
        previous_30_days = []
        now = datetime.now()

        for error in error_history:
            try:
                error_date = datetime.fromisoformat(error["timestamp"])
                days_ago = (now - error_date).days

                if days_ago <= 30:
                    recent_30_days.append(error)
                elif 30 < days_ago <= 60:
                    previous_30_days.append(error)
            except (ValueError, KeyError):
                continue

        recent_count = len(recent_30_days)
        previous_count = len(previous_30_days)

        if previous_count > 0:
            improvement_rate = (previous_count - recent_count) / previous_count
        else:
            improvement_rate = 0

        return {
            "recent_error_count": recent_count,
            "previous_period_count": previous_count,
            "improvement_rate": improvement_rate,
            "improvement_status": "improving" if improvement_rate > 0.1 else "stable" if improvement_rate > -0.1 else "needs_attention"
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get error agent statistics"""
        total_users = len(self.user_error_patterns)
        total_errors = sum(user_data["total_errors"] for user_data in self.user_error_patterns.values())

        # Most common error types across all users
        global_error_counts = {}
        for user_data in self.user_error_patterns.values():
            for error_key, count in user_data["error_counts"].items():
                global_error_counts[error_key] = global_error_counts.get(error_key, 0) + count

        most_common_global = sorted(global_error_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_users": total_users,
            "total_errors_tracked": total_errors,
            "error_rules_count": sum(len(categories) for categories in self.error_rules.values()),
            "most_common_errors": [{"type": error[0], "count": error[1]} for error in most_common_global],
            "average_errors_per_user": total_errors / total_users if total_users > 0 else 0,
            "error_categories": list(self.error_rules.keys())
        }