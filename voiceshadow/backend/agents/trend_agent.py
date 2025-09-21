"""
Trend Agent

Analyzes trending topics and social media trends to suggest relevant conversation topics.
This agent integrates with the trends database to provide current and engaging topics.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import the trends database using robust resolver
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_resolver import TrendsDatabase


class TrendAgent:
    """Agent for analyzing and suggesting trending topics"""

    def __init__(self):
        self.trends_db = TrendsDatabase()
        self.trend_cache = {}
        self.cache_expiry = {}
        self.cache_duration_minutes = 30
        self.relevance_threshold = 0.3

    async def initialize(self) -> None:
        """Initialize the trend agent"""
        await self.trends_db.initialize()
        print("Trend Agent initialized")

    async def get_relevant_trends(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get trends relevant to user input and context"""

        # Extract user profile from context
        user_profile = context.get("user_profile", {})
        interests = user_profile.get("interests", [])
        recent_topics = context.get("recent_topics", [])

        # Check cache first
        cache_key = f"{user_input}_{hash(str(sorted(interests)))}"
        if await self.is_cache_valid(cache_key):
            return self.trend_cache[cache_key]

        # Analyze user input for topics
        input_keywords = await self.extract_keywords_from_input(user_input)

        # Search for relevant trends
        relevant_trends = []

        # Search by input keywords
        if input_keywords:
            for keyword in input_keywords:
                matching_trends = await self.trends_db.search_trends(keyword)
                relevant_trends.extend(matching_trends)

        # Get trends based on user interests
        if interests:
            for interest in interests:
                interest_trends = await self.trends_db.search_trends(interest)
                relevant_trends.extend(interest_trends)

        # Get today's trending topic as fallback
        if not relevant_trends:
            today_topic = await self.trends_db.get_todays_topic(user_profile)
            if today_topic:
                relevant_trends = [today_topic]

        # Remove duplicates and score relevance
        unique_trends = {}
        for trend in relevant_trends:
            trend_id = trend.get("id", trend.get("name", "unknown"))
            if trend_id not in unique_trends:
                unique_trends[trend_id] = trend

        # Score and rank trends
        scored_trends = []
        for trend in unique_trends.values():
            relevance_score = await self.calculate_relevance_score(
                trend, user_input, interests, recent_topics
            )
            if relevance_score >= self.relevance_threshold:
                scored_trends.append((relevance_score, trend))

        # Sort by relevance and select best trend
        scored_trends.sort(key=lambda x: x[0], reverse=True)

        if scored_trends:
            best_score, best_trend = scored_trends[0]
            result = {
                "topic": best_trend.get("name", "general conversation"),
                "category": best_trend.get("category", "general"),
                "keywords": best_trend.get("keywords", []),
                "confidence": best_score,
                "description": best_trend.get("description", ""),
                "conversation_starters": best_trend.get("conversation_starters", []),
                "trend_data": best_trend,
                "alternative_topics": [
                    {
                        "topic": trend[1].get("name", ""),
                        "score": trend[0],
                        "category": trend[1].get("category", "general")
                    }
                    for trend in scored_trends[1:4]  # Top 3 alternatives
                ]
            }
        else:
            # Default fallback
            result = await self.get_fallback_trend(user_input, context)

        # Cache the result
        self.trend_cache[cache_key] = result
        self.cache_expiry[cache_key] = datetime.now() + timedelta(minutes=self.cache_duration_minutes)

        return result

    async def calculate_relevance_score(
        self,
        trend: Dict[str, Any],
        user_input: str,
        interests: List[str],
        recent_topics: List[str]
    ) -> float:
        """Calculate relevance score for a trend"""
        score = 0.0

        # Base popularity score
        popularity = trend.get("popularity_score", 0.5)
        score += popularity * 0.3

        # Input keyword matching
        input_lower = user_input.lower()
        trend_keywords = [kw.lower() for kw in trend.get("keywords", [])]
        trend_name = trend.get("name", "").lower()

        # Direct keyword matches
        keyword_matches = sum(1 for kw in trend_keywords if kw in input_lower)
        if keyword_matches > 0:
            score += min(0.4, keyword_matches * 0.1)

        # Name matching
        if any(word in trend_name for word in input_lower.split()):
            score += 0.2

        # Interest alignment
        interest_matches = 0
        for interest in interests:
            interest_lower = interest.lower()
            if interest_lower in trend_name or interest_lower == trend.get("category", "").lower():
                interest_matches += 1
            if any(interest_lower in kw for kw in trend_keywords):
                interest_matches += 0.5

        if interest_matches > 0:
            score += min(0.3, interest_matches * 0.1)

        # Recency bonus (avoid recently discussed topics)
        trend_topic = trend.get("name", "").lower()
        recent_penalty = sum(0.1 for topic in recent_topics if topic.lower() in trend_topic)
        score -= min(0.2, recent_penalty)

        # Ensure score is within bounds
        return max(0.0, min(1.0, score))

    async def extract_keywords_from_input(self, user_input: str) -> List[str]:
        """Extract relevant keywords from user input"""
        input_lower = user_input.lower()

        # Remove common words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after",
            "above", "below", "between", "among", "i", "you", "he", "she", "it", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
            "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "shall"
        }

        # Extract words
        words = []
        for word in input_lower.split():
            # Remove punctuation
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2 and clean_word not in stop_words:
                words.append(clean_word)

        # Also look for key phrases
        key_phrases = []
        phrase_indicators = {
            "ai": ["artificial intelligence", "machine learning", "chatgpt"],
            "environment": ["climate change", "global warming", "sustainability"],
            "technology": ["social media", "mobile apps", "virtual reality"],
            "health": ["mental health", "physical fitness", "nutrition"],
            "entertainment": ["streaming services", "video games", "movies"]
        }

        for category, phrases in phrase_indicators.items():
            for phrase in phrases:
                if phrase in input_lower:
                    key_phrases.append(category)

        return words + key_phrases

    async def get_fallback_trend(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get a fallback trend when no relevant trends are found"""

        # Try to get general trending topics
        general_trends = await self.trends_db.get_trending_topics(limit=5)

        if general_trends:
            # Pick a random trending topic
            selected_trend = random.choice(general_trends)
            return {
                "topic": selected_trend.get("name", "current events"),
                "category": selected_trend.get("category", "general"),
                "keywords": selected_trend.get("keywords", ["news", "discussion"]),
                "confidence": 0.4,
                "description": selected_trend.get("description", "General trending topic"),
                "conversation_starters": selected_trend.get("conversation_starters", []),
                "trend_data": selected_trend,
                "alternative_topics": []
            }
        else:
            # Ultimate fallback
            return {
                "topic": "general conversation",
                "category": "general",
                "keywords": ["chat", "conversation", "talk"],
                "confidence": 0.3,
                "description": "General conversation topic",
                "conversation_starters": [
                    "What's been on your mind lately?",
                    "How has your day been going?",
                    "Is there anything interesting happening in your life?"
                ],
                "trend_data": {},
                "alternative_topics": []
            }

    async def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached trend data is still valid"""
        if cache_key not in self.trend_cache:
            return False

        if cache_key not in self.cache_expiry:
            return False

        return datetime.now() < self.cache_expiry[cache_key]

    async def get_trending_categories(self) -> List[Dict[str, Any]]:
        """Get trending categories with their popularity"""
        categories = {}
        trends = await self.trends_db.get_trending_topics(limit=20)

        for trend in trends:
            category = trend.get("category", "general")
            if category not in categories:
                categories[category] = {
                    "name": category,
                    "count": 0,
                    "total_popularity": 0.0,
                    "trends": []
                }

            categories[category]["count"] += 1
            categories[category]["total_popularity"] += trend.get("popularity_score", 0.5)
            categories[category]["trends"].append(trend.get("name", ""))

        # Calculate average popularity and format results
        result = []
        for cat_data in categories.values():
            avg_popularity = cat_data["total_popularity"] / cat_data["count"]
            result.append({
                "category": cat_data["name"],
                "trend_count": cat_data["count"],
                "average_popularity": avg_popularity,
                "sample_trends": cat_data["trends"][:3]  # First 3 trends as samples
            })

        # Sort by average popularity
        result.sort(key=lambda x: x["average_popularity"], reverse=True)
        return result

    async def suggest_conversation_starter(self, trend_topic: str, user_context: Dict[str, Any]) -> str:
        """Generate a conversation starter for a specific trend topic"""

        # Try to get trend-specific starter
        trend_data = await self.trends_db.search_trends(trend_topic)
        if trend_data:
            trend = trend_data[0]
            starters = trend.get("conversation_starters", [])
            if starters:
                return random.choice(starters)

        # Generate generic starter based on topic
        user_profile = user_context.get("user_profile", {})
        conversation_style = user_profile.get("conversation_style", "casual")

        if conversation_style == "formal":
            starters = [
                f"I'd like to discuss {trend_topic}. What are your thoughts on this matter?",
                f"Have you been following the developments regarding {trend_topic}?",
                f"I'm interested in your perspective on {trend_topic}."
            ]
        else:  # casual or friendly
            starters = [
                f"Have you heard about {trend_topic}? What do you think?",
                f"I've been seeing a lot about {trend_topic} lately. Are you into that?",
                f"So, {trend_topic} is trending. Have you checked it out?",
                f"What's your take on all this {trend_topic} stuff?"
            ]

        return random.choice(starters)

    async def analyze_trend_sentiment(self, trend_name: str) -> Dict[str, Any]:
        """Analyze the sentiment and engagement potential of a trend"""

        # Simple sentiment analysis based on keywords
        positive_indicators = ["innovation", "success", "breakthrough", "improvement", "growth", "positive"]
        negative_indicators = ["crisis", "problem", "decline", "controversy", "issue", "concern"]
        neutral_indicators = ["update", "change", "development", "news", "report", "study"]

        trend_lower = trend_name.lower()

        positive_score = sum(1 for word in positive_indicators if word in trend_lower)
        negative_score = sum(1 for word in negative_indicators if word in trend_lower)
        neutral_score = sum(1 for word in neutral_indicators if word in trend_lower)

        # Determine dominant sentiment
        if positive_score > negative_score and positive_score > neutral_score:
            sentiment = "positive"
            confidence = positive_score / (positive_score + negative_score + neutral_score + 1)
        elif negative_score > positive_score and negative_score > neutral_score:
            sentiment = "negative"
            confidence = negative_score / (positive_score + negative_score + neutral_score + 1)
        else:
            sentiment = "neutral"
            confidence = (neutral_score + 1) / (positive_score + negative_score + neutral_score + 2)

        # Estimate engagement potential
        engagement_keywords = ["viral", "trending", "popular", "breaking", "exclusive", "amazing"]
        engagement_score = sum(1 for word in engagement_keywords if word in trend_lower)
        engagement_potential = min(1.0, engagement_score * 0.2 + 0.3)

        return {
            "sentiment": sentiment,
            "sentiment_confidence": confidence,
            "engagement_potential": engagement_potential,
            "conversation_safety": "high" if sentiment != "negative" else "medium",
            "recommended_approach": self.get_conversation_approach(sentiment)
        }

    def get_conversation_approach(self, sentiment: str) -> str:
        """Get recommended conversation approach based on sentiment"""
        approaches = {
            "positive": "enthusiastic and encouraging",
            "negative": "thoughtful and balanced",
            "neutral": "curious and exploratory"
        }
        return approaches.get(sentiment, "balanced")

    async def clear_cache(self) -> None:
        """Clear the trend cache"""
        self.trend_cache.clear()
        self.cache_expiry.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get trend agent statistics"""
        return {
            "cache_size": len(self.trend_cache),
            "cache_duration_minutes": self.cache_duration_minutes,
            "relevance_threshold": self.relevance_threshold,
            "trends_db_available": hasattr(self.trends_db, 'trends') and bool(self.trends_db.trends),
            "active_cache_entries": sum(1 for key in self.cache_expiry.keys()
                                      if datetime.now() < self.cache_expiry[key])
        }