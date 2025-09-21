"""Google Trends backed database for trending conversation topics."""

from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from pytrends.request import TrendReq  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    TrendReq = None  # type: ignore


logger = logging.getLogger(__name__)


class TrendsDatabase:
    """Fetches and caches trending topics from Google Trends."""

    def __init__(self) -> None:
        self.trends: List[Dict[str, Any]] = []
        self.historical_trends: List[Dict[str, Any]] = []
        self.loaded = False
        self.last_updated: Optional[datetime] = None
        self.cache_duration = int(os.environ.get("TRENDS_CACHE_DURATION", "3600"))
        self.geo = os.environ.get("GOOGLE_TRENDS_GEO", "united_states")
        self.pytrends: Optional[TrendReq] = None
        self.saved_topics: List[Dict[str, Any]] = []  # Store manually saved topics

        self.sample_trends = self._build_sample_trends()

    async def initialize(self) -> None:
        self._initialize_pytrends()
        await self.load()

    async def load(self) -> None:
        if self.last_updated and (datetime.now() - self.last_updated).total_seconds() < self.cache_duration:
            return

        try:
            fetched = await self._fetch_google_trends()
            if fetched:
                self.trends = fetched
                self.historical_trends.extend(fetched)
                logger.info("Trends Database: Loaded %s topics from Google Trends", len(self.trends))
            else:
                logger.warning("Trends Database: Falling back to sample trends")
                self.trends = self._generate_sample_trends()

            self.last_updated = datetime.now()
            self.loaded = True
        except Exception as exc:
            logger.error("Error loading trends database: %s", exc)
            self.trends = self._generate_sample_trends()
            self.loaded = True

    async def get_todays_topic(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        if not self.loaded:
            await self.load()

        if not self.trends:
            return {
                "title": "general conversation",
                "keywords": ["chat", "conversation", "talk"],
                "category": "general",
                "popularity_score": 0.5,
            }

        if user_profile and "interests" in user_profile:
            user_interests = [interest.lower() for interest in user_profile["interests"]]
            scored: List[tuple[float, Dict[str, Any]]] = []
            for trend in self.trends:
                score = trend.get("popularity_score", 0.5)
                trend_keywords = [kw.lower() for kw in trend.get("keywords", [])]
                for interest in user_interests:
                    if any(interest in keyword for keyword in trend_keywords):
                        score += 0.2
                scored.append((score, trend))
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1]

        return random.choice(self.trends)

    async def get_trending_topics(self, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending topics, optionally filtered by category"""
        if not self.loaded:
            await self.load()

        if not self.trends:
            return []

        filtered_trends = self.trends
        if category:
            filtered_trends = [trend for trend in self.trends if trend.get("category", "").lower() == category.lower()]

        # Sort by popularity score
        sorted_trends = sorted(filtered_trends, key=lambda x: x.get("popularity_score", 0.5), reverse=True)
        return sorted_trends[:limit]

    async def save_topic(self, topic: Dict[str, Any]) -> None:
        """Save a new topic to the database."""
        # Add timestamp if not present
        if 'timestamp' not in topic:
            topic['timestamp'] = datetime.now().isoformat()

        # Add to saved topics
        self.saved_topics.append(topic)

        # Also add to trends for immediate availability
        self.trends.append(topic)

        logger.info(f"Saved topic: {topic.get('title', 'Unknown')}")

    async def search_topics(self, query: str) -> List[Dict[str, Any]]:
        """Search for existing topics by title (to check for duplicates)."""
        query_lower = query.lower()
        results = []

        # Search in all available topics
        all_topics = self.trends + self.saved_topics
        for topic in all_topics:
            title = topic.get('title', '').lower()
            if query_lower in title:
                results.append(topic)

        return results

    def close(self) -> None:
        """Clean up resources."""
        # Currently no persistent connection to close
        # This method is for compatibility with the ingestion system
        logger.info("TrendsDatabase closed")

    async def search_trends(self, query: str) -> List[Dict[str, Any]]:
        """Search trends by query string"""
        if not self.loaded:
            await self.load()

        if not self.trends or not query:
            return []

        query_lower = query.lower()
        matching_trends = []

        for trend in self.trends:
            score = 0.0

            # Check title/name match
            title = trend.get("title", "").lower()
            name = trend.get("name", "").lower()
            if query_lower in title or query_lower in name:
                score += 1.0

            # Check keywords match
            keywords = trend.get("keywords", [])
            for keyword in keywords:
                if query_lower in keyword.lower():
                    score += 0.5

            # Check description match
            description = trend.get("description", "").lower()
            if query_lower in description:
                score += 0.3

            if score > 0:
                trend_copy = trend.copy()
                trend_copy["search_score"] = score
                matching_trends.append(trend_copy)

        # Sort by search score and popularity
        matching_trends.sort(key=lambda x: (x.get("search_score", 0), x.get("popularity_score", 0.5)), reverse=True)
        return matching_trends

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_pytrends(self) -> None:
        if TrendReq is None:
            logger.warning("pytrends is not installed; trends will use static samples")
            self.pytrends = None
            return

        if self.pytrends is not None:
            return

        try:
            hl = os.environ.get("GOOGLE_TRENDS_HL", "en-US")
            tz = int(os.environ.get("GOOGLE_TRENDS_TZ", "0"))
            self.pytrends = TrendReq(hl=hl, tz=tz)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to initialize pytrends: %s", exc)
            self.pytrends = None

    async def _fetch_google_trends(self) -> List[Dict[str, Any]]:
        if not self.pytrends:
            return []

        try:
            df = self.pytrends.trending_searches(pn=self.geo)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to fetch Google Trends: %s", exc)
            return []

        if df.empty:
            return []

        terms = df.iloc[:, 0].tolist()[:10]
        trends: List[Dict[str, Any]] = []

        for term in terms:
            keywords = await self._fetch_related_keywords(term)
            popularity = await self._estimate_popularity(term)
            trends.append(
                {
                    "id": f"google_{term.replace(' ', '_').lower()}",
                    "title": term,
                    "name": term,
                    "category": "trending",
                    "keywords": keywords or [term.lower()],
                    "popularity_score": popularity,
                    "sentiment": "neutral",
                    "description": f"Trending topic: {term}",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "sources": ["google_trends"],
                    "conversation_starters": [
                        f"Have you heard about {term}?",
                        f"What do you think about {term}?",
                        f"{term} has been very popular lately."
                    ],
                }
            )

        return trends

    async def _fetch_related_keywords(self, term: str) -> List[str]:
        if not self.pytrends:
            return []

        try:
            self.pytrends.build_payload([term], timeframe="now 7-d", geo=self.geo)
            related = self.pytrends.related_queries()
        except Exception as exc:  # pragma: no cover
            logger.debug("Unable to fetch related queries for %s: %s", term, exc)
            return []

        if term not in related or related[term]["top"] is None:
            return []

        top_df = related[term]["top"]
        return [str(q).lower() for q in top_df["query"].head(5).tolist()]

    async def _estimate_popularity(self, term: str) -> float:
        if not self.pytrends:
            return 0.5

        try:
            interest = self.pytrends.interest_over_time()
        except Exception:
            interest = None

        if interest is None or term not in interest.columns or interest.empty:
            return 0.5

        score = float(interest[term].mean()) / 100.0
        return max(0.1, min(1.0, score))

    def _generate_sample_trends(self) -> List[Dict[str, Any]]:
        sample_copy = [trend.copy() for trend in self.sample_trends]
        for trend in sample_copy:
            trend["popularity_score"] = max(0.1, min(1.0, trend["popularity_score"] + random.uniform(-0.1, 0.1)))
            trend["date"] = datetime.now().strftime("%Y-%m-%d")
        return sample_copy

    def _build_sample_trends(self) -> List[Dict[str, Any]]:
        now = datetime.now().strftime("%Y-%m-%d")
        return [
            {
                "id": "trend_ai",
                "name": "AI Technology",
                "title": "AI Technology",
                "category": "technology",
                "keywords": ["artificial intelligence", "machine learning", "chatgpt", "automation", "ai ethics"],
                "popularity_score": 0.95,
                "sentiment": "positive",
                "description": "Latest developments in artificial intelligence and machine learning",
                "date": now,
                "sources": ["sample"],
                "conversation_starters": [
                    "Have you heard about the latest AI developments?",
                    "What's your take on AI changing our daily lives?",
                    "I've been following the AI news lately. It's fascinating!",
                ],
            },
            {
                "id": "trend_sustainable",
                "name": "Sustainable Living",
                "title": "Sustainable Living",
                "category": "lifestyle",
                "keywords": ["sustainability", "eco-friendly", "climate change", "green energy", "recycling"],
                "popularity_score": 0.88,
                "sentiment": "positive",
                "description": "Growing interest in sustainable lifestyle choices and environmental consciousness",
                "date": now,
                "sources": ["sample"],
                "conversation_starters": [
                    "Have you made any eco-friendly changes recently?",
                    "The sustainability movement is really picking up steam.",
                    "I've been trying to live more sustainably lately.",
                ],
            },
        ]
