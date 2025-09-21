"""
Trend Data Ingestor - Collects trending topics from Google Trends and News sources
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json

# Google Trends
from pytrends.request import TrendReq

# News and web scraping
import feedparser
import aiohttp
from bs4 import BeautifulSoup

# Database
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_resolver import TrendsDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendIngestor:
    """Ingests trending topics from various sources"""

    def __init__(self):
        self.db = TrendsDatabase()
        self.pytrends = None
        self.session = None

    async def initialize(self):
        """Initialize database and HTTP session"""
        await self.db.initialize()
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
        if hasattr(self.db, 'close'):
            if asyncio.iscoroutinefunction(self.db.close):
                await self.db.close()
            else:
                self.db.close()

    async def fetch_google_trends(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """Fetch trending topics from Google Trends"""
        try:
            loop = asyncio.get_event_loop()

            # Default keywords if none provided
            if not keywords:
                keywords = ['AI', 'technology', 'health', 'education', 'business',
                           'entertainment', 'sports', 'science', 'politics']

            # Initialize pytrends in executor (blocking operation)
            self.pytrends = await loop.run_in_executor(
                None,
                lambda: TrendReq(hl='en-US', tz=360)
            )

            trends_data = []

            for keyword in keywords:
                try:
                    # Build payload
                    await loop.run_in_executor(
                        None,
                        self.pytrends.build_payload,
                        [keyword],
                        'now 1-d'  # Last 24 hours
                    )

                    # Get related queries
                    related = await loop.run_in_executor(
                        None,
                        self.pytrends.related_queries
                    )

                    if keyword in related and related[keyword]['rising'] is not None:
                        rising_queries = related[keyword]['rising']

                        for _, row in rising_queries.iterrows():
                            query = row['query']
                            value = row['value']

                            # Determine difficulty based on search volume
                            difficulty = 'beginner'
                            if value > 1000:
                                difficulty = 'intermediate'
                            if value > 5000:
                                difficulty = 'advanced'

                            trend_topic = {
                                'title': f"Trending: {query}",
                                'category': self._categorize_keyword(keyword),
                                'keywords': [query, keyword],
                                'difficulty': difficulty,
                                'relevance_score': min(value / 100, 10.0),  # Normalize to 0-10
                                'source': 'google_trends',
                                'metadata': {
                                    'search_volume': value,
                                    'trend_type': 'rising',
                                    'base_keyword': keyword
                                }
                            }

                            trends_data.append(trend_topic)

                    # Also get trending searches (if available for region)
                    try:
                        trending = await loop.run_in_executor(
                            None,
                            self.pytrends.trending_searches,
                            'united_states'
                        )

                        for search_term in trending[0][:5]:  # Top 5 trending
                            trend_topic = {
                                'title': f"Hot Topic: {search_term}",
                                'category': 'general',
                                'keywords': [search_term],
                                'difficulty': 'intermediate',
                                'relevance_score': 8.0,
                                'source': 'google_trends_hot',
                                'metadata': {
                                    'trend_type': 'hot_topic'
                                }
                            }
                            trends_data.append(trend_topic)
                    except:
                        pass  # Trending searches might not be available for all regions

                except Exception as e:
                    logger.warning(f"Error fetching trends for keyword {keyword}: {e}")
                    continue

                # Small delay to avoid rate limiting
                await asyncio.sleep(1)

            logger.info(f"Fetched {len(trends_data)} topics from Google Trends")
            return trends_data

        except Exception as e:
            logger.error(f"Error in fetch_google_trends: {e}")
            return []

    async def fetch_news_topics(self) -> List[Dict[str, Any]]:
        """Fetch trending topics from news RSS feeds"""
        news_feeds = {
            'bbc_tech': ('http://feeds.bbci.co.uk/news/technology/rss.xml', 'technology'),
            'bbc_health': ('http://feeds.bbci.co.uk/news/health/rss.xml', 'health'),
            'reuters_world': ('http://feeds.reuters.com/reuters/worldNews', 'world'),
            'techcrunch': ('http://feeds.feedburner.com/TechCrunch/', 'technology'),
            'nyt_science': ('https://rss.nytimes.com/services/xml/rss/nyt/Science.xml', 'science'),
        }

        news_topics = []

        for feed_name, (feed_url, category) in news_feeds.items():
            try:
                # Parse RSS feed
                feed = await self._parse_feed_async(feed_url)

                if not feed or not feed.entries:
                    continue

                # Process top 5 entries from each feed
                for entry in feed.entries[:5]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    published = entry.get('published_parsed', None)

                    # Extract keywords from title and summary
                    keywords = self._extract_keywords(title + ' ' + summary)

                    # Calculate recency score (newer = higher score)
                    recency_score = 5.0
                    if published:
                        published_date = datetime.fromtimestamp(
                            asyncio.get_event_loop().time()
                        )
                        days_old = (datetime.now() - published_date).days
                        recency_score = max(1.0, 10.0 - days_old)

                    news_topic = {
                        'title': title[:200],  # Limit title length
                        'category': category,
                        'keywords': keywords[:10],  # Top 10 keywords
                        'difficulty': self._assess_difficulty(title, summary),
                        'relevance_score': recency_score,
                        'source': f'news_{feed_name}',
                        'metadata': {
                            'summary': summary[:500],
                            'feed': feed_name,
                            'url': entry.get('link', '')
                        }
                    }

                    news_topics.append(news_topic)

            except Exception as e:
                logger.warning(f"Error fetching news from {feed_name}: {e}")
                continue

        logger.info(f"Fetched {len(news_topics)} topics from news feeds")
        return news_topics

    async def _parse_feed_async(self, url: str) -> Any:
        """Parse RSS feed asynchronously"""
        try:
            async with self.session.get(url, timeout=10) as response:
                content = await response.text()
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, feedparser.parse, content)
        except Exception as e:
            logger.error(f"Error parsing feed {url}: {e}")
            return None

    def _categorize_keyword(self, keyword: str) -> str:
        """Map keywords to categories"""
        category_map = {
            'AI': 'technology',
            'technology': 'technology',
            'health': 'health',
            'education': 'education',
            'business': 'business',
            'entertainment': 'entertainment',
            'sports': 'sports',
            'science': 'science',
            'politics': 'world'
        }
        return category_map.get(keyword, 'general')

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple frequency analysis"""
        # Simple keyword extraction (in production, use NLP libraries)
        import re
        from collections import Counter

        # Common stop words to exclude
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are',
                     'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                     'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'shall', 'to', 'of', 'in', 'for', 'with',
                     'by', 'from', 'about', 'into', 'through', 'during', 'how',
                     'when', 'where', 'why', 'what', 'which', 'who', 'whom',
                     'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                     'it', 'we', 'they', 'them', 'their', 'and', 'but', 'or',
                     'if', 'because', 'as', 'until', 'while', 'not', 'no', 'yes'}

        # Extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 3]

        # Get most common words
        word_freq = Counter(words)
        return [word for word, _ in word_freq.most_common(15)]

    def _assess_difficulty(self, title: str, summary: str) -> str:
        """Assess content difficulty based on text complexity"""
        text = title + ' ' + summary

        # Simple heuristic based on average word length and technical terms
        tech_terms = ['algorithm', 'neural', 'quantum', 'blockchain', 'encryption',
                     'protocol', 'infrastructure', 'implementation', 'architecture',
                     'framework', 'methodology', 'analysis', 'synthesis']

        words = text.lower().split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        tech_count = sum(1 for term in tech_terms if term in text.lower())

        if avg_word_length > 7 or tech_count >= 3:
            return 'advanced'
        elif avg_word_length > 5 or tech_count >= 1:
            return 'intermediate'
        else:
            return 'beginner'

    async def deduplicate_topics(self, topics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate topics based on title similarity"""
        seen_hashes = set()
        unique_topics = []

        for topic in topics:
            # Create a simple hash of the title for deduplication
            title_hash = hashlib.md5(
                topic['title'].lower().encode()
            ).hexdigest()[:8]

            if title_hash not in seen_hashes:
                seen_hashes.add(title_hash)
                unique_topics.append(topic)

        return unique_topics

    async def ingest_all(self) -> int:
        """Main ingestion method - fetches from all sources and saves to database"""
        try:
            all_topics = []

            # Fetch from Google Trends
            trends_topics = await self.fetch_google_trends()
            all_topics.extend(trends_topics)

            # Fetch from News feeds
            news_topics = await self.fetch_news_topics()
            all_topics.extend(news_topics)

            # Deduplicate
            unique_topics = await self.deduplicate_topics(all_topics)

            # Save to database
            saved_count = 0
            for topic in unique_topics:
                try:
                    # Check if topic already exists (by title)
                    existing = await self.db.search_topics(topic['title'][:50])
                    if not existing:
                        await self.db.save_topic(topic)
                        saved_count += 1
                except Exception as e:
                    logger.warning(f"Error saving topic: {e}")
                    continue

            logger.info(f"Ingestion complete. Saved {saved_count} new topics")
            return saved_count

        except Exception as e:
            logger.error(f"Error in ingest_all: {e}")
            return 0


async def run_single_ingestion():
    """Run a single ingestion cycle"""
    ingestor = TrendIngestor()
    try:
        await ingestor.initialize()
        count = await ingestor.ingest_all()
        print(f"Successfully ingested {count} new topics")
    finally:
        await ingestor.close()


if __name__ == "__main__":
    # Run ingestion when module is executed directly
    asyncio.run(run_single_ingestion())