"""
Enhanced Trend Data Ingestor - Multiple data sources for comprehensive trend collection
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib
import json

# HTTP and Web scraping
import aiohttp
from bs4 import BeautifulSoup
import feedparser

# Database
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_resolver import TrendsDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTrendIngestor:
    """Enhanced ingestion with multiple data sources"""

    def __init__(self):
        self.db = TrendsDatabase()
        self.session = None

        # API Keys (loaded from environment)
        self.newsapi_key = os.environ.get('NEWSAPI_KEY', '')
        self.reddit_client_id = os.environ.get('REDDIT_CLIENT_ID', '')
        self.reddit_client_secret = os.environ.get('REDDIT_CLIENT_SECRET', '')

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

    async def fetch_newsapi_headlines(self) -> List[Dict[str, Any]]:
        """Fetch top headlines from NewsAPI"""
        if not self.newsapi_key:
            logger.warning("NewsAPI key not configured")
            return []

        topics = []

        # Different categories to fetch
        categories = ['technology', 'science', 'health', 'business', 'entertainment', 'sports']

        for category in categories:
            try:
                url = 'https://newsapi.org/v2/top-headlines'
                params = {
                    'apiKey': self.newsapi_key,
                    'category': category,
                    'country': 'us',
                    'pageSize': 10
                }

                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()

                        for article in data.get('articles', [])[:5]:  # Top 5 per category
                            title = article.get('title', '')
                            description = article.get('description', '')

                            if not title:
                                continue

                            # Extract keywords from title and description
                            keywords = self._extract_keywords(title + ' ' + description)

                            topic = {
                                'title': title[:200],
                                'category': category,
                                'keywords': keywords[:10],
                                'difficulty': self._assess_difficulty(title, description),
                                'relevance_score': 8.0,  # NewsAPI headlines are generally relevant
                                'source': 'newsapi',
                                'metadata': {
                                    'description': description[:500] if description else '',
                                    'url': article.get('url', ''),
                                    'source_name': article.get('source', {}).get('name', ''),
                                    'published_at': article.get('publishedAt', '')
                                }
                            }
                            topics.append(topic)
                    else:
                        logger.warning(f"NewsAPI returned status {response.status} for category {category}")

                # Rate limiting
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching NewsAPI category {category}: {e}")
                continue

        logger.info(f"Fetched {len(topics)} topics from NewsAPI")
        return topics

    async def fetch_reddit_trending(self) -> List[Dict[str, Any]]:
        """Fetch trending topics from Reddit"""
        if not self.reddit_client_id or not self.reddit_client_secret:
            logger.warning("Reddit API credentials not configured")
            return []

        topics = []

        try:
            # Get OAuth token
            auth_url = 'https://www.reddit.com/api/v1/access_token'
            auth = aiohttp.BasicAuth(self.reddit_client_id, self.reddit_client_secret)
            data = {'grant_type': 'client_credentials'}
            headers = {'User-Agent': 'VoiceShadowTrendBot/1.0'}

            async with self.session.post(auth_url, auth=auth, data=data, headers=headers) as response:
                if response.status != 200:
                    logger.warning(f"Reddit auth failed with status {response.status}")
                    return []

                token_data = await response.json()
                access_token = token_data.get('access_token')

            # Fetch from different subreddits
            headers['Authorization'] = f'bearer {access_token}'

            subreddits = [
                ('technology', 'technology'),
                ('science', 'science'),
                ('worldnews', 'world'),
                ('health', 'health'),
                ('todayilearned', 'education'),
                ('programming', 'technology'),
                ('MachineLearning', 'technology'),
                ('Futurology', 'technology')
            ]

            for subreddit, category in subreddits:
                try:
                    url = f'https://oauth.reddit.com/r/{subreddit}/hot'
                    params = {'limit': 5}

                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            data = await response.json()

                            for post in data.get('data', {}).get('children', []):
                                post_data = post.get('data', {})
                                title = post_data.get('title', '')
                                selftext = post_data.get('selftext', '')[:500]
                                score = post_data.get('score', 0)

                                if not title:
                                    continue

                                # Calculate relevance based on score
                                relevance = min(10.0, score / 1000) if score > 0 else 5.0

                                keywords = self._extract_keywords(title + ' ' + selftext)

                                topic = {
                                    'title': title[:200],
                                    'category': category,
                                    'keywords': keywords[:10],
                                    'difficulty': self._assess_difficulty(title, selftext),
                                    'relevance_score': relevance,
                                    'source': 'reddit',
                                    'metadata': {
                                        'subreddit': subreddit,
                                        'score': score,
                                        'num_comments': post_data.get('num_comments', 0),
                                        'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                        'created_utc': post_data.get('created_utc', 0)
                                    }
                                }
                                topics.append(topic)

                    # Rate limiting
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.warning(f"Error fetching Reddit r/{subreddit}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error in fetch_reddit_trending: {e}")

        logger.info(f"Fetched {len(topics)} topics from Reddit")
        return topics

    async def fetch_expanded_rss_feeds(self) -> List[Dict[str, Any]]:
        """Fetch from expanded list of RSS feeds"""
        rss_feeds = {
            # Technology
            'ars_technica': ('http://feeds.arstechnica.com/arstechnica/index', 'technology'),
            'wired': ('https://www.wired.com/feed/rss', 'technology'),
            'verge': ('https://www.theverge.com/rss/index.xml', 'technology'),
            'hackernews': ('https://hnrss.org/frontpage', 'technology'),
            'mit_tech': ('https://www.technologyreview.com/feed/', 'technology'),

            # Science
            'nature': ('http://feeds.nature.com/nature/rss/current', 'science'),
            'science_daily': ('https://www.sciencedaily.com/rss/all.xml', 'science'),
            'nasa': ('https://www.nasa.gov/rss/dyn/breaking_news.rss', 'science'),

            # Health
            'medical_news': ('https://www.medicalnewstoday.com/rss/featurednews.xml', 'health'),
            'health_harvard': ('https://www.health.harvard.edu/blog/feed', 'health'),

            # Business
            'forbes': ('https://www.forbes.com/innovation/feed2/', 'business'),
            'bloomberg': ('https://feeds.bloomberg.com/markets/news.rss', 'business'),

            # Education
            'edweek': ('http://www.edweek.org/ew/section/feeds/rss.html', 'education'),

            # General News
            'guardian_world': ('https://www.theguardian.com/world/rss', 'world'),
            'ap_news': ('https://feeds.apnews.com/rss/apf-topnews', 'world'),
            'npr': ('https://feeds.npr.org/1001/rss.xml', 'general'),

            # Entertainment
            'variety': ('https://variety.com/feed/', 'entertainment'),

            # Sports
            'espn': ('https://www.espn.com/espn/rss/news', 'sports'),
        }

        all_topics = []

        for feed_name, (feed_url, category) in rss_feeds.items():
            try:
                feed = await self._parse_feed_async(feed_url)

                if not feed or not feed.entries:
                    continue

                # Process top entries from each feed
                for entry in feed.entries[:3]:  # Reduced to 3 per feed to avoid overload
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    published = entry.get('published_parsed', None)

                    if not title:
                        continue

                    keywords = self._extract_keywords(title + ' ' + summary)

                    # Calculate recency score
                    recency_score = 5.0
                    if published:
                        try:
                            from time import mktime
                            published_date = datetime.fromtimestamp(mktime(published))
                            days_old = (datetime.now() - published_date).days
                            recency_score = max(1.0, 10.0 - days_old)
                        except:
                            pass

                    topic = {
                        'title': title[:200],
                        'category': category,
                        'keywords': keywords[:10],
                        'difficulty': self._assess_difficulty(title, summary),
                        'relevance_score': recency_score,
                        'source': f'rss_{feed_name}',
                        'metadata': {
                            'summary': summary[:500],
                            'feed': feed_name,
                            'url': entry.get('link', '')
                        }
                    }

                    all_topics.append(topic)

            except Exception as e:
                logger.debug(f"Error fetching RSS feed {feed_name}: {e}")
                continue

        logger.info(f"Fetched {len(all_topics)} topics from RSS feeds")
        return all_topics

    async def fetch_hackernews_trending(self) -> List[Dict[str, Any]]:
        """Fetch trending stories from Hacker News API"""
        topics = []

        try:
            # Get top story IDs
            async with self.session.get('https://hacker-news.firebaseio.com/v0/topstories.json') as response:
                if response.status != 200:
                    return topics

                story_ids = await response.json()

                # Fetch top 10 stories
                for story_id in story_ids[:10]:
                    try:
                        async with self.session.get(f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json') as response:
                            if response.status == 200:
                                story = await response.json()

                                if not story or story.get('type') != 'story':
                                    continue

                                title = story.get('title', '')
                                score = story.get('score', 0)

                                # Determine category based on keywords in title
                                category = self._categorize_hackernews_story(title)

                                topic = {
                                    'title': title[:200],
                                    'category': category,
                                    'keywords': self._extract_keywords(title)[:10],
                                    'difficulty': 'intermediate',  # HN content tends to be technical
                                    'relevance_score': min(10.0, score / 100),
                                    'source': 'hackernews',
                                    'metadata': {
                                        'score': score,
                                        'comments': story.get('descendants', 0),
                                        'url': story.get('url', ''),
                                        'hn_url': f"https://news.ycombinator.com/item?id={story_id}"
                                    }
                                }

                                topics.append(topic)

                    except Exception as e:
                        logger.debug(f"Error fetching HN story {story_id}: {e}")
                        continue

                    # Small delay to be respectful
                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error fetching from Hacker News: {e}")

        logger.info(f"Fetched {len(topics)} topics from Hacker News")
        return topics

    def _categorize_hackernews_story(self, title: str) -> str:
        """Categorize HN story based on title keywords"""
        title_lower = title.lower()

        if any(word in title_lower for word in ['ai', 'ml', 'machine learning', 'neural', 'gpt', 'llm']):
            return 'technology'
        elif any(word in title_lower for word in ['health', 'medical', 'disease', 'therapy']):
            return 'health'
        elif any(word in title_lower for word in ['science', 'research', 'study', 'quantum']):
            return 'science'
        elif any(word in title_lower for word in ['business', 'startup', 'ipo', 'funding']):
            return 'business'
        elif any(word in title_lower for word in ['education', 'learning', 'university']):
            return 'education'
        else:
            return 'technology'  # Default for HN

    async def _parse_feed_async(self, url: str) -> Any:
        """Parse RSS feed asynchronously"""
        try:
            async with self.session.get(url, timeout=10) as response:
                content = await response.text()
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, feedparser.parse, content)
        except Exception as e:
            logger.debug(f"Error parsing feed {url}: {e}")
            return None

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        import re
        from collections import Counter

        # Common stop words
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

    def _assess_difficulty(self, title: str, content: str) -> str:
        """Assess content difficulty"""
        text = title + ' ' + content

        # Technical terms that indicate advanced content
        tech_terms = ['algorithm', 'neural', 'quantum', 'blockchain', 'encryption',
                     'protocol', 'infrastructure', 'implementation', 'architecture',
                     'framework', 'methodology', 'analysis', 'synthesis', 'paradigm',
                     'optimization', 'distributed', 'cryptographic', 'polynomial']

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
        """Remove duplicate topics"""
        seen_hashes = set()
        unique_topics = []

        for topic in topics:
            # Create hash of normalized title for deduplication
            normalized_title = ''.join(topic['title'].lower().split())[:50]
            title_hash = hashlib.md5(normalized_title.encode()).hexdigest()[:8]

            if title_hash not in seen_hashes:
                seen_hashes.add(title_hash)
                unique_topics.append(topic)

        return unique_topics

    async def ingest_all(self) -> int:
        """Fetch from all sources and save to database"""
        try:
            all_topics = []

            # Run all fetch operations concurrently
            tasks = [
                self.fetch_newsapi_headlines(),
                self.fetch_reddit_trending(),
                self.fetch_expanded_rss_feeds(),
                self.fetch_hackernews_trending()
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in fetch task: {result}")
                elif isinstance(result, list):
                    all_topics.extend(result)

            # Deduplicate
            unique_topics = await self.deduplicate_topics(all_topics)

            # Save to database
            saved_count = 0
            for topic in unique_topics:
                try:
                    # Check if topic already exists
                    existing = await self.db.search_topics(topic['title'][:50], limit=1)
                    if not existing:
                        success = await self.db.save_topic(topic)
                        if success:
                            saved_count += 1
                except Exception as e:
                    logger.debug(f"Error saving topic: {e}")
                    continue

            logger.info(f"Enhanced ingestion complete. Saved {saved_count} new topics out of {len(unique_topics)} unique")
            return saved_count

        except Exception as e:
            logger.error(f"Error in ingest_all: {e}")
            return 0


async def run_enhanced_ingestion():
    """Run enhanced ingestion"""
    ingestor = EnhancedTrendIngestor()
    try:
        await ingestor.initialize()
        count = await ingestor.ingest_all()
        print(f"Successfully ingested {count} new topics")
    finally:
        await ingestor.close()


if __name__ == "__main__":
    asyncio.run(run_enhanced_ingestion())