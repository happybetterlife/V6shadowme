"""
Unified Trends Database - Handles both SQLite and in-memory storage with proper API
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Determine which database to use based on environment
USE_SQLITE = os.environ.get('USE_SQLITE_DB', 'true').lower() == 'true'
OFFLINE_MODE = os.environ.get('OFFLINE_MODE', 'false').lower() == 'true'

if USE_SQLITE and not OFFLINE_MODE:
    try:
        from .sqlite_trends_db import SQLiteTrendsDatabase

        class TrendsDatabase(SQLiteTrendsDatabase):
            """SQLite-backed database with extended API"""

            def __init__(self):
                super().__init__()
                logger.info("Using SQLite database for trends")

    except ImportError:
        logger.warning("SQLite database not available, falling back to in-memory")
        USE_SQLITE = False

if not USE_SQLITE or OFFLINE_MODE:
    # Use the original in-memory TrendsDatabase with extended methods
    from .trends_db import TrendsDatabase as OriginalTrendsDB

    class TrendsDatabase(OriginalTrendsDB):
        """In-memory database with full API compatibility"""

        def __init__(self):
            super().__init__()
            self.saved_topics: List[Dict[str, Any]] = []
            logger.info("Using in-memory database for trends (offline/dev mode)")

        async def save_topic(self, topic: Dict[str, Any]) -> bool:
            """Save a topic to in-memory storage"""
            try:
                # Add timestamp if not present
                if 'timestamp' not in topic:
                    topic['timestamp'] = datetime.now().isoformat()

                # Add to saved topics
                self.saved_topics.append(topic)

                # Also add to trends for immediate availability
                self.trends.append(topic)

                logger.info(f"Saved topic (in-memory): {topic.get('title', 'Unknown')}")
                return True
            except Exception as e:
                logger.error(f"Error saving topic: {e}")
                return False

        async def search_topics(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
            """Search topics by title"""
            query_lower = query.lower()
            results = []

            # Search in all available topics
            all_topics = self.trends + self.saved_topics

            for topic in all_topics:
                title = topic.get('title', '').lower()
                if query_lower in title:
                    results.append(topic)
                    if len(results) >= limit:
                        break

            return results

        async def execute_query(self, query: str, params=None) -> List[Dict[str, Any]]:
            """Mock execute_query for compatibility"""
            # Handle common queries
            if "COUNT(*)" in query:
                total = len(self.trends) + len(self.saved_topics)
                return [{'count': total}]
            elif "SELECT * FROM trending_topics" in query:
                return self.trends + self.saved_topics
            else:
                return []

        def close(self):
            """No-op for in-memory database"""
            logger.info("In-memory database closed")

        async def close(self):
            """Async no-op for in-memory database"""
            logger.info("In-memory database closed")


# Export the unified TrendsDatabase
__all__ = ['TrendsDatabase']