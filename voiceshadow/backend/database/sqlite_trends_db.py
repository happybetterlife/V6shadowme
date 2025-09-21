"""SQLite-backed database for persistent trend storage."""

import aiosqlite
import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SQLiteTrendsDatabase:
    """SQLite database for persistent trend topic storage."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Create database directory if it doesn't exist
            db_dir = Path(__file__).parent / "data"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "trends.db")

        self.db_path = db_path
        self.connection = None

    async def initialize(self):
        """Initialize database and create tables if they don't exist."""
        self.connection = await aiosqlite.connect(self.db_path)

        # Create trending_topics table
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS trending_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT,
                keywords TEXT,
                difficulty TEXT,
                relevance_score REAL,
                source TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(title)
            )
        """)

        # Create index for faster searches
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_title ON trending_topics(title)
        """)
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_category ON trending_topics(category)
        """)
        await self.connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON trending_topics(timestamp)
        """)

        await self.connection.commit()
        logger.info(f"SQLite database initialized at {self.db_path}")

    async def save_topic(self, topic: Dict[str, Any]) -> bool:
        """Save a topic to the database."""
        try:
            # Convert lists/dicts to JSON strings for storage
            keywords_json = json.dumps(topic.get('keywords', []))
            metadata_json = json.dumps(topic.get('metadata', {}))

            await self.connection.execute("""
                INSERT OR IGNORE INTO trending_topics
                (title, category, keywords, difficulty, relevance_score, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                topic.get('title'),
                topic.get('category'),
                keywords_json,
                topic.get('difficulty'),
                topic.get('relevance_score', 5.0),
                topic.get('source'),
                metadata_json
            ))

            await self.connection.commit()
            return True

        except Exception as e:
            logger.error(f"Error saving topic: {e}")
            return False

    async def search_topics(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for topics by title."""
        cursor = await self.connection.execute("""
            SELECT * FROM trending_topics
            WHERE title LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f"%{query}%", limit))

        rows = await cursor.fetchall()
        return self._rows_to_dicts(rows)

    async def get_trending_topics(self, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending topics, optionally filtered by category."""
        if category:
            cursor = await self.connection.execute("""
                SELECT * FROM trending_topics
                WHERE category = ?
                ORDER BY timestamp DESC, relevance_score DESC
                LIMIT ?
            """, (category, limit))
        else:
            cursor = await self.connection.execute("""
                SELECT * FROM trending_topics
                ORDER BY timestamp DESC, relevance_score DESC
                LIMIT ?
            """, (limit,))

        rows = await cursor.fetchall()
        return self._rows_to_dicts(rows)

    async def execute_query(self, query: str, params=None) -> List[Dict[str, Any]]:
        """Execute a custom query and return results."""
        cursor = await self.connection.execute(query, params or ())
        rows = await cursor.fetchall()

        # For COUNT queries
        if rows and len(rows[0]) == 1 and isinstance(rows[0][0], int):
            return [{'count': rows[0][0]}]

        return self._rows_to_dicts(rows)

    async def cleanup_old_topics(self, days: int = 30):
        """Remove topics older than specified days."""
        await self.connection.execute("""
            DELETE FROM trending_topics
            WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
        """, (days,))
        await self.connection.commit()
        logger.info(f"Cleaned up topics older than {days} days")

    def _rows_to_dicts(self, rows) -> List[Dict[str, Any]]:
        """Convert database rows to dictionaries."""
        topics = []
        for row in rows:
            topic = {
                'id': row[0],
                'title': row[1],
                'category': row[2],
                'keywords': json.loads(row[3]) if row[3] else [],
                'difficulty': row[4],
                'relevance_score': row[5],
                'source': row[6],
                'metadata': json.loads(row[7]) if row[7] else {},
                'timestamp': row[8]
            }
            topics.append(topic)
        return topics

    async def close(self):
        """Close the database connection."""
        if self.connection:
            await self.connection.close()
            logger.info("SQLite database connection closed")

    # Compatibility methods for existing code
    def close(self):
        """Synchronous close for compatibility."""
        if self.connection:
            asyncio.create_task(self.connection.close())
            logger.info("SQLite database connection closed")


# Use SQLite version as the main TrendsDatabase
TrendsDatabase = SQLiteTrendsDatabase