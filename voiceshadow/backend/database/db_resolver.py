"""
Database resolver - Robust import chain for database implementations
"""

import importlib
import logging
from typing import Type, List, Tuple, Any

logger = logging.getLogger(__name__)


def resolve_database_class(candidates: List[Tuple[str, str]], purpose: str = "database") -> Type[Any]:
    """
    Resolve database class from a list of candidates in priority order.

    Args:
        candidates: List of (module_name, class_name) tuples in priority order
        purpose: Description of what this database is for (for error messages)

    Returns:
        The first successfully imported database class

    Raises:
        ImportError: If no database implementation is available
    """
    last_error = None

    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            logger.info("Using %s.%s for %s", module_name, class_name, purpose)
            return cls
        except (ImportError, AttributeError) as exc:
            last_error = exc
            logger.debug("Cannot load %s.%s: %s", module_name, class_name, exc)

    # If we get here, no candidates worked
    candidate_list = [f"{mod}.{cls}" for mod, cls in candidates]
    raise ImportError(
        f"No {purpose} implementation available. Tried: {', '.join(candidate_list)}"
    ) from last_error


def resolve_trends_db() -> Type[Any]:
    """Resolve TrendsDatabase implementation with fallback chain."""
    candidates = [
        ("database.unified_trends_db", "TrendsDatabase"),
        ("database.sqlite_trends_db", "SQLiteTrendsDatabase"),
        ("database.trends_db", "TrendsDatabase"),
    ]
    return resolve_database_class(candidates, "trends database")


def resolve_cornell_db() -> Type[Any]:
    """Resolve CornellDatabase implementation."""
    candidates = [
        ("database.cornell_db", "CornellDatabase"),
    ]
    return resolve_database_class(candidates, "Cornell dialogue database")


def resolve_personachat_db() -> Type[Any]:
    """Resolve PersonaChatDatabase implementation."""
    candidates = [
        ("database.personachat_db", "PersonaChatDatabase"),
    ]
    return resolve_database_class(candidates, "PersonaChat database")


# Pre-resolved classes for common use
try:
    TrendsDatabase = resolve_trends_db()
except ImportError as e:
    logger.error("Failed to resolve TrendsDatabase: %s", e)
    TrendsDatabase = None

try:
    CornellDatabase = resolve_cornell_db()
except ImportError as e:
    logger.error("Failed to resolve CornellDatabase: %s", e)
    CornellDatabase = None

try:
    PersonaChatDatabase = resolve_personachat_db()
except ImportError as e:
    logger.error("Failed to resolve PersonaChatDatabase: %s", e)
    PersonaChatDatabase = None


def create_fallback_trends_db():
    """Create a minimal fallback TrendsDatabase for testing/development."""
    class FallbackTrendsDatabase:
        """Minimal fallback implementation for development/testing."""

        def __init__(self):
            logger.warning("Using fallback TrendsDatabase - limited functionality")

        async def initialize(self):
            logger.info("Fallback TrendsDatabase initialized")

        async def get_todays_topic(self, user_profile):
            return {"name": "general conversation", "keywords": ["chat", "discuss"]}

        async def get_trending_topics(self, category=None, limit=10):
            return [
                {
                    "title": "Sample Topic: Technology Trends",
                    "category": "technology",
                    "keywords": ["tech", "innovation"],
                    "difficulty": "intermediate",
                    "relevance_score": 7.5
                }
            ]

        async def search_trends(self, query):
            return []

        async def save_topic(self, topic):
            logger.debug("Fallback: topic save ignored")
            return True

        async def search_topics(self, query, limit=10):
            return []

        def close(self):
            pass

        async def close(self):
            pass

    return FallbackTrendsDatabase


# If no TrendsDatabase was resolved, use fallback
if TrendsDatabase is None:
    logger.warning("No TrendsDatabase implementation found, using fallback")
    TrendsDatabase = create_fallback_trends_db()