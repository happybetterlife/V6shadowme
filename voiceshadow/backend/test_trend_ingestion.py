#!/usr/bin/env python3
"""
Test script for trend data ingestion
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.trend_ingestor import TrendIngestor
try:
    from database.sqlite_trends_db import SQLiteTrendsDatabase as TrendsDatabase
except ImportError:
    from database.trends_db import TrendsDatabase
from agents.trend_agent import TrendAgent

async def test_ingestion():
    """Test the complete trend ingestion pipeline"""
    print("=" * 60)
    print("TREND DATA INGESTION TEST")
    print("=" * 60)

    # Step 1: Run the ingestor
    print("\n1. Running trend data ingestion...")
    ingestor = TrendIngestor()

    try:
        await ingestor.initialize()
        count = await ingestor.ingest_all()
        print(f"   ✓ Ingested {count} new topics")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    finally:
        await ingestor.close()

    # Step 2: Check database contents
    print("\n2. Checking database contents...")
    db = TrendsDatabase()

    try:
        await db.initialize()

        # Get total count
        total_topics = await db.execute_query("SELECT COUNT(*) as count FROM trending_topics")
        print(f"   ✓ Total topics in database: {total_topics[0]['count']}")

        # Get recent topics
        recent = await db.get_trending_topics(limit=5)
        print(f"   ✓ Recent topics:")
        for topic in recent[:5]:
            print(f"      - {topic['title'][:60]}... (Category: {topic['category']})")

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    finally:
        if hasattr(db, 'close'):
            if asyncio.iscoroutinefunction(db.close):
                await db.close()
            else:
                db.close()

    # Step 3: Test TrendAgent with new data
    print("\n3. Testing TrendAgent with new data...")
    agent = TrendAgent()

    try:
        await agent.initialize()

        # Test with different contexts
        test_contexts = [
            "I'm learning about artificial intelligence",
            "health and fitness tips",
            "latest technology news"
        ]

        for context in test_contexts:
            print(f"\n   Context: '{context}'")
            trends = await agent.get_relevant_trends(context, max_results=3)

            if trends:
                print(f"   ✓ Found {len(trends)} relevant trends:")
                for trend in trends:
                    print(f"      - {trend['title'][:50]}... (Score: {trend.get('relevance_score', 0):.2f})")
            else:
                print("   ⚠ No trends found for this context")

    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


async def test_individual_sources():
    """Test individual data sources"""
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL DATA SOURCES")
    print("=" * 60)

    ingestor = TrendIngestor()
    await ingestor.initialize()

    # Test Google Trends
    print("\n1. Testing Google Trends...")
    try:
        trends = await ingestor.fetch_google_trends(['AI', 'technology'])
        print(f"   ✓ Fetched {len(trends)} topics from Google Trends")
        if trends:
            print(f"      Sample: {trends[0]['title']}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test News feeds
    print("\n2. Testing News Feeds...")
    try:
        news = await ingestor.fetch_news_topics()
        print(f"   ✓ Fetched {len(news)} topics from news feeds")
        if news:
            print(f"      Sample: {news[0]['title'][:60]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    await ingestor.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\nStarting trend ingestion tests...\n")

    # Run main test
    asyncio.run(test_ingestion())

    # Optionally test individual sources
    print("\nWould you like to test individual data sources? (y/n): ", end="")
    if input().lower() == 'y':
        asyncio.run(test_individual_sources())