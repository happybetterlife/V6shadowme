"""
Cornell Movie Dialog Corpus Database Module

Provides access to movie dialogues for conversation training and pattern matching.
This module implements the Cornell Movie Dialogues Corpus functionality.
"""

import json
import os
import asyncio
import random
from typing import List, Dict, Any, Optional
from pathlib import Path


class CornellDatabase:
    """Cornell Movie Dialog Corpus database interface"""

    def __init__(self):
        self.dialogues = []
        self.characters = {}
        self.movies = {}
        self.loaded = False

        # Sample data structure for development
        self.sample_dialogues = [
            {
                "id": "L1045",
                "character1": "BIANCA",
                "character2": "CAMERON",
                "text1": "I hate the way you talk to me.",
                "text2": "I hate the way you drive.",
                "movie": "10 things i hate about you",
                "genre": "comedy",
                "keywords": ["relationship", "argument", "feelings"]
            },
            {
                "id": "L1046",
                "character1": "CAMERON",
                "character2": "BIANCA",
                "text1": "I hate the way you look at me.",
                "text2": "I hate your big dumb combat boots.",
                "movie": "10 things i hate about you",
                "genre": "comedy",
                "keywords": ["relationship", "argument", "appearance"]
            },
            {
                "id": "L2001",
                "character1": "HARRY",
                "character2": "SALLY",
                "text1": "I love that you get cold when it's 71 degrees out.",
                "text2": "I love that it takes you an hour and a half to order a sandwich.",
                "movie": "when harry met sally",
                "genre": "romance",
                "keywords": ["love", "romance", "quirks", "appreciation"]
            },
            {
                "id": "L2002",
                "character1": "SALLY",
                "character2": "HARRY",
                "text1": "You're the worst kind of high maintenance.",
                "text2": "You're the best kind of low maintenance.",
                "movie": "when harry met sally",
                "genre": "romance",
                "keywords": ["relationship", "personality", "maintenance"]
            },
            {
                "id": "L3001",
                "character1": "JACK",
                "character2": "ROSE",
                "text1": "I want you to draw me like one of your French girls.",
                "text2": "I can't do this, Jack.",
                "movie": "titanic",
                "genre": "drama",
                "keywords": ["art", "vulnerability", "trust"]
            },
            {
                "id": "L4001",
                "character1": "ANDY",
                "character2": "RED",
                "text1": "Hope is a good thing, maybe the best of things.",
                "text2": "Hope is a dangerous thing.",
                "movie": "the shawshank redemption",
                "genre": "drama",
                "keywords": ["hope", "philosophy", "life", "prison"]
            },
            {
                "id": "L5001",
                "character1": "DOROTHY",
                "character2": "GLINDA",
                "text1": "There's no place like home.",
                "text2": "You always had the power to go back to Kansas.",
                "movie": "the wizard of oz",
                "genre": "fantasy",
                "keywords": ["home", "power", "journey", "realization"]
            },
            {
                "id": "L6001",
                "character1": "LUKE",
                "character2": "VADER",
                "text1": "I'll never join you!",
                "text2": "Search your feelings. You know it to be true.",
                "movie": "star wars",
                "genre": "sci-fi",
                "keywords": ["family", "conflict", "truth", "force"]
            },
            {
                "id": "L7001",
                "character1": "MARTY",
                "character2": "DOC",
                "text1": "This is heavy, Doc.",
                "text2": "Great Scott! You're absolutely right.",
                "movie": "back to the future",
                "genre": "sci-fi",
                "keywords": ["time", "realization", "science", "friendship"]
            },
            {
                "id": "L8001",
                "character1": "ROCKY",
                "character2": "ADRIAN",
                "text1": "I did it!",
                "text2": "I love you!",
                "movie": "rocky",
                "genre": "drama",
                "keywords": ["achievement", "love", "support", "victory"]
            }
        ]

    async def load(self) -> None:
        """Load Cornell movie dialogue corpus"""
        try:
            # In a real implementation, this would load from actual Cornell corpus files
            # For now, we'll use sample data
            self.dialogues = self.sample_dialogues
            self.loaded = True
            print(f"Cornell Database: Loaded {len(self.dialogues)} sample dialogues")

        except Exception as e:
            print(f"Error loading Cornell database: {e}")
            # Fallback to sample data
            self.dialogues = self.sample_dialogues
            self.loaded = True

    async def find_topic_dialogues(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Find dialogues related to specific topics/keywords"""
        if not self.loaded:
            await self.load()

        if not keywords:
            return random.sample(self.dialogues, min(limit, len(self.dialogues)))

        # Score dialogues based on keyword matches
        scored_dialogues = []
        for dialogue in self.dialogues:
            score = 0
            dialogue_text = f"{dialogue['text1']} {dialogue['text2']}".lower()
            dialogue_keywords = dialogue.get('keywords', [])

            # Check for keyword matches in text and metadata
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in dialogue_text:
                    score += 2
                if any(keyword_lower in dk.lower() for dk in dialogue_keywords):
                    score += 3

            if score > 0:
                scored_dialogues.append((score, dialogue))

        # Sort by score and return top results
        scored_dialogues.sort(key=lambda x: x[0], reverse=True)
        result = [dialogue for _, dialogue in scored_dialogues[:limit]]

        # If not enough scored results, add random ones to fill limit
        if len(result) < limit:
            remaining = [d for d in self.dialogues if d not in result]
            additional = random.sample(remaining, min(limit - len(result), len(remaining)))
            result.extend(additional)

        return result

    async def find_related_patterns(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Find conversation patterns related to keywords"""
        dialogues = await self.find_topic_dialogues(keywords, limit=10)

        patterns = []
        for dialogue in dialogues:
            pattern = {
                "pattern_type": "cornell_dialogue",
                "context": dialogue.get("genre", "general"),
                "initiator": dialogue["text1"],
                "response": dialogue["text2"],
                "movie": dialogue.get("movie", "unknown"),
                "keywords": dialogue.get("keywords", []),
                "character_dynamics": {
                    "speaker1": dialogue.get("character1", "unknown"),
                    "speaker2": dialogue.get("character2", "unknown")
                }
            }
            patterns.append(pattern)

        return patterns

    async def get_random_dialogue(self) -> Dict[str, Any]:
        """Get a random dialogue from the corpus"""
        if not self.loaded:
            await self.load()

        return random.choice(self.dialogues)

    async def get_dialogues_by_genre(self, genre: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get dialogues filtered by movie genre"""
        if not self.loaded:
            await self.load()

        genre_dialogues = [d for d in self.dialogues if d.get("genre", "").lower() == genre.lower()]
        return random.sample(genre_dialogues, min(limit, len(genre_dialogues)))

    async def search_dialogues(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search dialogues by text content"""
        if not self.loaded:
            await self.load()

        query_lower = query.lower()
        matching_dialogues = []

        for dialogue in self.dialogues:
            if (query_lower in dialogue["text1"].lower() or
                query_lower in dialogue["text2"].lower() or
                query_lower in dialogue.get("movie", "").lower()):
                matching_dialogues.append(dialogue)

        return matching_dialogues[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_dialogues": len(self.dialogues),
            "loaded": self.loaded,
            "genres": list(set(d.get("genre", "unknown") for d in self.dialogues)),
            "movies": list(set(d.get("movie", "unknown") for d in self.dialogues))
        }