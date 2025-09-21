"""
PersonaChat Database Module

Provides access to PersonaChat dataset for personality-based conversation generation.
This module implements personality profiles and conversation patterns.
"""

import json
import random
import asyncio
from typing import List, Dict, Any, Optional


class PersonaChatDatabase:
    """PersonaChat dataset database interface"""

    def __init__(self):
        self.personas = []
        self.conversations = []
        self.loaded = False

        # Sample PersonaChat data for development
        self.sample_data = {
            "personas": [
                {
                    "id": "persona_001",
                    "traits": [
                        "I love to cook and try new recipes",
                        "I have two cats named Luna and Oliver",
                        "I work as a graphic designer",
                        "I enjoy hiking on weekends",
                        "I'm learning to speak Spanish"
                    ],
                    "personality_type": "creative_adventurous",
                    "interests": ["cooking", "pets", "design", "nature", "languages"],
                    "communication_style": "friendly and enthusiastic"
                },
                {
                    "id": "persona_002",
                    "traits": [
                        "I'm a software engineer at a tech startup",
                        "I play video games in my free time",
                        "I collect vintage comic books",
                        "I live in a small apartment in the city",
                        "I prefer staying indoors"
                    ],
                    "personality_type": "introverted_tech_savvy",
                    "interests": ["technology", "gaming", "comics", "urban_life"],
                    "communication_style": "analytical and thoughtful"
                },
                {
                    "id": "persona_003",
                    "traits": [
                        "I'm a teacher at a local elementary school",
                        "I volunteer at the animal shelter",
                        "I enjoy reading mystery novels",
                        "I have a garden where I grow vegetables",
                        "I'm married with two children"
                    ],
                    "personality_type": "nurturing_community_oriented",
                    "interests": ["education", "animals", "reading", "gardening", "family"],
                    "communication_style": "patient and caring"
                },
                {
                    "id": "persona_004",
                    "traits": [
                        "I'm a fitness trainer and nutritionist",
                        "I wake up at 5 AM every day",
                        "I compete in marathon races",
                        "I meal prep every Sunday",
                        "I don't drink alcohol or caffeine"
                    ],
                    "personality_type": "disciplined_health_focused",
                    "interests": ["fitness", "nutrition", "running", "healthy_living"],
                    "communication_style": "motivational and direct"
                },
                {
                    "id": "persona_005",
                    "traits": [
                        "I'm a musician who plays in a jazz band",
                        "I travel frequently for performances",
                        "I collect vintage vinyl records",
                        "I love trying different types of coffee",
                        "I write my own songs"
                    ],
                    "personality_type": "artistic_free_spirited",
                    "interests": ["music", "travel", "vintage_items", "coffee", "creativity"],
                    "communication_style": "expressive and passionate"
                }
            ],
            "conversation_templates": [
                {
                    "topic": "hobbies",
                    "openers": [
                        "What do you like to do in your free time?",
                        "I've been really into {hobby} lately. Do you have any hobbies?",
                        "I'm looking for a new hobby to try. Any suggestions?",
                        "Tell me about something you're passionate about."
                    ],
                    "responses": [
                        "That sounds really interesting! How did you get started with that?",
                        "I've always wanted to try that! Is it hard to learn?",
                        "That's so cool! I can see why you'd enjoy that.",
                        "I'm the opposite - I'm more into {contrasting_activity}."
                    ]
                },
                {
                    "topic": "work",
                    "openers": [
                        "What do you do for work?",
                        "How's your job going?",
                        "I had such a crazy day at work today.",
                        "Do you enjoy what you do for a living?"
                    ],
                    "responses": [
                        "That must be really {adjective}! What's the best part about it?",
                        "I bet that's challenging. How do you handle the pressure?",
                        "That's so different from what I do!",
                        "I've always been curious about that field."
                    ]
                },
                {
                    "topic": "food",
                    "openers": [
                        "What's your favorite type of food?",
                        "I just tried this amazing restaurant.",
                        "I'm trying to cook more at home lately.",
                        "Have you ever tried {cuisine} food?"
                    ],
                    "responses": [
                        "That sounds delicious! What makes it so good?",
                        "I love {food_type} too! Do you have a favorite dish?",
                        "I'm not much of a cook, but that sounds impressive.",
                        "I'll have to try that sometime!"
                    ]
                }
            ]
        }

    async def load(self) -> None:
        """Load PersonaChat dataset"""
        try:
            # In a real implementation, this would load from actual PersonaChat files
            # For now, we'll use sample data
            self.personas = self.sample_data["personas"]
            self.conversations = self.sample_data["conversation_templates"]
            self.loaded = True
            print(f"PersonaChat Database: Loaded {len(self.personas)} personas and {len(self.conversations)} conversation templates")

        except Exception as e:
            print(f"Error loading PersonaChat database: {e}")
            # Fallback to sample data
            self.personas = self.sample_data["personas"]
            self.conversations = self.sample_data["conversation_templates"]
            self.loaded = True

    async def get_casual_opener(self, topic: Dict[str, Any]) -> str:
        """Get a casual conversation opener for a given topic"""
        if not self.loaded:
            await self.load()

        topic_name = topic.get("name", "general")
        keywords = topic.get("keywords", [])

        # Try to find a matching conversation template
        matching_template = None
        for template in self.conversations:
            if template["topic"] in topic_name.lower() or any(kw in template["topic"] for kw in keywords):
                matching_template = template
                break

        if matching_template:
            opener = random.choice(matching_template["openers"])
            # Simple template substitution
            if "{hobby}" in opener and keywords:
                opener = opener.replace("{hobby}", random.choice(keywords))
            return opener
        else:
            # Generic openers
            generic_openers = [
                f"I've been thinking about {topic_name} lately. What's your take on it?",
                f"Have you had any experience with {topic_name}?",
                f"I'm curious about your thoughts on {topic_name}.",
                f"What comes to mind when you think about {topic_name}?"
            ]
            return random.choice(generic_openers)

    async def find_topic_dialogues(self, topic: Dict[str, Any], user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find persona-based dialogues for a topic"""
        if not self.loaded:
            await self.load()

        topic_name = topic.get("name", "general")
        keywords = topic.get("keywords", [])

        # Match user profile to closest persona
        matched_persona = await self.match_persona(user_profile)

        # Generate contextual dialogues based on persona and topic
        dialogues = []

        # Find relevant conversation template
        matching_template = None
        for template in self.conversations:
            if template["topic"] in topic_name.lower() or any(kw in template["topic"] for kw in keywords):
                matching_template = template
                break

        if matching_template:
            # Generate dialogues using persona traits and conversation template
            for i in range(3):  # Generate 3 sample dialogues
                opener = random.choice(matching_template["openers"])
                response = random.choice(matching_template["responses"])

                # Customize based on persona
                if matched_persona:
                    # Add personality-specific touches
                    if matched_persona["communication_style"] == "friendly and enthusiastic":
                        response = f"Oh wow! {response}"
                    elif matched_persona["communication_style"] == "analytical and thoughtful":
                        response = f"Interesting. {response}"
                    elif matched_persona["communication_style"] == "patient and caring":
                        response = f"That's wonderful. {response}"

                dialogue = {
                    "id": f"persona_dialogue_{topic_name}_{i}",
                    "topic": topic_name,
                    "persona_id": matched_persona["id"] if matched_persona else "generic",
                    "opener": opener,
                    "response": response,
                    "context": matched_persona["personality_type"] if matched_persona else "generic",
                    "relevance_score": random.uniform(0.7, 1.0)
                }
                dialogues.append(dialogue)

        return dialogues

    async def match_persona(self, user_profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Match user profile to closest persona"""
        if not self.loaded:
            await self.load()

        if not user_profile:
            return random.choice(self.personas)

        user_interests = user_profile.get("interests", [])
        user_traits = user_profile.get("traits", [])

        # Score personas based on interest overlap
        persona_scores = []
        for persona in self.personas:
            score = 0
            persona_interests = persona.get("interests", [])
            persona_traits = persona.get("traits", [])

            # Calculate interest overlap
            for interest in user_interests:
                if interest.lower() in [pi.lower() for pi in persona_interests]:
                    score += 2

            # Calculate trait similarity (simple keyword matching)
            for user_trait in user_traits:
                for persona_trait in persona_traits:
                    if any(word in persona_trait.lower() for word in user_trait.lower().split()):
                        score += 1

            persona_scores.append((score, persona))

        # Return highest scoring persona, or random if no matches
        if persona_scores:
            persona_scores.sort(key=lambda x: x[0], reverse=True)
            return persona_scores[0][1]
        else:
            return random.choice(self.personas)

    async def get_persona_by_id(self, persona_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific persona by ID"""
        if not self.loaded:
            await self.load()

        for persona in self.personas:
            if persona["id"] == persona_id:
                return persona
        return None

    async def get_random_persona(self) -> Dict[str, Any]:
        """Get a random persona"""
        if not self.loaded:
            await self.load()

        return random.choice(self.personas)

    async def get_conversation_starters(self, personality_type: str) -> List[str]:
        """Get conversation starters for a specific personality type"""
        if not self.loaded:
            await self.load()

        starters = []

        # Find personas with matching personality type
        matching_personas = [p for p in self.personas if p.get("personality_type") == personality_type]

        if matching_personas:
            persona = random.choice(matching_personas)
            interests = persona.get("interests", [])

            # Generate starters based on interests
            for interest in interests[:3]:  # Top 3 interests
                starters.append(f"I'm really into {interest}. What about you?")
                starters.append(f"Have you ever tried anything related to {interest}?")

        # Add generic starters if not enough specific ones
        generic_starters = [
            "What's been the highlight of your week?",
            "I'm always looking for new things to try. Any recommendations?",
            "What's something you're passionate about?",
            "Tell me about something that made you smile recently."
        ]

        starters.extend(generic_starters)
        return starters[:5]  # Return top 5

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            "total_personas": len(self.personas),
            "conversation_templates": len(self.conversations),
            "loaded": self.loaded,
            "personality_types": list(set(p.get("personality_type", "unknown") for p in self.personas))
        }