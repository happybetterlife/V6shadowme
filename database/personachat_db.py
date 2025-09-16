"""
PersonaChat Database
페르소나 기반 대화 데이터베이스
"""

import sqlite3
import json
import asyncio
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import random
from datetime import datetime

logger = logging.getLogger(__name__)

class PersonaChatDatabase:
    def __init__(self, db_path: str = "database/personachat.db"):
        self.db_path = db_path
        self.conn = None
        
    async def initialize(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        await self.create_tables()
        
        # 데이터가 없으면 샘플 데이터 삽입
        if not await self.has_data():
            await self.insert_sample_data()
        
        logger.info("PersonaChat Database initialized")
    
    async def create_tables(self):
        """테이블 생성"""
        cursor = self.conn.cursor()
        
        # 페르소나 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age_group TEXT,
                personality_traits TEXT,
                interests TEXT,
                background TEXT,
                speaking_style TEXT,
                vocabulary_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 대화 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dialogues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona1_id INTEGER,
                persona2_id INTEGER,
                topic TEXT,
                context TEXT,
                difficulty TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (persona1_id) REFERENCES personas (id),
                FOREIGN KEY (persona2_id) REFERENCES personas (id)
            )
        """)
        
        # 대화 턴 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dialogue_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dialogue_id INTEGER,
                speaker_persona_id INTEGER,
                text TEXT NOT NULL,
                turn_number INTEGER,
                emotion TEXT,
                intent TEXT,
                FOREIGN KEY (dialogue_id) REFERENCES dialogues (id),
                FOREIGN KEY (speaker_persona_id) REFERENCES personas (id)
            )
        """)
        
        # 페르소나 매칭 규칙 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persona_matching_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_level TEXT,
                user_interests TEXT,
                recommended_persona_traits TEXT,
                speaking_style TEXT,
                priority INTEGER DEFAULT 0
            )
        """)
        
        # 학습 스타일 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_styles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                style_name TEXT NOT NULL,
                description TEXT,
                persona_traits TEXT,
                teaching_approach TEXT,
                error_correction_style TEXT,
                encouragement_level TEXT
            )
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_personas_traits ON personas (personality_traits)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_dialogues_topic ON dialogues (topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_turns_text ON dialogue_turns (text)")
        
        self.conn.commit()
    
    async def has_data(self) -> bool:
        """데이터 존재 여부 확인"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM personas")
        count = cursor.fetchone()[0]
        return count > 0
    
    async def insert_sample_data(self):
        """샘플 데이터 삽입"""
        cursor = self.conn.cursor()
        
        # 샘플 페르소나
        personas = [
            # 친근한 교사
            ("Friendly Teacher", "30-40", "patient,encouraging,supportive", 
             "education,languages,culture", "English teacher with 10 years experience",
             "clear,simple,encouraging", "adaptive", datetime.now()),
            
            # 비즈니스 전문가
            ("Business Professional", "35-45", "professional,direct,efficient",
             "business,technology,economics", "MBA graduate, tech company executive",
             "formal,precise,technical", "advanced", datetime.now()),
            
            # 젊은 학생
            ("Young Student", "18-25", "curious,energetic,casual",
             "social media,gaming,music", "College student studying computer science",
             "casual,modern,slang", "intermediate", datetime.now()),
            
            # 여행 블로거
            ("Travel Blogger", "25-35", "adventurous,creative,descriptive",
             "travel,food,photography", "Travel blogger visiting 50+ countries",
             "descriptive,storytelling,enthusiastic", "intermediate", datetime.now()),
            
            # 과학자
            ("Research Scientist", "30-50", "analytical,precise,curious",
             "science,research,innovation", "PhD in Biology, research at university",
             "technical,analytical,explanatory", "advanced", datetime.now()),
            
            # 예술가
            ("Creative Artist", "25-40", "creative,expressive,philosophical",
             "art,music,philosophy", "Freelance artist and musician",
             "metaphorical,expressive,thoughtful", "intermediate", datetime.now())
        ]
        
        cursor.executemany(
            """INSERT INTO personas 
               (name, age_group, personality_traits, interests, background, 
                speaking_style, vocabulary_level, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            personas
        )
        
        # 샘플 대화
        dialogues = [
            (1, 3, "Technology and Social Media", "Discussing impact of social media", "intermediate"),
            (2, 5, "Innovation in Business", "Talking about AI in business", "advanced"),
            (4, 6, "Travel and Art", "Discussing art from different cultures", "intermediate"),
            (1, 2, "Business English", "Learning business vocabulary", "intermediate"),
            (3, 4, "Student Travel", "Planning a backpacking trip", "beginner")
        ]
        
        for dialogue in dialogues:
            cursor.execute(
                """INSERT INTO dialogues 
                   (persona1_id, persona2_id, topic, context, difficulty)
                   VALUES (?, ?, ?, ?, ?)""",
                dialogue
            )
            dialogue_id = cursor.lastrowid
            
            # 각 대화에 대한 샘플 턴 추가
            self.add_sample_turns(cursor, dialogue_id, dialogue[0], dialogue[1])
        
        # 페르소나 매칭 규칙
        matching_rules = [
            ("beginner", "general", "patient,encouraging", "simple,clear", 5),
            ("intermediate", "business", "professional,supportive", "formal,clear", 4),
            ("advanced", "technology", "analytical,challenging", "technical,precise", 4),
            ("beginner", "travel", "friendly,descriptive", "simple,visual", 3),
            ("intermediate", "culture", "expressive,informative", "descriptive,cultural", 3)
        ]
        
        cursor.executemany(
            """INSERT INTO persona_matching_rules 
               (user_level, user_interests, recommended_persona_traits, speaking_style, priority)
               VALUES (?, ?, ?, ?, ?)""",
            matching_rules
        )
        
        # 학습 스타일
        learning_styles = [
            ("Conversational", "Natural conversation flow", "friendly,patient", 
             "Learn through dialogue", "Gentle indirect correction", "High"),
            
            ("Academic", "Structured learning approach", "professional,precise",
             "Systematic teaching", "Direct correction with explanation", "Moderate"),
            
            ("Immersive", "Full immersion method", "native-like,natural",
             "Sink or swim approach", "Minimal correction", "Low"),
            
            ("Gamified", "Game-based learning", "playful,energetic",
             "Learn through games and challenges", "Points-based feedback", "Very High"),
            
            ("Practical", "Real-world application", "practical,direct",
             "Focus on useful phrases", "Correction only for major errors", "Moderate")
        ]
        
        cursor.executemany(
            """INSERT INTO learning_styles 
               (style_name, description, persona_traits, teaching_approach, 
                error_correction_style, encouragement_level)
               VALUES (?, ?, ?, ?, ?, ?)""",
            learning_styles
        )
        
        self.conn.commit()
        logger.info("Sample PersonaChat data inserted")
    
    def add_sample_turns(
        self,
        cursor,
        dialogue_id: int,
        persona1_id: int,
        persona2_id: int
    ):
        """대화 턴 샘플 추가"""
        sample_turns = {
            "Technology and Social Media": [
                (persona1_id, "How do you think social media affects learning?", 1, "curious", "question"),
                (persona2_id, "It's like, totally changed everything! I can learn anything on YouTube now.", 2, "excited", "opinion"),
                (persona1_id, "That's true! But do you think it helps with focus?", 3, "thoughtful", "follow-up"),
                (persona2_id, "Honestly? Sometimes it's super distracting. Too many notifications!", 4, "honest", "admission")
            ],
            "Innovation in Business": [
                (persona1_id, "AI implementation requires careful strategic planning.", 1, "professional", "statement"),
                (persona2_id, "Indeed. The key is identifying processes that benefit most from automation.", 2, "analytical", "agreement"),
                (persona1_id, "What's your view on the ROI timeline?", 3, "inquisitive", "question"),
                (persona2_id, "Typically 18-24 months for substantial returns, depending on scale.", 4, "informative", "answer")
            ]
        }
        
        # 기본 턴 생성
        default_turns = [
            (persona1_id, "Hello! How are you today?", 1, "friendly", "greeting"),
            (persona2_id, "I'm doing well, thank you! How about you?", 2, "polite", "response"),
            (persona1_id, "Great! Shall we discuss our topic?", 3, "enthusiastic", "suggestion"),
            (persona2_id, "Yes, I'd love to hear your thoughts.", 4, "interested", "agreement")
        ]
        
        turns = default_turns
        
        for turn in turns:
            cursor.execute(
                """INSERT INTO dialogue_turns 
                   (dialogue_id, speaker_persona_id, text, turn_number, emotion, intent)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (dialogue_id, turn[0], turn[1], turn[2], turn[3], turn[4])
            )
    
    async def match_persona(self, user_profile: Dict) -> Dict:
        """사용자 프로필에 맞는 페르소나 매칭"""
        cursor = self.conn.cursor()
        
        user_level = user_profile.get('level', 'intermediate')
        user_interests = user_profile.get('interests', [])
        learning_style = user_profile.get('learning_style', 'Conversational')
        
        # 매칭 규칙 기반 페르소나 찾기
        query = """
            SELECT p.*, ls.teaching_approach, ls.error_correction_style
            FROM personas p
            LEFT JOIN learning_styles ls ON ls.persona_traits LIKE '%' || p.personality_traits || '%'
            WHERE p.vocabulary_level = ? OR p.vocabulary_level = 'adaptive'
            ORDER BY RANDOM()
            LIMIT 1
        """
        
        cursor.execute(query, (user_level,))
        row = cursor.fetchone()
        
        if row:
            return {
                'persona_id': row[0],
                'name': row[1],
                'traits': row[3].split(','),
                'interests': row[4].split(','),
                'background': row[5],
                'speaking_style': row[6],
                'vocabulary_level': row[7],
                'teaching_approach': row[9] if len(row) > 9 else "Natural conversation",
                'error_correction_style': row[10] if len(row) > 10 else "Gentle"
            }
        
        # 기본 페르소나 반환
        return self.get_default_persona(user_level)
    
    async def get_dialogue_examples(
        self,
        topic: str,
        difficulty: str,
        limit: int = 5
    ) -> List[Dict]:
        """토픽과 난이도에 맞는 대화 예시 가져오기"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT d.id, d.topic, d.context, dt.text, dt.emotion, dt.intent,
                   p.name, p.speaking_style
            FROM dialogues d
            JOIN dialogue_turns dt ON dt.dialogue_id = d.id
            JOIN personas p ON dt.speaker_persona_id = p.id
            WHERE d.topic LIKE ? AND d.difficulty = ?
            ORDER BY d.id, dt.turn_number
            LIMIT ?
        """
        
        cursor.execute(query, (f"%{topic}%", difficulty, limit * 4))
        
        dialogues = {}
        for row in cursor.fetchall():
            dialogue_id = row[0]
            if dialogue_id not in dialogues:
                dialogues[dialogue_id] = {
                    'id': dialogue_id,
                    'topic': row[1],
                    'context': row[2],
                    'turns': []
                }
            
            dialogues[dialogue_id]['turns'].append({
                'text': row[3],
                'emotion': row[4],
                'intent': row[5],
                'speaker': row[6],
                'style': row[7]
            })
        
        return list(dialogues.values())[:limit]
    
    async def get_persona_dialogue_style(
        self,
        persona_id: int
    ) -> Dict:
        """페르소나의 대화 스타일 가져오기"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT personality_traits, speaking_style, vocabulary_level, interests
            FROM personas
            WHERE id = ?
        """
        
        cursor.execute(query, (persona_id,))
        row = cursor.fetchone()
        
        if row:
            return {
                'traits': row[0].split(','),
                'speaking_style': row[1],
                'vocabulary_level': row[2],
                'interests': row[3].split(',')
            }
        
        return self.get_default_style()
    
    async def save_user_dialogue(
        self,
        user_id: str,
        persona_id: int,
        dialogue_turns: List[Tuple[str, str]],
        topic: str
    ):
        """사용자 대화 저장"""
        cursor = self.conn.cursor()
        
        # 새 대화 생성
        cursor.execute(
            """INSERT INTO dialogues (persona1_id, persona2_id, topic, context, difficulty)
               VALUES (?, ?, ?, ?, ?)""",
            (persona_id, None, topic, f"User {user_id} conversation", "user_generated")
        )
        
        dialogue_id = cursor.lastrowid
        
        # 턴 저장
        for i, (user_text, ai_text) in enumerate(dialogue_turns):
            # 사용자 턴
            cursor.execute(
                """INSERT INTO dialogue_turns 
                   (dialogue_id, speaker_persona_id, text, turn_number, emotion, intent)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (dialogue_id, None, user_text, i*2+1, "neutral", "user_input")
            )
            
            # AI 턴
            cursor.execute(
                """INSERT INTO dialogue_turns 
                   (dialogue_id, speaker_persona_id, text, turn_number, emotion, intent)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (dialogue_id, persona_id, ai_text, i*2+2, "helpful", "response")
            )
        
        self.conn.commit()
    
    async def get_learning_style_recommendations(
        self,
        user_profile: Dict
    ) -> List[Dict]:
        """학습 스타일 추천"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT style_name, description, teaching_approach, 
                   error_correction_style, encouragement_level
            FROM learning_styles
            ORDER BY 
                CASE 
                    WHEN style_name = ? THEN 0
                    ELSE 1
                END,
                RANDOM()
            LIMIT 3
        """
        
        preferred_style = user_profile.get('learning_style', 'Conversational')
        cursor.execute(query, (preferred_style,))
        
        recommendations = []
        for row in cursor.fetchall():
            recommendations.append({
                'style': row[0],
                'description': row[1],
                'teaching_approach': row[2],
                'correction_style': row[3],
                'encouragement': row[4]
            })
        
        return recommendations
    
    def get_default_persona(self, level: str) -> Dict:
        """기본 페르소나 반환"""
        return {
            'persona_id': 0,
            'name': 'Default Teacher',
            'traits': ['patient', 'encouraging', 'adaptive'],
            'interests': ['general', 'education', 'culture'],
            'background': 'Experienced English teacher',
            'speaking_style': 'clear' if level == 'beginner' else 'natural',
            'vocabulary_level': level,
            'teaching_approach': 'Conversational',
            'error_correction_style': 'Gentle'
        }
    
    def get_default_style(self) -> Dict:
        """기본 대화 스타일 반환"""
        return {
            'traits': ['friendly', 'helpful'],
            'speaking_style': 'natural',
            'vocabulary_level': 'intermediate',
            'interests': ['general']
        }
    
    async def analyze_conversation_patterns(
        self,
        user_id: str
    ) -> Dict:
        """사용자 대화 패턴 분석"""
        cursor = self.conn.cursor()
        
        # 사용자의 대화 통계
        query = """
            SELECT COUNT(*) as total_turns,
                   AVG(LENGTH(text)) as avg_length,
                   COUNT(DISTINCT dialogue_id) as total_dialogues
            FROM dialogue_turns
            WHERE dialogue_id IN (
                SELECT id FROM dialogues 
                WHERE context LIKE ?
            )
        """
        
        cursor.execute(query, (f"%User {user_id}%",))
        row = cursor.fetchone()
        
        if row:
            return {
                'total_turns': row[0],
                'average_message_length': row[1],
                'total_conversations': row[2],
                'patterns': await self.identify_patterns(user_id)
            }
        
        return {
            'total_turns': 0,
            'average_message_length': 0,
            'total_conversations': 0,
            'patterns': []
        }
    
    async def identify_patterns(self, user_id: str) -> List[str]:
        """대화 패턴 식별"""
        # 실제로는 더 복잡한 패턴 분석 로직
        patterns = [
            "Prefers asking questions",
            "Uses casual language",
            "Responds with short sentences"
        ]
        return random.sample(patterns, 2)
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()