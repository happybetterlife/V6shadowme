"""
Google Trends Database
트렌딩 토픽 및 실시간 이슈 관리
"""

import sqlite3
import json
import asyncio
from typing import List, Dict, Optional
import logging
from pathlib import Path
from datetime import datetime, timedelta
import random
import aiohttp

logger = logging.getLogger(__name__)

class TrendsDatabase:
    def __init__(self, db_path: str = "database/trends.db"):
        self.db_path = db_path
        self.conn = None
        self.cache_duration = timedelta(hours=6)  # 6시간 캐시
        
    async def initialize(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        await self.create_tables()
        
        # 초기 트렌드 데이터 삽입
        if not await self.has_data():
            await self.insert_initial_trends()
        
        logger.info("Trends Database initialized")
    
    async def create_tables(self):
        """테이블 생성"""
        cursor = self.conn.cursor()
        
        # 트렌드 토픽 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trending_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT,
                keywords TEXT,
                difficulty_level TEXT,
                region TEXT DEFAULT 'global',
                relevance_score REAL,
                search_volume INTEGER,
                trend_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP
            )
        """)
        
        # 토픽 상세 정보 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                summary TEXT,
                vocabulary_words TEXT,
                discussion_points TEXT,
                difficulty_analysis TEXT,
                educational_value REAL,
                FOREIGN KEY (topic_id) REFERENCES trending_topics (id)
            )
        """)
        
        # 관련 뉴스/기사 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS related_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic_id INTEGER,
                title TEXT,
                source TEXT,
                url TEXT,
                snippet TEXT,
                published_date TIMESTAMP,
                language TEXT DEFAULT 'en',
                FOREIGN KEY (topic_id) REFERENCES trending_topics (id)
            )
        """)
        
        # 카테고리별 토픽 템플릿
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                template_name TEXT,
                starter_questions TEXT,
                key_vocabulary TEXT,
                discussion_structure TEXT,
                level_adaptations TEXT
            )
        """)
        
        # 사용자 토픽 선호도
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_topic_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                topic_id INTEGER,
                interest_level INTEGER,
                completed BOOLEAN DEFAULT FALSE,
                feedback TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (topic_id) REFERENCES trending_topics (id)
            )
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trends_date ON trending_topics (trend_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trends_category ON trending_topics (category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_prefs ON user_topic_preferences (user_id)")
        
        self.conn.commit()
    
    async def has_data(self) -> bool:
        """데이터 존재 여부 확인"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trending_topics")
        count = cursor.fetchone()[0]
        return count > 0
    
    async def insert_initial_trends(self):
        """초기 트렌드 데이터 삽입"""
        cursor = self.conn.cursor()
        
        # 샘플 트렌딩 토픽
        today = datetime.now().date()
        expires = datetime.now() + self.cache_duration
        
        trending_topics = [
            # Technology
            ("AI in Healthcare", "technology", "artificial intelligence,healthcare,medical,diagnosis", 
             "intermediate", "global", 0.9, 150000, today, expires),
            
            ("Space Tourism", "technology", "space,tourism,commercial,spacecraft", 
             "advanced", "global", 0.85, 120000, today, expires),
            
            # Entertainment
            ("Oscar Nominations", "entertainment", "oscars,movies,awards,cinema", 
             "beginner", "global", 0.95, 200000, today, expires),
            
            ("K-Pop Global Impact", "entertainment", "kpop,music,culture,korean", 
             "intermediate", "global", 0.88, 180000, today, expires),
            
            # Sports
            ("World Cup Preparations", "sports", "soccer,football,world cup,teams", 
             "beginner", "global", 0.92, 250000, today, expires),
            
            # Business
            ("Sustainable Business", "business", "sustainability,green,eco-friendly,corporate", 
             "advanced", "global", 0.82, 90000, today, expires),
            
            # Lifestyle
            ("Digital Detox", "lifestyle", "wellness,mental health,technology,balance", 
             "intermediate", "global", 0.78, 75000, today, expires),
            
            ("Plant-Based Diet", "lifestyle", "vegan,vegetarian,health,nutrition", 
             "beginner", "global", 0.80, 110000, today, expires),
            
            # Current Events
            ("Climate Summit", "current_events", "climate,environment,global warming,policy", 
             "advanced", "global", 0.87, 130000, today, expires),
            
            ("Remote Work Future", "business", "remote,work from home,hybrid,office", 
             "intermediate", "global", 0.84, 95000, today, expires)
        ]
        
        for topic in trending_topics:
            cursor.execute(
                """INSERT INTO trending_topics 
                   (title, category, keywords, difficulty_level, region, 
                    relevance_score, search_volume, trend_date, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                topic
            )
            
            topic_id = cursor.lastrowid
            await self.add_topic_details(cursor, topic_id, topic[0], topic[2], topic[3])
        
        # 토픽 템플릿 추가
        await self.insert_topic_templates(cursor)
        
        self.conn.commit()
        logger.info("Initial trending topics inserted")
    
    async def add_topic_details(
        self,
        cursor,
        topic_id: int,
        title: str,
        keywords: str,
        difficulty: str
    ):
        """토픽 상세 정보 추가"""
        # 토픽별 어휘 및 토론 포인트
        topic_data = {
            "AI in Healthcare": {
                "summary": "Exploring how artificial intelligence is revolutionizing medical diagnosis and treatment",
                "vocabulary": ["diagnosis", "algorithm", "precision medicine", "data analysis", "automation"],
                "discussion_points": [
                    "Benefits of AI in early disease detection",
                    "Privacy concerns with medical data",
                    "Will AI replace doctors?",
                    "Cost and accessibility of AI healthcare"
                ]
            },
            "Space Tourism": {
                "summary": "Commercial space travel is becoming a reality for civilians",
                "vocabulary": ["spacecraft", "orbit", "astronaut", "zero gravity", "launch"],
                "discussion_points": [
                    "Environmental impact of space tourism",
                    "Safety considerations for civilians",
                    "Cost and accessibility",
                    "Future of space exploration"
                ]
            },
            "Digital Detox": {
                "summary": "Taking breaks from technology to improve mental health and well-being",
                "vocabulary": ["mindfulness", "screen time", "balance", "wellness", "disconnect"],
                "discussion_points": [
                    "Signs you need a digital detox",
                    "Benefits of unplugging",
                    "Strategies for reducing screen time",
                    "Impact on productivity and relationships"
                ]
            }
        }
        
        # 기본 데이터
        default_data = {
            "summary": f"Current trending topic about {title}",
            "vocabulary": keywords.split(',')[:5],
            "discussion_points": [
                f"What do you know about {title}?",
                f"How does {title} affect daily life?",
                f"Future implications of {title}",
                "Personal experiences or opinions"
            ]
        }
        
        data = topic_data.get(title, default_data)
        
        # 난이도별 교육 가치 설정
        educational_values = {
            "beginner": 0.7,
            "intermediate": 0.85,
            "advanced": 0.9
        }
        
        cursor.execute(
            """INSERT INTO topic_details 
               (topic_id, summary, vocabulary_words, discussion_points, 
                difficulty_analysis, educational_value)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (topic_id, 
             data["summary"],
             json.dumps(data["vocabulary"]),
             json.dumps(data["discussion_points"]),
             f"Suitable for {difficulty} learners",
             educational_values.get(difficulty, 0.8))
        )
    
    async def insert_topic_templates(self, cursor):
        """토픽 템플릿 삽입"""
        templates = [
            ("technology", "Tech Discussion", 
             json.dumps([
                 "What technology do you use daily?",
                 "How has technology changed your life?",
                 "What's your opinion on this innovation?"
             ]),
             json.dumps(["innovation", "device", "application", "digital", "smart"]),
             json.dumps({
                 "intro": "Brief explanation",
                 "main": "Pros and cons discussion",
                 "conclusion": "Future predictions"
             }),
             json.dumps({
                 "beginner": "Focus on basic vocabulary and personal experience",
                 "intermediate": "Include technical terms and broader implications",
                 "advanced": "Deep dive into ethical and societal impacts"
             })),
            
            ("entertainment", "Entertainment Talk",
             json.dumps([
                 "What's your favorite type of entertainment?",
                 "How do you usually spend your free time?",
                 "What makes good entertainment?"
             ]),
             json.dumps(["audience", "performance", "creative", "popular", "genre"]),
             json.dumps({
                 "intro": "Personal preferences",
                 "main": "Cultural differences",
                 "conclusion": "Recommendations"
             }),
             json.dumps({
                 "beginner": "Simple preferences and descriptions",
                 "intermediate": "Comparisons and reviews",
                 "advanced": "Cultural analysis and criticism"
             })),
            
            ("current_events", "News Discussion",
             json.dumps([
                 "Have you heard about this news?",
                 "What's your opinion on this issue?",
                 "How does this affect your country?"
             ]),
             json.dumps(["impact", "global", "policy", "consequence", "perspective"]),
             json.dumps({
                 "intro": "Summary of events",
                 "main": "Different viewpoints",
                 "conclusion": "Personal stance"
             }),
             json.dumps({
                 "beginner": "Basic facts and personal reactions",
                 "intermediate": "Cause and effect discussion",
                 "advanced": "Complex analysis and debate"
             }))
        ]
        
        cursor.executemany(
            """INSERT INTO topic_templates 
               (category, template_name, starter_questions, key_vocabulary, 
                discussion_structure, level_adaptations)
               VALUES (?, ?, ?, ?, ?, ?)""",
            templates
        )
    
    async def get_cached_trends(self) -> List[str]:
        """캐시된 트렌드 가져오기"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT title 
            FROM trending_topics
            WHERE expires_at > ? AND trend_date >= ?
            ORDER BY relevance_score DESC, search_volume DESC
            LIMIT 10
        """
        
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        
        cursor.execute(query, (now, week_ago.date()))
        
        trends = [row[0] for row in cursor.fetchall()]
        
        # 캐시가 비어있으면 기본 트렌드 반환
        if not trends:
            trends = await self.get_default_trends()
        
        return trends
    
    async def get_default_trends(self) -> List[str]:
        """기본 트렌드 목록"""
        return [
            "Artificial Intelligence",
            "Climate Change",
            "Remote Work",
            "Mental Health",
            "Sustainable Living",
            "Space Exploration",
            "Online Learning",
            "Cryptocurrency",
            "Social Media Impact",
            "Future of Transportation"
        ]
    
    async def save_topic(self, topic_details: Dict):
        """새 토픽 저장"""
        cursor = self.conn.cursor()
        
        expires = datetime.now() + self.cache_duration
        
        cursor.execute(
            """INSERT INTO trending_topics 
               (title, category, keywords, difficulty_level, region, 
                relevance_score, search_volume, trend_date, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (topic_details['title'],
             topic_details.get('category', 'general'),
             ','.join(topic_details.get('keywords', [])),
             topic_details.get('difficulty', 'intermediate'),
             'global',
             topic_details.get('relevance_score', 0.5),
             random.randint(50000, 200000),
             datetime.now().date(),
             expires)
        )
        
        self.conn.commit()
    
    async def get_topic_by_category(
        self,
        category: str,
        difficulty: Optional[str] = None
    ) -> List[Dict]:
        """카테고리별 토픽 가져오기"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT t.*, td.summary, td.vocabulary_words, td.discussion_points
            FROM trending_topics t
            LEFT JOIN topic_details td ON t.id = td.topic_id
            WHERE t.category = ? AND t.expires_at > ?
        """
        
        params = [category, datetime.now()]
        
        if difficulty:
            query += " AND t.difficulty_level = ?"
            params.append(difficulty)
        
        query += " ORDER BY t.relevance_score DESC LIMIT 5"
        
        cursor.execute(query, params)
        
        topics = []
        for row in cursor.fetchall():
            topics.append({
                'id': row[0],
                'title': row[1],
                'category': row[2],
                'keywords': row[3].split(',') if row[3] else [],
                'difficulty': row[4],
                'relevance_score': row[6],
                'summary': row[11] if len(row) > 11 else None,
                'vocabulary': json.loads(row[12]) if len(row) > 12 and row[12] else [],
                'discussion_points': json.loads(row[13]) if len(row) > 13 and row[13] else []
            })
        
        return topics
    
    async def get_personalized_topics(
        self,
        user_id: str,
        user_profile: Dict
    ) -> List[Dict]:
        """개인화된 토픽 추천"""
        cursor = self.conn.cursor()
        
        # 사용자 관심사 기반 필터링
        interests = user_profile.get('interests', [])
        level = user_profile.get('level', 'intermediate')
        
        # 이전에 완료한 토픽 제외
        query = """
            SELECT t.*, td.summary, td.vocabulary_words, td.discussion_points
            FROM trending_topics t
            LEFT JOIN topic_details td ON t.id = td.topic_id
            LEFT JOIN user_topic_preferences utp ON t.id = utp.topic_id AND utp.user_id = ?
            WHERE t.expires_at > ? 
                AND t.difficulty_level = ?
                AND (utp.completed IS NULL OR utp.completed = FALSE)
            ORDER BY t.relevance_score DESC
            LIMIT 10
        """
        
        cursor.execute(query, (user_id, datetime.now(), level))
        
        all_topics = []
        for row in cursor.fetchall():
            topic = {
                'id': row[0],
                'title': row[1],
                'category': row[2],
                'keywords': row[3].split(',') if row[3] else [],
                'difficulty': row[4],
                'relevance_score': row[6]
            }
            
            # 관심사와 매칭 점수 계산
            match_score = self.calculate_interest_match(topic, interests)
            topic['personal_relevance'] = match_score
            
            all_topics.append(topic)
        
        # 개인 관련성 기준으로 정렬
        all_topics.sort(key=lambda x: x['personal_relevance'], reverse=True)
        
        return all_topics[:5]
    
    def calculate_interest_match(self, topic: Dict, interests: List[str]) -> float:
        """관심사 매칭 점수 계산"""
        if not interests:
            return topic['relevance_score']
        
        score = 0.0
        topic_text = f"{topic['title']} {topic['category']} {' '.join(topic['keywords'])}".lower()
        
        for interest in interests:
            if interest.lower() in topic_text:
                score += 1.0
            elif any(word in topic_text for word in interest.lower().split()):
                score += 0.5
        
        # 기본 관련성 점수와 결합
        combined_score = (score / len(interests)) * 0.7 + topic['relevance_score'] * 0.3
        
        return min(combined_score, 1.0)
    
    async def record_topic_interaction(
        self,
        user_id: str,
        topic_id: int,
        interest_level: int,
        completed: bool = False,
        feedback: Optional[str] = None
    ):
        """사용자 토픽 상호작용 기록"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """INSERT OR REPLACE INTO user_topic_preferences 
               (user_id, topic_id, interest_level, completed, feedback, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, topic_id, interest_level, completed, feedback, datetime.now())
        )
        
        self.conn.commit()
    
    async def get_discussion_template(
        self,
        category: str,
        level: str
    ) -> Dict:
        """토론 템플릿 가져오기"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT template_name, starter_questions, key_vocabulary, 
                   discussion_structure, level_adaptations
            FROM topic_templates
            WHERE category = ?
            LIMIT 1
        """
        
        cursor.execute(query, (category,))
        row = cursor.fetchone()
        
        if row:
            level_adaptations = json.loads(row[4])
            
            return {
                'template_name': row[0],
                'starter_questions': json.loads(row[1]),
                'key_vocabulary': json.loads(row[2]),
                'discussion_structure': json.loads(row[3]),
                'level_specific': level_adaptations.get(level, "General discussion approach")
            }
        
        # 기본 템플릿
        return {
            'template_name': 'General Discussion',
            'starter_questions': [
                "What do you think about this topic?",
                "Have you experienced something similar?",
                "What's your opinion?"
            ],
            'key_vocabulary': ["interesting", "important", "perspective", "opinion", "experience"],
            'discussion_structure': {
                'intro': "Topic introduction",
                'main': "Open discussion",
                'conclusion': "Summary and reflection"
            },
            'level_specific': "Adapted to your level"
        }
    
    async def cleanup_expired_trends(self):
        """만료된 트렌드 정리"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            "DELETE FROM trending_topics WHERE expires_at < ?",
            (datetime.now(),)
        )
        
        deleted = cursor.rowcount
        self.conn.commit()
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} expired trends")
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()