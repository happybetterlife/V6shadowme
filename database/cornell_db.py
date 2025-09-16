"""
Cornell Movie Dialog Database
영화 대화 데이터베이스 관리
"""

import sqlite3
import json
import asyncio
from typing import List, Dict, Optional
import logging
import re
from pathlib import Path
import aiohttp
import zipfile
import io

logger = logging.getLogger(__name__)

class CornellDatabase:
    def __init__(self, db_path: str = "database/cornell_dialogs.db"):
        self.db_path = db_path
        self.conn = None
        self.data_url = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
        
    async def initialize(self):
        """데이터베이스 초기화"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        await self.create_tables()
        
        # 데이터가 없으면 다운로드
        if not await self.has_data():
            await self.download_and_import_data()
        
        logger.info("Cornell Database initialized")
    
    async def create_tables(self):
        """테이블 생성"""
        cursor = self.conn.cursor()
        
        # 영화 정보 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                year INTEGER,
                rating REAL,
                genres TEXT
            )
        """)
        
        # 캐릭터 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                movie_id TEXT,
                gender TEXT,
                FOREIGN KEY (movie_id) REFERENCES movies (id)
            )
        """)
        
        # 대화 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                character1_id TEXT,
                character2_id TEXT,
                movie_id TEXT,
                FOREIGN KEY (character1_id) REFERENCES characters (id),
                FOREIGN KEY (character2_id) REFERENCES characters (id),
                FOREIGN KEY (movie_id) REFERENCES movies (id)
            )
        """)
        
        # 대사 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lines (
                id TEXT PRIMARY KEY,
                character_id TEXT,
                conversation_id TEXT,
                text TEXT NOT NULL,
                position INTEGER,
                FOREIGN KEY (character_id) REFERENCES characters (id),
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
        """)
        
        # 대화 패턴 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT,
                pattern_type TEXT,
                example_conversation_id TEXT,
                keywords TEXT,
                difficulty_level TEXT,
                usage_count INTEGER DEFAULT 0,
                FOREIGN KEY (example_conversation_id) REFERENCES conversations (id)
            )
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lines_text ON lines (text)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_keywords ON patterns (keywords)")
        
        self.conn.commit()
    
    async def has_data(self) -> bool:
        """데이터 존재 여부 확인"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM lines")
        count = cursor.fetchone()[0]
        return count > 0
    
    async def download_and_import_data(self):
        """Cornell 데이터 다운로드 및 임포트"""
        logger.info("Downloading Cornell Movie Dialogs Corpus...")
        
        # 실제 구현에서는 데이터를 다운로드하고 파싱
        # 여기서는 샘플 데이터 삽입
        await self.insert_sample_data()
    
    async def insert_sample_data(self):
        """샘플 데이터 삽입"""
        cursor = self.conn.cursor()
        
        # 샘플 영화
        movies = [
            ("m1", "The Social Network", 2010, 7.7, "Drama,Biography"),
            ("m2", "Inception", 2010, 8.8, "Action,Sci-Fi,Thriller"),
            ("m3", "The Devil Wears Prada", 2006, 6.9, "Comedy,Drama")
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO movies (id, title, year, rating, genres) VALUES (?, ?, ?, ?, ?)",
            movies
        )
        
        # 샘플 캐릭터
        characters = [
            ("c1", "Mark Zuckerberg", "m1", "M"),
            ("c2", "Eduardo Saverin", "m1", "M"),
            ("c3", "Dom Cobb", "m2", "M"),
            ("c4", "Ariadne", "m2", "F"),
            ("c5", "Miranda Priestly", "m3", "F"),
            ("c6", "Andy Sachs", "m3", "F")
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO characters (id, name, movie_id, gender) VALUES (?, ?, ?, ?)",
            characters
        )
        
        # 샘플 대화
        conversations = [
            ("conv1", "c1", "c2", "m1"),
            ("conv2", "c3", "c4", "m2"),
            ("conv3", "c5", "c6", "m3")
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO conversations (id, character1_id, character2_id, movie_id) VALUES (?, ?, ?, ?)",
            conversations
        )
        
        # 샘플 대사
        lines = [
            ("L1", "c1", "conv1", "We need to expand to other schools.", 1),
            ("L2", "c2", "conv1", "I agree, but we need more servers first.", 2),
            ("L3", "c1", "conv1", "I'll handle the technical side. You focus on the business.", 3),
            
            ("L4", "c3", "conv2", "You're asking me to let someone else into my dreams?", 1),
            ("L5", "c4", "conv2", "Not just anyone. Someone who understands the architecture.", 2),
            ("L6", "c3", "conv2", "Dreams feel real while we're in them. It's only when we wake up that we realize something was strange.", 3),
            
            ("L7", "c5", "conv3", "Is there some reason that my coffee isn't here? Has she died or something?", 1),
            ("L8", "c6", "conv3", "No, I just... I thought you didn't want it.", 2),
            ("L9", "c5", "conv3", "Details of your incompetence do not interest me.", 3)
        ]
        
        cursor.executemany(
            "INSERT OR IGNORE INTO lines (id, character_id, conversation_id, text, position) VALUES (?, ?, ?, ?, ?)",
            lines
        )
        
        # 샘플 패턴
        patterns = [
            ("Agreement and Expansion", "collaborative", "conv1", "agree,expand,need,but", "intermediate", 0),
            ("Question and Explanation", "explanatory", "conv2", "asking,understand,realize,strange", "advanced", 0),
            ("Criticism and Response", "confrontational", "conv3", "reason,thought,interest,incompetence", "advanced", 0),
            ("Small Talk", "casual", None, "how,today,weather,weekend", "beginner", 0),
            ("Problem Solving", "analytical", None, "solution,issue,fix,handle", "intermediate", 0)
        ]
        
        cursor.executemany(
            """INSERT OR IGNORE INTO patterns 
               (pattern_name, pattern_type, example_conversation_id, keywords, difficulty_level, usage_count) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            patterns
        )
        
        self.conn.commit()
        logger.info("Sample data inserted")
    
    async def find_related_patterns(self, keywords: List[str]) -> List[Dict]:
        """키워드와 관련된 대화 패턴 찾기"""
        cursor = self.conn.cursor()
        
        # 키워드 매칭
        keyword_str = ' '.join(keywords).lower()
        
        query = """
            SELECT p.*, c.id as conv_id, 
                   GROUP_CONCAT(l.text, ' | ') as example_lines
            FROM patterns p
            LEFT JOIN conversations c ON p.example_conversation_id = c.id
            LEFT JOIN lines l ON l.conversation_id = c.id
            WHERE p.keywords LIKE ? OR p.pattern_name LIKE ?
            GROUP BY p.id
            ORDER BY p.usage_count DESC
            LIMIT 10
        """
        
        results = []
        for keyword in keywords:
            cursor.execute(query, (f"%{keyword}%", f"%{keyword}%"))
            rows = cursor.fetchall()
            
            for row in rows:
                results.append({
                    'id': row[0],
                    'pattern_name': row[1],
                    'pattern_type': row[2],
                    'keywords': row[4],
                    'difficulty_level': row[5],
                    'usage_count': row[6],
                    'example_lines': row[8] if row[8] else None
                })
        
        # 중복 제거 및 상위 5개 반환
        seen = set()
        unique_results = []
        for r in results:
            if r['id'] not in seen:
                seen.add(r['id'])
                unique_results.append(r)
        
        return unique_results[:5]
    
    async def search_conversations(
        self,
        text_query: str,
        limit: int = 10
    ) -> List[Dict]:
        """텍스트로 대화 검색"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT DISTINCT c.id, c.character1_id, c.character2_id, 
                   m.title, m.year, GROUP_CONCAT(l.text, ' | ') as dialogue
            FROM conversations c
            JOIN lines l ON l.conversation_id = c.id
            JOIN movies m ON c.movie_id = m.id
            WHERE l.text LIKE ?
            GROUP BY c.id
            LIMIT ?
        """
        
        cursor.execute(query, (f"%{text_query}%", limit))
        results = []
        
        for row in cursor.fetchall():
            results.append({
                'conversation_id': row[0],
                'movie_title': row[3],
                'movie_year': row[4],
                'dialogue': row[5]
            })
        
        return results
    
    async def get_conversation_by_difficulty(
        self,
        difficulty: str,
        limit: int = 5
    ) -> List[Dict]:
        """난이도별 대화 가져오기"""
        cursor = self.conn.cursor()
        
        # 단어 수와 복잡도로 난이도 판단
        if difficulty == "beginner":
            word_count_range = (1, 10)
        elif difficulty == "intermediate":
            word_count_range = (10, 20)
        else:  # advanced
            word_count_range = (20, 100)
        
        query = """
            SELECT c.id, l.text, m.title
            FROM conversations c
            JOIN lines l ON l.conversation_id = c.id
            JOIN movies m ON c.movie_id = m.id
            WHERE LENGTH(l.text) - LENGTH(REPLACE(l.text, ' ', '')) + 1 BETWEEN ? AND ?
            ORDER BY RANDOM()
            LIMIT ?
        """
        
        cursor.execute(query, (word_count_range[0], word_count_range[1], limit))
        results = []
        
        for row in cursor.fetchall():
            results.append({
                'conversation_id': row[0],
                'text': row[1],
                'movie': row[2],
                'difficulty': difficulty
            })
        
        return results
    
    async def get_genre_conversations(self, genre: str) -> List[Dict]:
        """장르별 대화 가져오기"""
        cursor = self.conn.cursor()
        
        query = """
            SELECT DISTINCT c.id, l.text, m.title, m.genres
            FROM conversations c
            JOIN lines l ON l.conversation_id = c.id
            JOIN movies m ON c.movie_id = m.id
            WHERE m.genres LIKE ?
            ORDER BY RANDOM()
            LIMIT 10
        """
        
        cursor.execute(query, (f"%{genre}%",))
        results = []
        
        for row in cursor.fetchall():
            results.append({
                'conversation_id': row[0],
                'text': row[1],
                'movie': row[2],
                'genres': row[3]
            })
        
        return results
    
    async def update_pattern_usage(self, pattern_id: int):
        """패턴 사용 횟수 업데이트"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE patterns SET usage_count = usage_count + 1 WHERE id = ?",
            (pattern_id,)
        )
        self.conn.commit()
    
    async def add_custom_pattern(
        self,
        pattern_name: str,
        pattern_type: str,
        keywords: List[str],
        difficulty: str
    ) -> int:
        """커스텀 패턴 추가"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """INSERT INTO patterns 
               (pattern_name, pattern_type, keywords, difficulty_level, usage_count)
               VALUES (?, ?, ?, ?, 0)""",
            (pattern_name, pattern_type, ','.join(keywords), difficulty)
        )
        
        self.conn.commit()
        return cursor.lastrowid
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()