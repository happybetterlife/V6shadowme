"""
데이터베이스 매니저 - 통합 데이터베이스 관리
"""

import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import json
import hashlib

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "app.db"):
        self.db_path = db_path
        self.conn = None
        
    async def initialize(self):
        """데이터베이스 초기화"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        await self.create_tables()
        logger.info("Database manager initialized")
    
    async def create_tables(self):
        """모든 테이블 생성"""
        cursor = self.conn.cursor()
        
        # 사용자 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                native_language TEXT DEFAULT 'Korean',
                target_level TEXT DEFAULT 'intermediate',
                current_level TEXT DEFAULT 'beginner',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP,
                total_practice_time INTEGER DEFAULT 0,
                streak_days INTEGER DEFAULT 0
            )
        """)
        
        # 학습 세션 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_type TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration_minutes INTEGER,
                activities_completed INTEGER DEFAULT 0,
                performance_score REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 사용자 진도 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                skill_area TEXT,
                current_level REAL,
                target_level REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                progress_data TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 음성 프로필 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                audio_sample_path TEXT,
                voice_features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 대화 기록 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_message TEXT,
                ai_response TEXT,
                errors_detected TEXT,
                corrections_made TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (session_id) REFERENCES learning_sessions (id)
            )
        """)
        
        # 학습 통계 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date DATE,
                total_interactions INTEGER DEFAULT 0,
                errors_made INTEGER DEFAULT 0,
                errors_corrected INTEGER DEFAULT 0,
                new_vocabulary INTEGER DEFAULT 0,
                practice_minutes INTEGER DEFAULT 0,
                activities_completed INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON learning_sessions (user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_progress_user ON user_progress (user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversation_logs (session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_statistics_user_date ON learning_statistics (user_id, date)")
        
        self.conn.commit()
        logger.info("Database tables created")
    
    async def create_user(
        self,
        username: str,
        email: str,
        native_language: str = "Korean",
        target_level: str = "intermediate"
    ) -> str:
        """새 사용자 생성"""
        user_id = self.generate_user_id(email)
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(
                """INSERT INTO users 
                   (id, username, email, native_language, target_level, last_active)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, username, email, native_language, target_level, datetime.now())
            )
            
            # 초기 진도 데이터 생성
            await self.initialize_user_progress(user_id)
            
            self.conn.commit()
            logger.info(f"User created: {user_id}")
            return user_id
            
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed: {e}")
            raise ValueError("Username or email already exists")
    
    async def initialize_user_progress(self, user_id: str):
        """사용자 초기 진도 설정"""
        cursor = self.conn.cursor()
        
        skill_areas = [
            ('conversation', 1.0, 5.0),
            ('pronunciation', 1.0, 5.0),
            ('vocabulary', 1.0, 5.0),
            ('grammar', 1.0, 5.0),
            ('listening', 1.0, 5.0)
        ]
        
        for skill, current, target in skill_areas:
            cursor.execute(
                """INSERT INTO user_progress 
                   (user_id, skill_area, current_level, target_level, progress_data)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, skill, current, target, json.dumps({}))
            )
        
        self.conn.commit()
    
    async def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """사용자 프로필 조회"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """SELECT id, username, email, native_language, target_level, 
                      current_level, total_practice_time, streak_days, created_at
               FROM users WHERE id = ?""",
            (user_id,)
        )
        
        row = cursor.fetchone()
        if not row:
            return None
        
        # 진도 정보 가져오기
        cursor.execute(
            """SELECT skill_area, current_level, target_level, progress_data
               FROM user_progress WHERE user_id = ?""",
            (user_id,)
        )
        
        progress_rows = cursor.fetchall()
        progress = {}
        for skill, current, target, data in progress_rows:
            progress[skill] = {
                'current_level': current,
                'target_level': target,
                'data': json.loads(data) if data else {}
            }
        
        return {
            'id': row[0],
            'username': row[1],
            'email': row[2],
            'native_language': row[3],
            'target_level': row[4],
            'current_level': row[5],
            'total_practice_time': row[6],
            'streak_days': row[7],
            'created_at': row[8],
            'progress': progress
        }
    
    async def create_learning_session(
        self,
        user_id: str,
        session_type: str = "general"
    ) -> str:
        """학습 세션 생성"""
        session_id = self.generate_session_id()
        cursor = self.conn.cursor()
        
        cursor.execute(
            """INSERT INTO learning_sessions 
               (id, user_id, session_type) VALUES (?, ?, ?)""",
            (session_id, user_id, session_type)
        )
        
        self.conn.commit()
        return session_id
    
    async def end_learning_session(
        self,
        session_id: str,
        duration_minutes: int,
        activities_completed: int,
        performance_score: float
    ):
        """학습 세션 종료"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """UPDATE learning_sessions 
               SET ended_at = ?, duration_minutes = ?, 
                   activities_completed = ?, performance_score = ?
               WHERE id = ?""",
            (datetime.now(), duration_minutes, activities_completed, 
             performance_score, session_id)
        )
        
        self.conn.commit()
    
    async def log_conversation(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        ai_response: str,
        errors_detected: List[Dict] = None,
        corrections_made: List[str] = None
    ):
        """대화 기록 저장"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """INSERT INTO conversation_logs 
               (session_id, user_id, user_message, ai_response, 
                errors_detected, corrections_made)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, user_id, user_message, ai_response,
             json.dumps(errors_detected or []),
             json.dumps(corrections_made or []))
        )
        
        self.conn.commit()
    
    async def update_user_progress(
        self,
        user_id: str,
        skill_area: str,
        new_level: float,
        progress_data: Dict = None
    ):
        """사용자 진도 업데이트"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """UPDATE user_progress 
               SET current_level = ?, progress_data = ?, last_updated = ?
               WHERE user_id = ? AND skill_area = ?""",
            (new_level, json.dumps(progress_data or {}), datetime.now(),
             user_id, skill_area)
        )
        
        self.conn.commit()
    
    async def update_daily_statistics(
        self,
        user_id: str,
        interactions: int = 0,
        errors_made: int = 0,
        errors_corrected: int = 0,
        new_vocabulary: int = 0,
        practice_minutes: int = 0,
        activities_completed: int = 0
    ):
        """일일 통계 업데이트"""
        cursor = self.conn.cursor()
        today = datetime.now().date()
        
        # 기존 데이터 확인
        cursor.execute(
            "SELECT id FROM learning_statistics WHERE user_id = ? AND date = ?",
            (user_id, today)
        )
        
        if cursor.fetchone():
            # 업데이트
            cursor.execute(
                """UPDATE learning_statistics 
                   SET total_interactions = total_interactions + ?,
                       errors_made = errors_made + ?,
                       errors_corrected = errors_corrected + ?,
                       new_vocabulary = new_vocabulary + ?,
                       practice_minutes = practice_minutes + ?,
                       activities_completed = activities_completed + ?
                   WHERE user_id = ? AND date = ?""",
                (interactions, errors_made, errors_corrected, new_vocabulary,
                 practice_minutes, activities_completed, user_id, today)
            )
        else:
            # 새로 생성
            cursor.execute(
                """INSERT INTO learning_statistics 
                   (user_id, date, total_interactions, errors_made, 
                    errors_corrected, new_vocabulary, practice_minutes, 
                    activities_completed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, today, interactions, errors_made, errors_corrected,
                 new_vocabulary, practice_minutes, activities_completed)
            )
        
        self.conn.commit()
    
    async def get_user_statistics(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict:
        """사용자 통계 조회"""
        cursor = self.conn.cursor()
        
        # 최근 N일 데이터
        cursor.execute(
            """SELECT date, total_interactions, errors_made, errors_corrected,
                      new_vocabulary, practice_minutes, activities_completed
               FROM learning_statistics 
               WHERE user_id = ? AND date >= date('now', '-{} days')
               ORDER BY date DESC""".format(days),
            (user_id,)
        )
        
        daily_stats = []
        for row in cursor.fetchall():
            daily_stats.append({
                'date': row[0],
                'interactions': row[1],
                'errors_made': row[2],
                'errors_corrected': row[3],
                'vocabulary_learned': row[4],
                'practice_minutes': row[5],
                'activities_completed': row[6]
            })
        
        # 전체 통계 계산
        total_interactions = sum(stat['interactions'] for stat in daily_stats)
        total_practice_time = sum(stat['practice_minutes'] for stat in daily_stats)
        total_errors = sum(stat['errors_made'] for stat in daily_stats)
        total_corrections = sum(stat['errors_corrected'] for stat in daily_stats)
        
        accuracy = (1 - (total_errors / max(total_interactions, 1))) * 100
        improvement_rate = (total_corrections / max(total_errors, 1)) * 100
        
        return {
            'daily_stats': daily_stats,
            'summary': {
                'total_interactions': total_interactions,
                'total_practice_hours': round(total_practice_time / 60, 1),
                'accuracy_percentage': round(accuracy, 1),
                'improvement_rate': round(improvement_rate, 1),
                'active_days': len([s for s in daily_stats if s['interactions'] > 0])
            }
        }
    
    async def save_voice_profile(
        self,
        user_id: str,
        profile_id: str,
        audio_sample_path: str,
        voice_features: Dict
    ):
        """음성 프로필 저장"""
        cursor = self.conn.cursor()
        
        # 기존 프로필 비활성화
        cursor.execute(
            "UPDATE voice_profiles SET is_active = FALSE WHERE user_id = ?",
            (user_id,)
        )
        
        # 새 프로필 저장
        cursor.execute(
            """INSERT INTO voice_profiles 
               (id, user_id, audio_sample_path, voice_features, is_active)
               VALUES (?, ?, ?, ?, TRUE)""",
            (profile_id, user_id, audio_sample_path, json.dumps(voice_features))
        )
        
        self.conn.commit()
    
    async def get_active_voice_profile(self, user_id: str) -> Optional[Dict]:
        """활성 음성 프로필 조회"""
        cursor = self.conn.cursor()
        
        cursor.execute(
            """SELECT id, audio_sample_path, voice_features, created_at
               FROM voice_profiles 
               WHERE user_id = ? AND is_active = TRUE
               ORDER BY created_at DESC LIMIT 1""",
            (user_id,)
        )
        
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'audio_sample_path': row[1],
                'voice_features': json.loads(row[2]),
                'created_at': row[3]
            }
        
        return None
    
    def generate_user_id(self, email: str) -> str:
        """사용자 ID 생성"""
        timestamp = str(datetime.now().timestamp())
        hash_input = f"{email}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def generate_session_id(self) -> str:
        """세션 ID 생성"""
        timestamp = str(datetime.now().timestamp())
        hash_input = f"session_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    async def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")