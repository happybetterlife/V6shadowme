"""
학습 엔진 - 적응형 학습 시스템
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import logging
from collections import defaultdict

from database.cornell_db import CornellDatabase
from database.personachat_db import PersonaChatDatabase
from database.trends_db import TrendsDatabase

logger = logging.getLogger(__name__)

class LearningEngine:
    def __init__(self):
        self.cornell_db = CornellDatabase()
        self.personachat_db = PersonaChatDatabase()
        self.trends_db = TrendsDatabase()
        
        # 학습 레벨 정의
        self.levels = {
            'beginner': {
                'vocabulary_range': (500, 1000),
                'sentence_length': (5, 10),
                'grammar_complexity': 'basic',
                'topics': ['daily_life', 'basic_conversation', 'simple_descriptions']
            },
            'intermediate': {
                'vocabulary_range': (1000, 3000),
                'sentence_length': (8, 15),
                'grammar_complexity': 'moderate',
                'topics': ['current_events', 'opinions', 'experiences', 'future_plans']
            },
            'advanced': {
                'vocabulary_range': (3000, 6000),
                'sentence_length': (12, 25),
                'grammar_complexity': 'complex',
                'topics': ['abstract_concepts', 'debates', 'analysis', 'professional']
            }
        }
        
        # 학습 세션 템플릿
        self.session_templates = {
            'conversation_practice': {
                'duration': 20,
                'activities': ['warm_up', 'main_conversation', 'feedback']
            },
            'shadowing_intensive': {
                'duration': 15,
                'activities': ['pronunciation_drill', 'shadowing_practice', 'fluency_check']
            },
            'vocabulary_expansion': {
                'duration': 25,
                'activities': ['word_introduction', 'context_practice', 'usage_scenarios']
            },
            'grammar_focus': {
                'duration': 30,
                'activities': ['rule_explanation', 'pattern_practice', 'error_correction']
            }
        }
        
    async def load_databases(self):
        """데이터베이스 로드"""
        await self.cornell_db.initialize()
        await self.personachat_db.initialize()
        await self.trends_db.initialize()
        logger.info("Learning databases loaded")
    
    async def create_daily_session(self, user_id: str) -> Dict:
        """일일 학습 세션 생성"""
        
        # 사용자 프로필 로드
        user_profile = await self.load_user_profile(user_id)
        
        # 오늘의 학습 목표 설정
        learning_goals = await self.set_daily_goals(user_profile)
        
        # 개인화된 커리큘럼 생성
        curriculum = await self.generate_personalized_curriculum(
            user_profile, 
            learning_goals
        )
        
        # 세션 구성
        session = {
            'id': self.generate_session_id(),
            'user_id': user_id,
            'created_at': datetime.now(),
            'goals': learning_goals,
            'curriculum': curriculum,
            'materials': await self.prepare_learning_materials(curriculum),
            'duration': self.calculate_session_duration(curriculum),
            'progress': {
                'completed_activities': 0,
                'total_activities': len(curriculum['activities']),
                'current_level': user_profile.get('level', 'intermediate')
            }
        }
        
        return session
    
    async def generate_personalized_curriculum(
        self, 
        user_profile: Dict, 
        goals: List[str]
    ) -> Dict:
        """개인화된 커리큘럼 생성"""
        
        level = user_profile.get('level', 'intermediate')
        interests = user_profile.get('interests', [])
        weak_areas = user_profile.get('weak_areas', [])
        
        activities = []
        
        # 약점 보완 활동 우선 추가
        for weak_area in weak_areas[:2]:
            activities.extend(
                await self.create_remedial_activities(weak_area, level)
            )
        
        # 관심사 기반 활동 추가
        for interest in interests[:2]:
            activities.extend(
                await self.create_interest_based_activities(interest, level)
            )
        
        # 레벨별 핵심 활동 추가
        activities.extend(
            await self.create_level_appropriate_activities(level)
        )
        
        # 트렌딩 토픽 기반 활동 (선택적)
        if 'current_events' in goals:
            trending_activities = await self.create_trending_topic_activities(
                user_profile
            )
            activities.extend(trending_activities[:1])
        
        return {
            'level': level,
            'focus_areas': weak_areas + interests,
            'activities': activities[:5],  # 최대 5개 활동
            'adaptive_elements': await self.prepare_adaptive_elements(user_profile)
        }
    
    async def prepare_learning_materials(self, curriculum: Dict) -> Dict:
        """학습 자료 준비"""
        materials = {
            'conversations': [],
            'shadowing_texts': [],
            'vocabulary_lists': [],
            'grammar_explanations': [],
            'pronunciation_guides': []
        }
        
        level = curriculum['level']
        
        for activity in curriculum['activities']:
            if activity['type'] == 'conversation':
                # Cornell 데이터베이스에서 대화 패턴 가져오기
                conversations = await self.cornell_db.get_conversation_by_difficulty(
                    level, limit=3
                )
                materials['conversations'].extend(conversations)
                
            elif activity['type'] == 'shadowing':
                # Shadowing용 문장 생성
                shadowing_texts = await self.generate_shadowing_sentences(
                    activity.get('topic', 'general'), 
                    level
                )
                materials['shadowing_texts'].extend(shadowing_texts)
                
            elif activity['type'] == 'vocabulary':
                # 어휘 리스트 생성
                vocab_list = await self.generate_vocabulary_list(
                    activity.get('topic', 'general'), 
                    level
                )
                materials['vocabulary_lists'].extend(vocab_list)
        
        return materials
    
    async def get_shadowing_materials(
        self, 
        user_id: str, 
        session_id: str
    ) -> List[Dict]:
        """Shadowing 연습 자료 가져오기"""
        
        user_profile = await self.load_user_profile(user_id)
        level = user_profile.get('level', 'intermediate')
        
        # 레벨별 문장 생성
        sentences = []
        
        # 기본 문장 (쉬운 것부터)
        basic_sentences = await self.generate_basic_shadowing_sentences(level)
        sentences.extend(basic_sentences)
        
        # Cornell 데이터베이스에서 영화 대화
        movie_dialogues = await self.cornell_db.get_conversation_by_difficulty(
            level, limit=5
        )
        
        for dialogue in movie_dialogues:
            sentences.append({
                'text': dialogue['text'],
                'source': f"Movie: {dialogue.get('movie', 'Unknown')}",
                'difficulty': dialogue['difficulty'],
                'type': 'dialogue',
                'phonetics': await self.generate_phonetics(dialogue['text']),
                'speed_variations': [0.8, 1.0, 1.2]
            })
        
        # 트렌딩 토픽 기반 문장
        trending_sentences = await self.generate_trending_shadowing_sentences(
            user_profile
        )
        sentences.extend(trending_sentences)
        
        return sentences[:15]  # 최대 15개 문장
    
    async def generate_shadowing_sentences(
        self, 
        topic: str, 
        level: str
    ) -> List[Dict]:
        """토픽별 Shadowing 문장 생성"""
        
        level_config = self.levels[level]
        sentence_count = self.session_templates['shadowing_intensive']['duration'] // 2
        
        sentences = []
        
        # 토픽별 문장 템플릿
        topic_templates = {
            'daily_life': [
                "I usually wake up at seven o'clock every morning.",
                "What do you like to do in your free time?",
                "The weather is really nice today, isn't it?",
                "I'm planning to go shopping this weekend."
            ],
            'current_events': [
                "Have you heard about the latest news?",
                "Technology is changing our lives rapidly.",
                "Climate change is a serious global issue.",
                "Social media has both positive and negative effects."
            ],
            'professional': [
                "Let's schedule a meeting to discuss the project.",
                "I need to review the quarterly financial reports.",
                "Our team's productivity has improved significantly.",
                "We should consider implementing new strategies."
            ]
        }
        
        templates = topic_templates.get(topic, topic_templates['daily_life'])
        
        for i in range(min(sentence_count, len(templates))):
            sentence = templates[i]
            sentences.append({
                'text': sentence,
                'source': f"Generated for {topic}",
                'difficulty': level,
                'type': 'template',
                'phonetics': await self.generate_phonetics(sentence),
                'focus_sounds': await self.identify_focus_sounds(sentence)
            })
        
        return sentences
    
    async def process_feedback(
        self,
        user_id: str,
        session_id: str,
        feedback: Dict
    ):
        """학습 피드백 처리 및 적응형 조정"""
        
        # 피드백 분석
        performance_score = feedback.get('performance_score', 0)
        difficulty_rating = feedback.get('difficulty_rating', 3)  # 1-5 스케일
        enjoyed_activities = feedback.get('enjoyed_activities', [])
        struggled_areas = feedback.get('struggled_areas', [])
        
        # 사용자 프로필 업데이트
        user_profile = await self.load_user_profile(user_id)
        
        # 레벨 조정 검토
        if performance_score >= 80 and difficulty_rating <= 2:
            # 레벨 업 고려
            await self.consider_level_adjustment(user_id, 'up')
        elif performance_score <= 50 and difficulty_rating >= 4:
            # 레벨 다운 고려
            await self.consider_level_adjustment(user_id, 'down')
        
        # 약점 영역 업데이트
        if struggled_areas:
            await self.update_weak_areas(user_id, struggled_areas)
        
        # 선호 활동 기록
        if enjoyed_activities:
            await self.update_preferred_activities(user_id, enjoyed_activities)
        
        # 다음 세션 추천 조정
        await self.adjust_next_session_recommendations(user_id, feedback)
        
        logger.info(f"Processed feedback for user {user_id}: {performance_score}% performance")
    
    async def create_remedial_activities(
        self, 
        weak_area: str, 
        level: str
    ) -> List[Dict]:
        """약점 보완 활동 생성"""
        activities = []
        
        if weak_area == 'pronunciation':
            activities.append({
                'type': 'pronunciation_drill',
                'topic': 'common_sounds',
                'duration': 5,
                'focus': 'difficult_sounds',
                'materials': await self.get_pronunciation_materials(level)
            })
        
        elif weak_area == 'grammar':
            activities.append({
                'type': 'grammar_practice',
                'topic': 'sentence_structure',
                'duration': 8,
                'focus': 'common_errors',
                'materials': await self.get_grammar_exercises(level)
            })
        
        elif weak_area == 'vocabulary':
            activities.append({
                'type': 'vocabulary_building',
                'topic': 'high_frequency_words',
                'duration': 6,
                'focus': 'contextual_usage',
                'materials': await self.get_vocabulary_exercises(level)
            })
        
        return activities
    
    async def create_interest_based_activities(
        self, 
        interest: str, 
        level: str
    ) -> List[Dict]:
        """관심사 기반 활동 생성"""
        activities = []
        
        # 관심사별 대화 토픽 생성
        topic_conversations = await self.personachat_db.get_dialogue_examples(
            topic=interest,
            difficulty=level,
            limit=2
        )
        
        for conv in topic_conversations:
            activities.append({
                'type': 'conversation',
                'topic': interest,
                'duration': 10,
                'focus': 'topic_discussion',
                'materials': conv
            })
        
        return activities
    
    def generate_session_id(self) -> str:
        """세션 ID 생성"""
        import uuid
        return str(uuid.uuid4())[:12]
    
    def calculate_session_duration(self, curriculum: Dict) -> int:
        """세션 총 소요 시간 계산"""
        total_duration = 0
        for activity in curriculum['activities']:
            total_duration += activity.get('duration', 5)
        return total_duration
    
    async def load_user_profile(self, user_id: str) -> Dict:
        """사용자 프로필 로드"""
        # 실제 DB에서 로드 (여기서는 예시)
        return {
            'user_id': user_id,
            'level': 'intermediate',
            'interests': ['technology', 'travel'],
            'weak_areas': ['pronunciation', 'grammar'],
            'preferred_activities': ['conversation', 'shadowing'],
            'learning_style': 'visual',
            'progress': {
                'sessions_completed': 15,
                'current_streak': 5,
                'total_practice_hours': 12.5
            }
        }
    
    async def set_daily_goals(self, user_profile: Dict) -> List[str]:
        """일일 학습 목표 설정"""
        level = user_profile.get('level', 'intermediate')
        weak_areas = user_profile.get('weak_areas', [])
        
        goals = []
        
        # 레벨별 기본 목표
        if level == 'beginner':
            goals.extend(['basic_conversation', 'pronunciation_improvement'])
        elif level == 'intermediate':
            goals.extend(['fluency_building', 'vocabulary_expansion'])
        else:
            goals.extend(['advanced_discussion', 'accent_refinement'])
        
        # 약점 개선 목표 추가
        if weak_areas:
            goals.append(f"{weak_areas[0]}_focus")
        
        return goals[:3]  # 최대 3개 목표
    
    async def generate_phonetics(self, text: str) -> str:
        """음성학적 표기 생성 (간단 버전)"""
        # 실제로는 음성학 라이브러리나 API 사용
        return f"/{text.lower().replace(' ', ' ')}/"
    
    async def identify_focus_sounds(self, text: str) -> List[str]:
        """중점 연습 음소 식별"""
        # 한국어 화자가 어려워하는 음소들
        difficult_sounds = ['th', 'r', 'l', 'v', 'f']
        found_sounds = []
        
        for sound in difficult_sounds:
            if sound in text.lower():
                found_sounds.append(sound)
        
        return found_sounds
    
    async def generate_basic_shadowing_sentences(self, level: str) -> List[Dict]:
        """기본 Shadowing 문장 생성"""
        sentences = []
        
        if level == 'beginner':
            basic_texts = [
                "Hello, how are you today?",
                "Nice to meet you.",
                "What's your name?",
                "I'm fine, thank you."
            ]
        elif level == 'intermediate':
            basic_texts = [
                "Could you please help me with this?",
                "I'd like to make a reservation.",
                "What do you think about this idea?",
                "Let me know if you need anything."
            ]
        else:  # advanced
            basic_texts = [
                "I'd appreciate it if you could consider my proposal.",
                "The implications of this decision are far-reaching.",
                "We need to approach this issue from multiple perspectives.",
                "This phenomenon requires careful analysis."
            ]
        
        for text in basic_texts:
            sentences.append({
                'text': text,
                'source': 'Generated basic sentences',
                'difficulty': level,
                'type': 'basic',
                'phonetics': await self.generate_phonetics(text)
            })
        
        return sentences
    
    async def generate_trending_shadowing_sentences(
        self, 
        user_profile: Dict
    ) -> List[Dict]:
        """트렌딩 토픽 기반 Shadowing 문장"""
        trending_topics = await self.trends_db.get_cached_trends()
        sentences = []
        
        for topic in trending_topics[:3]:
            sentence = f"Have you heard about {topic}? It's really interesting."
            sentences.append({
                'text': sentence,
                'source': f'Trending: {topic}',
                'difficulty': user_profile.get('level', 'intermediate'),
                'type': 'trending',
                'phonetics': await self.generate_phonetics(sentence)
            })
        
        return sentences