"""
대화 엔진 - Cornell DB, PersonaChat, Google Trends 통합
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import openai
from pytrends.request import TrendReq
import random
import logging
import hashlib
from collections import defaultdict

from database.cornell_db import CornellDatabase
from database.personachat_db import PersonaChatDatabase
from database.trends_db import TrendsDatabase

logger = logging.getLogger(__name__)

class ConversationEngine:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.cornell_db = CornellDatabase()
        self.personachat_db = PersonaChatDatabase()
        self.trends_db = TrendsDatabase()
        self.pytrends = TrendReq()
        self.sessions = {}
        
        # OpenAI API 설정
        if openai_api_key:
            openai.api_key = openai_api_key
            
        # 학습 레벨 정의
        self.levels = {
            'beginner': {'vocab_size': 1000, 'sentence_complexity': 'simple'},
            'intermediate': {'vocab_size': 3000, 'sentence_complexity': 'moderate'},
            'advanced': {'vocab_size': 6000, 'sentence_complexity': 'complex'}
        }
        
    async def initialize_databases(self):
        """데이터베이스 초기화"""
        await self.cornell_db.initialize()
        await self.personachat_db.initialize()
        await self.trends_db.initialize()
        logger.info("All databases initialized successfully")
        
    async def initialize_session(self, user_id: str) -> Dict:
        """대화 세션 초기화"""
        
        # 사용자 프로필 로드
        user_profile = await self.load_user_profile(user_id)
        
        # 오늘의 트렌드 토픽 선택
        trending_topic = await self.select_trending_topic(user_profile)
        
        # 관련 대화 패턴 로드
        cornell_patterns = await self.cornell_db.find_related_patterns(
            trending_topic['keywords']
        )
        persona_context = await self.personachat_db.match_persona(user_profile)
        
        # 세션 생성
        session = {
            'id': self.generate_session_id(),
            'user_id': user_id,
            'created_at': datetime.now(),
            'topic': trending_topic,
            'cornell_patterns': cornell_patterns,
            'persona_context': persona_context,
            'conversation_history': [],
            'error_log': [],
            'learning_points': [],
            'user_level': user_profile.get('level', 'intermediate'),
            'corrections_made': 0,
            'vocabulary_introduced': []
        }
        
        self.sessions[session['id']] = session
        return session
    
    async def generate_response(
        self, 
        user_input: str, 
        session_id: str,
        include_correction: bool = True
    ) -> Dict:
        """AI 응답 생성"""
        
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # 문법 오류 체크
        errors = await self.check_grammar_errors(user_input)
        
        # 컨텍스트 구성
        context = self.build_context(session)
        
        # 적절한 Cornell 대화 패턴 선택
        relevant_pattern = self.select_relevant_pattern(
            user_input,
            session['cornell_patterns']
        )
        
        # GPT-4 응답 생성
        prompt = self.build_prompt(
            user_input=user_input,
            session=session,
            errors=errors if include_correction else [],
            pattern=relevant_pattern
        )
        
        try:
            response = await self.call_openai_api(prompt)
            ai_response = response['content']
            
            # 새로운 어휘 추출
            new_vocabulary = self.extract_new_vocabulary(
                ai_response,
                session['user_level']
            )
            
            # 대화 기록 저장
            session['conversation_history'].append({
                'user': user_input,
                'ai': ai_response,
                'timestamp': datetime.now(),
                'errors_detected': errors,
                'vocabulary_introduced': new_vocabulary
            })
            
            # 학습 포인트 업데이트
            if errors:
                session['corrections_made'] += len(errors)
                session['error_log'].extend(errors)
            
            if new_vocabulary:
                session['vocabulary_introduced'].extend(new_vocabulary)
            
            return {
                'text': ai_response,
                'session_id': session_id,
                'topic_reference': session['topic']['title'],
                'corrections': errors if include_correction else [],
                'new_vocabulary': new_vocabulary,
                'pattern_used': relevant_pattern.get('pattern_name') if relevant_pattern else None
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback response
            return self.generate_fallback_response(user_input, session)
    
    async def select_trending_topic(self, user_profile: Dict) -> Dict:
        """사용자에게 맞는 트렌딩 토픽 선택"""
        
        try:
            # Google Trends에서 오늘의 트렌드 가져오기
            trending = self.pytrends.trending_searches(pn='united_states')
            trends_list = trending[0].tolist()[:10] if not trending.empty else []
            
            # 캐시된 트렌드 사용 (API 제한 대응)
            if not trends_list:
                trends_list = await self.trends_db.get_cached_trends()
            
            # 사용자 관심사와 매칭
            best_match = None
            max_score = 0
            
            for trend in trends_list:
                score = await self.calculate_relevance_score(
                    trend, 
                    user_profile.get('interests', [])
                )
                if score > max_score:
                    max_score = score
                    best_match = trend
            
            # 트렌드가 없으면 기본 토픽 사용
            if not best_match:
                best_match = self.get_default_topic(user_profile)
            
            # 트렌드 상세 정보 구성
            topic_details = {
                'title': best_match,
                'keywords': self.extract_keywords(best_match),
                'difficulty': self.assess_difficulty(best_match, user_profile.get('level', 'intermediate')),
                'relevance_score': max_score,
                'category': self.categorize_topic(best_match)
            }
            
            # 트렌드 DB에 저장
            await self.trends_db.save_topic(topic_details)
            
            return topic_details
            
        except Exception as e:
            logger.error(f"Error selecting trending topic: {e}")
            return self.get_default_topic(user_profile)
    
    def build_context(self, session: Dict) -> str:
        """대화 컨텍스트 구성"""
        history = session['conversation_history']
        
        context = f"Topic: {session['topic']['title']}\n"
        context += f"User Level: {session.get('user_level', 'intermediate')}\n"
        context += f"Topic Category: {session['topic'].get('category', 'general')}\n"
        
        if history:
            context += "\nRecent conversation:\n"
            for turn in history[-3:]:
                context += f"User: {turn['user']}\n"
                context += f"AI: {turn['ai']}\n"
        
        # 페르소나 컨텍스트 추가
        if session.get('persona_context'):
            context += f"\nPersona traits: {session['persona_context'].get('traits', [])}\n"
        
        return context
    
    def build_prompt(
        self,
        user_input: str,
        session: Dict,
        errors: List[Dict],
        pattern: Optional[Dict]
    ) -> str:
        """GPT 프롬프트 구성"""
        
        level_info = self.levels[session.get('user_level', 'intermediate')]
        
        prompt = f"""You are an English teacher having a natural conversation about: {session['topic']['title']}

User Level: {session.get('user_level', 'intermediate')}
Vocabulary limit: {level_info['vocab_size']} words
Sentence complexity: {level_info['sentence_complexity']}

Context from movie dialogues (Cornell):
{json.dumps(session['cornell_patterns'][:3], indent=2) if session.get('cornell_patterns') else 'None'}

Persona context:
{json.dumps(session.get('persona_context', {}), indent=2)}

Conversation history:
{self.format_history(session['conversation_history'][-5:])}

User said: "{user_input}"
"""

        if errors:
            prompt += f"\nGrammar errors detected: {json.dumps(errors, indent=2)}"
            prompt += "\nGently correct these errors in your response."
        
        if pattern:
            prompt += f"\nConsider using this conversation pattern: {pattern.get('pattern', '')}"
        
        prompt += """

Generate a natural, educational response that:
1. Continues the conversation naturally about the trending topic
2. Introduces 1-2 new vocabulary words appropriate for the user's level
3. Corrects errors gently if present (don't be obvious about it)
4. Matches the user's proficiency level
5. Encourages continued conversation
6. Uses natural, conversational English

Keep the response concise (2-3 sentences max)."""
        
        return prompt
    
    async def check_grammar_errors(self, text: str) -> List[Dict]:
        """문법 오류 체크"""
        # 실제로는 LanguageTool API나 다른 문법 체크 서비스 사용
        # 여기서는 간단한 예시
        errors = []
        
        common_errors = {
            "dont": "don't",
            "doesnt": "doesn't",
            "wont": "won't",
            "cant": "can't",
            "im": "I'm",
            "youre": "you're",
            "theyre": "they're"
        }
        
        words = text.lower().split()
        for word in words:
            if word in common_errors:
                errors.append({
                    'error': word,
                    'correction': common_errors[word],
                    'type': 'contraction'
                })
        
        return errors
    
    def select_relevant_pattern(
        self,
        user_input: str,
        patterns: List[Dict]
    ) -> Optional[Dict]:
        """관련 대화 패턴 선택"""
        if not patterns:
            return None
        
        # 간단한 키워드 매칭
        user_words = set(user_input.lower().split())
        best_match = None
        max_overlap = 0
        
        for pattern in patterns:
            pattern_words = set(pattern.get('text', '').lower().split())
            overlap = len(user_words & pattern_words)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = pattern
        
        return best_match if max_overlap > 2 else random.choice(patterns[:3])
    
    def extract_new_vocabulary(self, text: str, user_level: str) -> List[Dict]:
        """새로운 어휘 추출"""
        # 실제로는 더 정교한 어휘 분석 필요
        level_vocab = {
            'beginner': ['great', 'interesting', 'important', 'helpful'],
            'intermediate': ['fascinating', 'crucial', 'significant', 'innovative'],
            'advanced': ['paradigm', 'ubiquitous', 'unprecedented', 'quintessential']
        }
        
        vocabulary = []
        level_words = level_vocab.get(user_level, level_vocab['intermediate'])
        
        for word in level_words:
            if word in text.lower():
                vocabulary.append({
                    'word': word,
                    'level': user_level,
                    'context': text
                })
        
        return vocabulary[:2]  # 최대 2개
    
    async def calculate_relevance_score(
        self,
        trend: str,
        interests: List[str]
    ) -> float:
        """관련성 점수 계산"""
        if not interests:
            return random.random() * 0.5  # 0-0.5 사이의 랜덤 점수
        
        trend_lower = trend.lower()
        score = 0.0
        
        for interest in interests:
            if interest.lower() in trend_lower:
                score += 1.0
            elif any(word in trend_lower for word in interest.lower().split()):
                score += 0.5
        
        return min(score / len(interests), 1.0)
    
    def extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        # 간단한 키워드 추출 (실제로는 NLP 라이브러리 사용)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for'}
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:5]
    
    def assess_difficulty(self, topic: str, user_level: str) -> str:
        """토픽 난이도 평가"""
        # 간단한 난이도 평가
        complex_indicators = ['technology', 'science', 'politics', 'economics', 'philosophy']
        simple_indicators = ['food', 'travel', 'music', 'sports', 'movies']
        
        topic_lower = topic.lower()
        
        if any(ind in topic_lower for ind in complex_indicators):
            if user_level == 'beginner':
                return 'challenging'
            return 'appropriate'
        elif any(ind in topic_lower for ind in simple_indicators):
            if user_level == 'advanced':
                return 'easy'
            return 'appropriate'
        
        return 'appropriate'
    
    def categorize_topic(self, topic: str) -> str:
        """토픽 카테고리 분류"""
        categories = {
            'technology': ['tech', 'ai', 'computer', 'software', 'app', 'digital'],
            'entertainment': ['movie', 'music', 'game', 'show', 'celebrity', 'film'],
            'sports': ['football', 'basketball', 'soccer', 'tennis', 'olympics', 'team'],
            'business': ['company', 'market', 'stock', 'economy', 'business', 'trade'],
            'science': ['research', 'study', 'discovery', 'space', 'medical', 'health'],
            'lifestyle': ['food', 'travel', 'fashion', 'home', 'family', 'wellness']
        }
        
        topic_lower = topic.lower()
        
        for category, keywords in categories.items():
            if any(keyword in topic_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def get_default_topic(self, user_profile: Dict) -> str:
        """기본 토픽 반환"""
        default_topics = {
            'beginner': ['Daily Routines', 'Food and Cooking', 'Weather', 'Hobbies'],
            'intermediate': ['Technology Trends', 'Travel Experiences', 'Current Events', 'Culture'],
            'advanced': ['Global Issues', 'Innovation', 'Philosophy', 'Science Discoveries']
        }
        
        level = user_profile.get('level', 'intermediate')
        topics = default_topics.get(level, default_topics['intermediate'])
        return random.choice(topics)
    
    def format_history(self, history: List[Dict]) -> str:
        """대화 기록 포맷팅"""
        if not history:
            return "No previous conversation"
        
        formatted = []
        for turn in history:
            formatted.append(f"User: {turn['user']}")
            formatted.append(f"AI: {turn['ai']}")
        
        return "\n".join(formatted)
    
    async def call_openai_api(self, prompt: str) -> Dict:
        """OpenAI API 호출"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a friendly, encouraging English teacher who helps users learn through natural conversation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            return {
                'content': response.choices[0].message.content,
                'usage': response.usage
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_fallback_response(self, user_input: str, session: Dict) -> Dict:
        """폴백 응답 생성"""
        fallback_responses = [
            "That's interesting! Can you tell me more about that?",
            "I see what you mean. What do you think about it?",
            "That's a great point! How does it relate to {topic}?",
            "Interesting perspective! Have you considered the other side?",
            "Good observation! What made you think of that?"
        ]
        
        response = random.choice(fallback_responses)
        if '{topic}' in response:
            response = response.format(topic=session['topic']['title'])
        
        return {
            'text': response,
            'session_id': session['id'],
            'topic_reference': session['topic']['title'],
            'corrections': [],
            'new_vocabulary': [],
            'pattern_used': None,
            'is_fallback': True
        }
    
    def generate_session_id(self) -> str:
        """세션 ID 생성"""
        timestamp = datetime.now().isoformat()
        random_str = str(random.randint(1000, 9999))
        return hashlib.md5(f"{timestamp}{random_str}".encode()).hexdigest()[:12]
    
    async def load_user_profile(self, user_id: str) -> Dict:
        """사용자 프로필 로드"""
        # 실제로는 DB에서 로드
        # 여기서는 예시 프로필 반환
        return {
            'user_id': user_id,
            'level': 'intermediate',
            'interests': ['technology', 'travel', 'culture'],
            'learning_goals': ['improve speaking', 'expand vocabulary'],
            'native_language': 'korean',
            'age_group': '20-30'
        }
    
    async def save_session(self, session_id: str):
        """세션 저장"""
        session = self.sessions.get(session_id)
        if session:
            # DB에 저장 로직
            logger.info(f"Session {session_id} saved")
    
    async def get_learning_summary(self, session_id: str) -> Dict:
        """학습 요약 생성"""
        session = self.sessions.get(session_id)
        if not session:
            return {}
        
        return {
            'session_id': session_id,
            'duration': (datetime.now() - session['created_at']).total_seconds(),
            'topic': session['topic']['title'],
            'total_exchanges': len(session['conversation_history']),
            'corrections_made': session['corrections_made'],
            'vocabulary_learned': len(session['vocabulary_introduced']),
            'error_patterns': self.analyze_error_patterns(session['error_log']),
            'proficiency_assessment': self.assess_proficiency(session)
        }
    
    def analyze_error_patterns(self, errors: List[Dict]) -> Dict:
        """오류 패턴 분석"""
        patterns = defaultdict(int)
        for error in errors:
            patterns[error.get('type', 'unknown')] += 1
        
        return dict(patterns)
    
    def assess_proficiency(self, session: Dict) -> Dict:
        """능숙도 평가"""
        history = session['conversation_history']
        if not history:
            return {'level': session.get('user_level', 'intermediate'), 'confidence': 0.5}
        
        # 간단한 평가 로직
        avg_length = sum(len(turn['user'].split()) for turn in history) / len(history)
        error_rate = session['corrections_made'] / max(len(history), 1)
        
        if avg_length > 15 and error_rate < 0.1:
            suggested_level = 'advanced'
        elif avg_length > 8 and error_rate < 0.3:
            suggested_level = 'intermediate'
        else:
            suggested_level = 'beginner'
        
        return {
            'current_level': session.get('user_level', 'intermediate'),
            'suggested_level': suggested_level,
            'confidence': 1.0 - error_rate,
            'average_sentence_length': avg_length
        }