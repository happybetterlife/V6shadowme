"""
에이전트 오케스트레이터
"""

import asyncio
import logging
from typing import Dict, List
from datetime import datetime
import openai

from agents.level_manager import LevelManagerAgent
from agents.memory_agent import MemoryAgent
from agents.trend_agent import TrendAgent
from agents.goal_agent import GoalAgent
from agents.error_agent import ErrorAgent

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(self, openai_api_key: str = None):
        self.agents = {
            'level_manager': LevelManagerAgent(),
            'memory': MemoryAgent(),
            'trend': TrendAgent(),
            'goal': GoalAgent(),
            'error': ErrorAgent()
        }
        self.conversation_cache = {}
        
        # OpenAI API 설정
        if openai_api_key:
            openai.api_key = openai_api_key
        
    async def initialize(self):
        """모든 에이전트 초기화"""
        for name, agent in self.agents.items():
            await agent.initialize()
            logger.info(f"Agent {name} initialized")
    
    async def process_conversation(
        self,
        user_id: str,
        user_input: str,
        session_id: str
    ) -> Dict:
        """대화 처리 및 에이전트 조율"""
        
        try:
            # 1. Memory Agent - 컨텍스트 로드
            context = await self.agents['memory'].load_context(user_id)
            
            # 2. Level Manager - 현재 레벨 확인
            current_level = await self.agents['level_manager'].get_level(user_id)
            
            # 3. Trend Agent - 관련 트렌드 확인
            trending_context = await self.agents['trend'].get_relevant_trends(
                user_input,
                context
            )
            
            # 4. Error Agent - 오류 분석
            errors = await self.agents['error'].analyze(user_input)
            
            # 5. 통합 응답 생성
            response = await self.generate_integrated_response(
                user_input=user_input,
                context=context,
                level=current_level,
                trends=trending_context,
                errors=errors,
                session_id=session_id
            )
            
            # 6. Goal Agent - 진도 업데이트
            await self.agents['goal'].update_progress(user_id, user_input, response)
            
            # 7. Memory Agent - 대화 저장
            await self.agents['memory'].store_conversation(
                user_id,
                user_input,
                response,
                errors
            )
            
            return {
                'text': response['text'],
                'errors': errors,
                'corrections': response.get('corrections', []),
                'suggestions': response.get('suggestions', []),
                'emotion': response.get('emotion', 'neutral'),
                'learning_points': response.get('learning_points', []),
                'vocabulary_suggestions': response.get('vocabulary_suggestions', []),
                'pronunciation_tips': response.get('pronunciation_tips', []),
                'progress_update': response.get('progress_update', {})
            }
            
        except Exception as e:
            logger.error(f"Error in process_conversation: {e}")
            return self.generate_fallback_response(user_input)
    
    async def generate_integrated_response(
        self,
        user_input: str,
        context: Dict,
        level: str,
        trends: Dict,
        errors: List,
        session_id: str
    ) -> Dict:
        """통합 응답 생성"""
        
        # GPT-4를 사용한 응답 생성
        prompt = self.build_prompt(
            user_input,
            context,
            level,
            trends,
            errors
        )
        
        try:
            response = await self.call_llm(prompt)
            
            # 응답 후처리
            processed_response = await self.post_process_response(
                response,
                level,
                errors
            )
            
            return processed_response
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self.generate_simple_response(user_input, errors)
    
    def build_prompt(
        self,
        user_input: str,
        context: Dict,
        level: str,
        trends: Dict,
        errors: List
    ) -> str:
        """LLM 프롬프트 구성"""
        
        prompt = f"""You are an AI English teacher. Your task is to respond naturally to the student's input while providing educational value.

Student Level: {level}
Student Input: "{user_input}"

Context from previous conversations:
{context.get('recent_topics', [])}
{context.get('learning_preferences', {})}

Current trending topics (if relevant):
{trends.get('topics', [])}

Grammar/Language errors detected:
{errors if errors else 'No errors detected'}

Instructions:
1. Respond naturally and encouragingly to the student's input
2. If errors exist, correct them gently without being obvious
3. Introduce 1-2 new vocabulary words appropriate for their level
4. Ask a follow-up question to continue the conversation
5. Stay on topic but incorporate trending topics if natural
6. Keep response conversational and not lecture-like
7. Maximum 2-3 sentences

Response should be natural, supportive, and educational."""
        
        return prompt
    
    async def call_llm(self, prompt: str) -> Dict:
        """LLM API 호출"""
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a friendly, encouraging English teacher focused on natural conversation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            
            return {
                'text': response.choices[0].message.content,
                'usage': response.usage
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def post_process_response(
        self,
        response: Dict,
        level: str,
        errors: List
    ) -> Dict:
        """응답 후처리"""
        
        text = response['text']
        
        # 교정 제안 추가
        corrections = []
        if errors:
            corrections = [f"Instead of '{error.get('error', '')}', try '{error.get('correction', '')}'" for error in errors[:2]]
        
        # 어휘 제안 추가
        vocabulary_suggestions = await self.extract_vocabulary_from_response(text, level)
        
        # 발음 팁 추가
        pronunciation_tips = await self.generate_pronunciation_tips(text)
        
        # 학습 포인트 추출
        learning_points = await self.extract_learning_points(text, errors)
        
        return {
            'text': text,
            'corrections': corrections,
            'vocabulary_suggestions': vocabulary_suggestions,
            'pronunciation_tips': pronunciation_tips,
            'learning_points': learning_points,
            'emotion': 'encouraging' if errors else 'neutral'
        }
    
    async def extract_vocabulary_from_response(self, text: str, level: str) -> List[Dict]:
        """응답에서 어휘 추출"""
        # 레벨별 고급 어휘 식별
        advanced_words = {
            'beginner': ['interesting', 'important', 'beautiful', 'delicious'],
            'intermediate': ['fascinating', 'significant', 'gorgeous', 'exquisite'],
            'advanced': ['captivating', 'paramount', 'resplendent', 'sublime']
        }
        
        words_for_level = advanced_words.get(level, advanced_words['intermediate'])
        found_vocabulary = []
        
        for word in words_for_level:
            if word.lower() in text.lower():
                found_vocabulary.append({
                    'word': word,
                    'definition': await self.get_word_definition(word),
                    'example': f"Example: {text}"
                })
        
        return found_vocabulary[:2]  # 최대 2개
    
    async def get_word_definition(self, word: str) -> str:
        """단어 정의 제공"""
        definitions = {
            'fascinating': 'extremely interesting and attractive',
            'significant': 'important or having a major effect',
            'gorgeous': 'very beautiful and attractive',
            'exquisite': 'extremely beautiful and delicate'
        }
        return definitions.get(word, 'an important word to learn')
    
    async def generate_pronunciation_tips(self, text: str) -> List[str]:
        """발음 팁 생성"""
        tips = []
        
        # 한국어 화자가 어려워하는 음소 체크
        if 'th' in text.lower():
            tips.append("Practice 'th' sound: put your tongue between your teeth")
        if any(word in text.lower() for word in ['red', 'right', 'really']):
            tips.append("Practice 'r' sound: curl your tongue back slightly")
        if any(word in text.lower() for word in ['light', 'like', 'love']):
            tips.append("Practice 'l' sound: touch your tongue to the roof of your mouth")
        
        return tips[:2]
    
    async def extract_learning_points(self, text: str, errors: List) -> List[str]:
        """학습 포인트 추출"""
        points = []
        
        if errors:
            points.append(f"Grammar focus: {errors[0].get('type', 'sentence structure')}")
        
        if len(text.split()) > 15:
            points.append("Practice with complex sentences")
        else:
            points.append("Good use of clear, simple sentences")
        
        return points
    
    def generate_simple_response(self, user_input: str, errors: List) -> Dict:
        """간단한 응답 생성 (LLM 실패시)"""
        
        if errors:
            text = f"I understand what you mean! {self.gentle_correction(errors[0])} What else would you like to talk about?"
        else:
            text = f"That's really interesting! Can you tell me more about that?"
        
        return {
            'text': text,
            'corrections': [f"Try: '{errors[0].get('correction', '')}'" for error in errors[:1]] if errors else [],
            'vocabulary_suggestions': [],
            'pronunciation_tips': [],
            'learning_points': ['Keep practicing!'],
            'emotion': 'encouraging'
        }
    
    def gentle_correction(self, error: Dict) -> str:
        """부드러운 교정"""
        correction_templates = [
            f"You could also say '{error.get('correction', '')}'",
            f"Another way to express that is '{error.get('correction', '')}'",
            f"Native speakers often say '{error.get('correction', '')}'"
        ]
        
        import random
        return random.choice(correction_templates)
    
    def generate_fallback_response(self, user_input: str) -> Dict:
        """폴백 응답 (시스템 오류시)"""
        fallback_responses = [
            "That's a great point! Can you tell me more about your thoughts on that?",
            "I find that really interesting! What's your experience with that?",
            "That's worth discussing! How do you feel about it?",
            "Good observation! What made you think of that?"
        ]
        
        import random
        return {
            'text': random.choice(fallback_responses),
            'errors': [],
            'corrections': [],
            'suggestions': [],
            'emotion': 'neutral',
            'learning_points': [],
            'vocabulary_suggestions': [],
            'pronunciation_tips': [],
            'progress_update': {},
            'is_fallback': True
        }
    
    async def get_user_errors(self, user_id: str) -> List[Dict]:
        """사용자 오류 패턴 가져오기"""
        return await self.agents['error'].get_user_error_patterns(user_id)
    
    async def update_user_profile(self, user_id: str, feedback: Dict):
        """사용자 프로필 업데이트"""
        
        # Level Manager 업데이트
        await self.agents['level_manager'].update_level(user_id, feedback)
        
        # Goal Agent 업데이트
        await self.agents['goal'].update_goals(user_id, feedback)
        
        # Memory Agent 업데이트
        await self.agents['memory'].update_preferences(user_id, feedback)
    
    async def get_learning_recommendations(self, user_id: str) -> Dict:
        """학습 추천 제공"""
        
        # 각 에이전트에서 추천 수집
        level_recommendations = await self.agents['level_manager'].get_recommendations(user_id)
        goal_recommendations = await self.agents['goal'].get_recommendations(user_id)
        error_recommendations = await self.agents['error'].get_recommendations(user_id)
        
        return {
            'level_focus': level_recommendations,
            'goal_activities': goal_recommendations,
            'error_improvement': error_recommendations,
            'next_steps': await self.generate_next_steps(user_id)
        }
    
    async def generate_next_steps(self, user_id: str) -> List[str]:
        """다음 학습 단계 제안"""
        
        # 사용자 현재 상태 분석
        user_level = await self.agents['level_manager'].get_level(user_id)
        error_patterns = await self.agents['error'].get_user_error_patterns(user_id)
        goals_progress = await self.agents['goal'].get_progress(user_id)
        
        next_steps = []
        
        # 레벨 기반 추천
        if user_level == 'beginner':
            next_steps.extend([
                "Practice basic conversation starters",
                "Focus on simple present tense",
                "Build core vocabulary (500 words)"
            ])
        elif user_level == 'intermediate':
            next_steps.extend([
                "Practice expressing opinions",
                "Learn past and future tenses",
                "Expand vocabulary to 2000 words"
            ])
        else:  # advanced
            next_steps.extend([
                "Practice complex discussions",
                "Master conditional sentences",
                "Focus on idiomatic expressions"
            ])
        
        # 오류 패턴 기반 추천
        if error_patterns:
            common_error = error_patterns[0].get('type', '')
            if 'grammar' in common_error:
                next_steps.append("Review grammar rules with examples")
            elif 'pronunciation' in common_error:
                next_steps.append("Practice pronunciation with audio exercises")
        
        return next_steps[:5]  # 최대 5개 추천
    
    async def analyze_conversation_quality(self, user_id: str, session_id: str) -> Dict:
        """대화 품질 분석"""
        
        conversation_history = await self.agents['memory'].get_session_history(user_id, session_id)
        
        if not conversation_history:
            return {'quality_score': 0, 'feedback': 'No conversation data available'}
        
        # 대화 품질 메트릭 계산
        total_turns = len(conversation_history)
        user_messages = [turn for turn in conversation_history if turn.get('speaker') == 'user']
        
        # 평균 메시지 길이
        avg_message_length = sum(len(msg.get('text', '').split()) for msg in user_messages) / max(len(user_messages), 1)
        
        # 오류율
        total_errors = sum(len(msg.get('errors', [])) for msg in user_messages)
        error_rate = total_errors / max(total_turns, 1)
        
        # 품질 점수 (0-100)
        quality_score = max(0, 100 - (error_rate * 50) + min(avg_message_length * 5, 25))
        
        # 피드백 생성
        if quality_score >= 80:
            feedback = "Excellent conversation! You're expressing yourself clearly and naturally."
        elif quality_score >= 60:
            feedback = "Good conversation flow. Focus on reducing small errors."
        elif quality_score >= 40:
            feedback = "Making progress! Practice more complex sentence structures."
        else:
            feedback = "Keep practicing! Focus on basic grammar and vocabulary."
        
        return {
            'quality_score': round(quality_score, 1),
            'total_turns': total_turns,
            'avg_message_length': round(avg_message_length, 1),
            'error_rate': round(error_rate * 100, 1),
            'feedback': feedback,
            'improvement_areas': await self.identify_improvement_areas(user_messages)
        }
    
    async def identify_improvement_areas(self, user_messages: List[Dict]) -> List[str]:
        """개선 영역 식별"""
        improvement_areas = []
        
        # 메시지 길이 분석
        lengths = [len(msg.get('text', '').split()) for msg in user_messages]
        avg_length = sum(lengths) / max(len(lengths), 1)
        
        if avg_length < 5:
            improvement_areas.append("Try to use longer, more detailed sentences")
        elif avg_length > 20:
            improvement_areas.append("Practice breaking down complex ideas into simpler sentences")
        
        # 어휘 다양성 분석
        all_words = []
        for msg in user_messages:
            all_words.extend(msg.get('text', '').lower().split())
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        vocabulary_diversity = unique_words / max(total_words, 1)
        
        if vocabulary_diversity < 0.6:
            improvement_areas.append("Try to use more varied vocabulary")
        
        return improvement_areas[:3]  # 최대 3개 영역