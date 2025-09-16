"""
메모리 에이전트 - 대화 컨텍스트 및 사용자 선호도 관리
"""

import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class MemoryAgent:
    def __init__(self):
        # 사용자별 메모리 저장소 (실제로는 데이터베이스 사용)
        self.user_contexts = {}
        self.conversation_histories = {}
        self.user_preferences = {}
        self.topic_interests = {}
        
        # 메모리 제한 설정
        self.max_conversation_history = 100  # 최대 대화 기록
        self.context_window_days = 7  # 컨텍스트 유지 기간
        
    async def initialize(self):
        """메모리 에이전트 초기화"""
        logger.info("Memory Agent initialized")
    
    async def load_context(self, user_id: str) -> Dict:
        """사용자 컨텍스트 로드"""
        
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {
                'created_at': datetime.now(),
                'last_active': datetime.now(),
                'session_count': 0,
                'total_interactions': 0
            }
        
        context = self.user_contexts[user_id]
        
        # 최근 대화 주제
        recent_topics = await self.get_recent_topics(user_id)
        
        # 학습 선호도
        learning_preferences = await self.get_learning_preferences(user_id)
        
        # 관심 분야
        interest_areas = await self.get_interest_areas(user_id)
        
        # 최근 대화 히스토리 (요약된 형태)
        recent_conversations = await self.get_recent_conversation_summary(user_id)
        
        # 자주 틀리는 패턴
        common_errors = await self.get_common_error_patterns(user_id)
        
        return {
            'user_profile': {
                'total_sessions': context['session_count'],
                'total_interactions': context['total_interactions'],
                'active_since': context['created_at'].isoformat(),
                'last_active': context['last_active'].isoformat()
            },
            'recent_topics': recent_topics,
            'learning_preferences': learning_preferences,
            'interests': interest_areas,
            'recent_conversations': recent_conversations,
            'common_errors': common_errors,
            'context_summary': await self.generate_context_summary(user_id)
        }
    
    async def store_conversation(
        self,
        user_id: str,
        user_input: str,
        ai_response: Dict,
        errors: List[Dict]
    ):
        """대화 내용 저장"""
        
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        
        conversation_entry = {
            'timestamp': datetime.now(),
            'user_input': user_input,
            'ai_response': ai_response.get('text', ''),
            'errors': errors,
            'topics_discussed': await self.extract_topics(user_input + ' ' + ai_response.get('text', '')),
            'user_emotion': self.detect_user_emotion(user_input),
            'conversation_quality': await self.assess_conversation_quality(user_input, ai_response),
            'learning_outcomes': ai_response.get('learning_points', [])
        }
        
        self.conversation_histories[user_id].append(conversation_entry)
        
        # 메모리 제한 적용
        if len(self.conversation_histories[user_id]) > self.max_conversation_history:
            # 오래된 대화 요약 후 저장
            await self.compress_old_conversations(user_id)
        
        # 컨텍스트 업데이트
        await self.update_context(user_id, conversation_entry)
        
        # 학습 패턴 업데이트
        await self.update_learning_patterns(user_id, conversation_entry)
    
    async def get_recent_topics(self, user_id: str, days: int = 7) -> List[Dict]:
        """최근 대화 주제 조회"""
        
        if user_id not in self.conversation_histories:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_conversations = [
            conv for conv in self.conversation_histories[user_id]
            if conv['timestamp'] > cutoff_date
        ]
        
        # 주제별 빈도 계산
        topic_frequency = defaultdict(int)
        topic_details = {}
        
        for conv in recent_conversations:
            for topic in conv.get('topics_discussed', []):
                topic_frequency[topic] += 1
                if topic not in topic_details:
                    topic_details[topic] = {
                        'topic': topic,
                        'first_mentioned': conv['timestamp'],
                        'last_mentioned': conv['timestamp'],
                        'examples': []
                    }
                else:
                    topic_details[topic]['last_mentioned'] = conv['timestamp']
                
                if len(topic_details[topic]['examples']) < 3:
                    topic_details[topic]['examples'].append({
                        'user_input': conv['user_input'][:100],  # 처음 100자만
                        'timestamp': conv['timestamp']
                    })
        
        # 빈도순으로 정렬하여 상위 토픽 반환
        sorted_topics = sorted(topic_frequency.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                **topic_details[topic],
                'frequency': freq,
                'relevance_score': self.calculate_topic_relevance(topic, freq, days)
            }
            for topic, freq in sorted_topics[:10]
        ]
    
    async def get_learning_preferences(self, user_id: str) -> Dict:
        """학습 선호도 조회"""
        
        if user_id not in self.user_preferences:
            return {
                'preferred_topics': ['general conversation'],
                'learning_style': 'conversational',
                'correction_style': 'gentle',
                'difficulty_preference': 'adaptive',
                'session_length_preference': 'medium',  # short, medium, long
                'feedback_frequency': 'moderate'  # minimal, moderate, frequent
            }
        
        return self.user_preferences[user_id]
    
    async def get_interest_areas(self, user_id: str) -> List[Dict]:
        """관심 분야 조회"""
        
        if user_id not in self.topic_interests:
            return [
                {'area': 'technology', 'interest_level': 0.7, 'interactions': 0},
                {'area': 'culture', 'interest_level': 0.6, 'interactions': 0},
                {'area': 'daily_life', 'interest_level': 0.8, 'interactions': 0}
            ]
        
        return list(self.topic_interests[user_id].values())
    
    async def get_recent_conversation_summary(self, user_id: str, count: int = 5) -> List[Dict]:
        """최근 대화 요약"""
        
        if user_id not in self.conversation_histories:
            return []
        
        recent_conversations = self.conversation_histories[user_id][-count:]
        
        summaries = []
        for conv in recent_conversations:
            summaries.append({
                'timestamp': conv['timestamp'].isoformat(),
                'user_input_preview': conv['user_input'][:50] + '...' if len(conv['user_input']) > 50 else conv['user_input'],
                'main_topics': conv.get('topics_discussed', [])[:3],
                'errors_count': len(conv.get('errors', [])),
                'quality_score': conv.get('conversation_quality', 0.7)
            })
        
        return summaries
    
    async def get_common_error_patterns(self, user_id: str) -> List[Dict]:
        """자주 발생하는 오류 패턴 분석"""
        
        if user_id not in self.conversation_histories:
            return []
        
        error_patterns = defaultdict(list)
        
        # 최근 30일간의 오류 분석
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for conv in self.conversation_histories[user_id]:
            if conv['timestamp'] > cutoff_date:
                for error in conv.get('errors', []):
                    error_type = error.get('type', 'general')
                    error_patterns[error_type].append({
                        'error': error.get('error', ''),
                        'correction': error.get('correction', ''),
                        'timestamp': conv['timestamp'],
                        'context': conv['user_input']
                    })
        
        # 빈도별로 정렬하고 패턴 분석
        common_patterns = []
        for error_type, errors in error_patterns.items():
            if len(errors) >= 2:  # 2번 이상 반복된 오류만
                common_patterns.append({
                    'error_type': error_type,
                    'frequency': len(errors),
                    'recent_examples': errors[-3:],  # 최근 3개 예시
                    'improvement_trend': self.calculate_improvement_trend(errors),
                    'recommendation': self.get_error_recommendation(error_type)
                })
        
        return sorted(common_patterns, key=lambda x: x['frequency'], reverse=True)
    
    async def update_preferences(self, user_id: str, feedback: Dict):
        """사용자 선호도 업데이트"""
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = await self.get_learning_preferences(user_id)
        
        preferences = self.user_preferences[user_id]
        
        # 피드백을 기반으로 선호도 조정
        if 'enjoyed_activities' in feedback:
            enjoyed = feedback['enjoyed_activities']
            if 'conversation' in enjoyed:
                preferences['learning_style'] = 'conversational'
            elif 'grammar_focus' in enjoyed:
                preferences['learning_style'] = 'structured'
            elif 'vocabulary_building' in enjoyed:
                preferences['learning_style'] = 'vocabulary_focused'
        
        if 'correction_preference' in feedback:
            preferences['correction_style'] = feedback['correction_preference']
        
        if 'difficulty_rating' in feedback:
            rating = feedback['difficulty_rating']
            if rating <= 2:
                preferences['difficulty_preference'] = 'easier'
            elif rating >= 4:
                preferences['difficulty_preference'] = 'harder'
            else:
                preferences['difficulty_preference'] = 'adaptive'
        
        self.user_preferences[user_id] = preferences
    
    async def extract_topics(self, text: str) -> List[str]:
        """텍스트에서 주제 추출"""
        # 간단한 키워드 기반 주제 추출 (실제로는 더 정교한 NLP 사용)
        
        topic_keywords = {
            'technology': ['computer', 'software', 'app', 'digital', 'internet', 'ai', 'robot'],
            'travel': ['travel', 'trip', 'vacation', 'country', 'city', 'culture', 'visit'],
            'food': ['food', 'eat', 'restaurant', 'cook', 'recipe', 'meal', 'dinner'],
            'work': ['work', 'job', 'career', 'office', 'business', 'company', 'meeting'],
            'education': ['study', 'school', 'university', 'learn', 'student', 'teacher', 'course'],
            'health': ['health', 'doctor', 'exercise', 'hospital', 'medicine', 'fitness', 'diet'],
            'entertainment': ['movie', 'music', 'game', 'tv', 'show', 'fun', 'party', 'concert'],
            'weather': ['weather', 'rain', 'sunny', 'cloud', 'temperature', 'hot', 'cold'],
            'family': ['family', 'parent', 'child', 'brother', 'sister', 'mother', 'father'],
            'hobbies': ['hobby', 'sport', 'reading', 'painting', 'photography', 'gardening']
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics if detected_topics else ['general']
    
    def detect_user_emotion(self, user_input: str) -> str:
        """사용자 감정 감지"""
        # 간단한 감정 분석 (실제로는 감정 분석 모델 사용)
        
        positive_words = ['happy', 'good', 'great', 'excellent', 'wonderful', 'love', 'like', 'enjoy']
        negative_words = ['sad', 'bad', 'terrible', 'awful', 'hate', 'dislike', 'difficult', 'hard']
        excited_words = ['excited', 'amazing', 'fantastic', 'awesome', 'incredible', 'wow']
        confused_words = ['confused', 'don\'t understand', 'unclear', 'difficult', 'complicated']
        
        text_lower = user_input.lower()
        
        if any(word in text_lower for word in excited_words):
            return 'excited'
        elif any(word in text_lower for word in positive_words):
            return 'positive'
        elif any(word in text_lower for word in negative_words):
            return 'negative'
        elif any(word in text_lower for word in confused_words):
            return 'confused'
        else:
            return 'neutral'
    
    async def assess_conversation_quality(self, user_input: str, ai_response: Dict) -> float:
        """대화 품질 평가"""
        
        # 사용자 입력 길이
        input_length = len(user_input.split())
        length_score = min(1.0, input_length / 10)  # 10단어 이상이면 만점
        
        # AI 응답과의 관련성 (간단한 키워드 매칭)
        input_words = set(user_input.lower().split())
        response_words = set(ai_response.get('text', '').lower().split())
        relevance_score = len(input_words & response_words) / max(len(input_words), 1)
        
        # 종합 점수
        quality_score = (length_score * 0.4 + relevance_score * 0.6)
        
        return min(1.0, quality_score)
    
    async def update_context(self, user_id: str, conversation_entry: Dict):
        """컨텍스트 업데이트"""
        
        context = self.user_contexts[user_id]
        context['last_active'] = datetime.now()
        context['total_interactions'] += 1
        
        # 주제 관심도 업데이트
        if user_id not in self.topic_interests:
            self.topic_interests[user_id] = {}
        
        for topic in conversation_entry.get('topics_discussed', []):
            if topic in self.topic_interests[user_id]:
                self.topic_interests[user_id][topic]['interactions'] += 1
                # 관심도는 상호작용 빈도와 최근성을 고려하여 조정
                self.topic_interests[user_id][topic]['interest_level'] = min(1.0, 
                    self.topic_interests[user_id][topic]['interest_level'] + 0.1)
            else:
                self.topic_interests[user_id][topic] = {
                    'area': topic,
                    'interest_level': 0.7,
                    'interactions': 1,
                    'first_interaction': datetime.now(),
                    'last_interaction': datetime.now()
                }
    
    async def update_learning_patterns(self, user_id: str, conversation_entry: Dict):
        """학습 패턴 업데이트"""
        
        # 오류 패턴 분석 및 학습 성향 파악
        errors = conversation_entry.get('errors', [])
        user_emotion = conversation_entry.get('user_emotion', 'neutral')
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = await self.get_learning_preferences(user_id)
        
        preferences = self.user_preferences[user_id]
        
        # 감정 상태에 따른 학습 선호도 조정
        if user_emotion == 'confused' and errors:
            preferences['correction_style'] = 'gentle'
            preferences['difficulty_preference'] = 'easier'
        elif user_emotion == 'excited' and not errors:
            preferences['difficulty_preference'] = 'adaptive'
    
    async def generate_context_summary(self, user_id: str) -> str:
        """컨텍스트 요약 생성"""
        
        recent_topics = await self.get_recent_topics(user_id, days=3)
        preferences = await self.get_learning_preferences(user_id)
        
        if not recent_topics:
            return f"New learner with {preferences['learning_style']} learning style preference."
        
        top_topics = [topic['topic'] for topic in recent_topics[:3]]
        topic_str = ', '.join(top_topics)
        
        summary = f"Recently discussed: {topic_str}. "
        summary += f"Learning style: {preferences['learning_style']}. "
        summary += f"Prefers {preferences['correction_style']} corrections."
        
        return summary
    
    def calculate_topic_relevance(self, topic: str, frequency: int, days: int) -> float:
        """주제 관련성 점수 계산"""
        # 빈도와 최신성을 고려한 관련성 점수
        frequency_score = min(1.0, frequency / 5)  # 5회 이상이면 만점
        recency_score = max(0.1, 1.0 - (days / 30))  # 30일 이내는 높은 점수
        
        return (frequency_score * 0.7 + recency_score * 0.3)
    
    def calculate_improvement_trend(self, errors: List[Dict]) -> str:
        """개선 추세 계산"""
        if len(errors) < 2:
            return 'insufficient_data'
        
        # 시간순으로 정렬
        errors.sort(key=lambda x: x['timestamp'])
        
        recent_errors = errors[-3:]  # 최근 3개
        older_errors = errors[:-3] if len(errors) > 3 else []
        
        if not older_errors:
            return 'new_pattern'
        
        recent_avg_time = sum((datetime.now() - e['timestamp']).days for e in recent_errors) / len(recent_errors)
        older_avg_time = sum((datetime.now() - e['timestamp']).days for e in older_errors) / len(older_errors)
        
        if recent_avg_time < older_avg_time:
            return 'improving'
        elif recent_avg_time > older_avg_time:
            return 'worsening'
        else:
            return 'stable'
    
    def get_error_recommendation(self, error_type: str) -> str:
        """오류 유형별 추천"""
        recommendations = {
            'grammar': 'Practice basic sentence structures and verb tenses',
            'vocabulary': 'Build vocabulary through reading and flashcards',
            'pronunciation': 'Listen to native speakers and practice shadowing',
            'spelling': 'Use spell-check tools and read more frequently',
            'word_order': 'Study English sentence patterns and word order rules'
        }
        
        return recommendations.get(error_type, 'Continue practicing and focus on this area')
    
    async def compress_old_conversations(self, user_id: str):
        """오래된 대화 압축"""
        conversations = self.conversation_histories[user_id]
        
        # 오래된 대화들을 요약으로 변환
        cutoff_index = len(conversations) - self.max_conversation_history + 20  # 20개 여유분
        old_conversations = conversations[:cutoff_index]
        
        # 요약 생성
        summary = {
            'period': f"{old_conversations[0]['timestamp']} to {old_conversations[-1]['timestamp']}",
            'total_conversations': len(old_conversations),
            'main_topics': await self.summarize_topics(old_conversations),
            'common_errors': await self.summarize_errors(old_conversations),
            'learning_progress': await self.summarize_progress(old_conversations)
        }
        
        # 압축된 요약을 별도 저장소에 보관
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = {}
        
        if 'conversation_summaries' not in self.user_contexts[user_id]:
            self.user_contexts[user_id]['conversation_summaries'] = []
        
        self.user_contexts[user_id]['conversation_summaries'].append(summary)
        
        # 오래된 대화 삭제
        self.conversation_histories[user_id] = conversations[cutoff_index:]
    
    async def summarize_topics(self, conversations: List[Dict]) -> Dict:
        """주제 요약"""
        topic_count = defaultdict(int)
        for conv in conversations:
            for topic in conv.get('topics_discussed', []):
                topic_count[topic] += 1
        
        return dict(sorted(topic_count.items(), key=lambda x: x[1], reverse=True)[:5])
    
    async def summarize_errors(self, conversations: List[Dict]) -> Dict:
        """오류 요약"""
        error_count = defaultdict(int)
        for conv in conversations:
            for error in conv.get('errors', []):
                error_count[error.get('type', 'general')] += 1
        
        return dict(sorted(error_count.items(), key=lambda x: x[1], reverse=True))
    
    async def summarize_progress(self, conversations: List[Dict]) -> Dict:
        """진행 상황 요약"""
        total_errors = sum(len(conv.get('errors', [])) for conv in conversations)
        avg_quality = sum(conv.get('conversation_quality', 0.7) for conv in conversations) / len(conversations)
        
        return {
            'total_interactions': len(conversations),
            'average_errors_per_conversation': round(total_errors / len(conversations), 2),
            'average_conversation_quality': round(avg_quality, 2)
        }
    
    async def get_session_history(self, user_id: str, session_id: str) -> List[Dict]:
        """세션별 대화 기록 조회"""
        # 실제 구현에서는 session_id로 필터링
        return self.conversation_histories.get(user_id, [])[-10:]  # 최근 10개 대화