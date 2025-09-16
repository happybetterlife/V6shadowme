"""
레벨 매니저 에이전트 - 사용자 영어 레벨 관리
"""

import json
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LevelManagerAgent:
    def __init__(self):
        self.level_criteria = {
            'beginner': {
                'vocabulary_range': (0, 1000),
                'grammar_accuracy': 0.6,
                'sentence_complexity': 'simple',
                'conversation_flow': 'basic',
                'description': 'Basic words and simple sentences'
            },
            'intermediate': {
                'vocabulary_range': (1000, 3000),
                'grammar_accuracy': 0.75,
                'sentence_complexity': 'moderate',
                'conversation_flow': 'good',
                'description': 'Good vocabulary and complex sentences'
            },
            'advanced': {
                'vocabulary_range': (3000, 6000),
                'grammar_accuracy': 0.85,
                'sentence_complexity': 'complex',
                'conversation_flow': 'natural',
                'description': 'Rich vocabulary and natural conversation'
            }
        }
        
        # 사용자 레벨 데이터 (실제로는 데이터베이스에서 관리)
        self.user_levels = {}
        
    async def initialize(self):
        """에이전트 초기화"""
        logger.info("Level Manager Agent initialized")
    
    async def get_level(self, user_id: str) -> str:
        """사용자 현재 레벨 조회"""
        if user_id not in self.user_levels:
            # 기본 레벨 설정
            self.user_levels[user_id] = {
                'current_level': 'intermediate',
                'target_level': 'advanced',
                'assessment_history': [],
                'last_updated': datetime.now()
            }
        
        return self.user_levels[user_id]['current_level']
    
    async def assess_level(self, user_id: str, conversation_data: Dict) -> Dict:
        """대화 데이터를 기반으로 레벨 평가"""
        
        # 어휘 분석
        vocabulary_score = await self.analyze_vocabulary(conversation_data)
        
        # 문법 정확도 분석
        grammar_score = await self.analyze_grammar(conversation_data)
        
        # 문장 복잡도 분석
        complexity_score = await self.analyze_sentence_complexity(conversation_data)
        
        # 대화 흐름 분석
        flow_score = await self.analyze_conversation_flow(conversation_data)
        
        # 종합 점수 계산
        overall_score = (
            vocabulary_score * 0.3 +
            grammar_score * 0.3 +
            complexity_score * 0.2 +
            flow_score * 0.2
        )
        
        # 레벨 결정
        suggested_level = self.determine_level_from_score(overall_score)
        
        assessment = {
            'user_id': user_id,
            'timestamp': datetime.now(),
            'scores': {
                'vocabulary': vocabulary_score,
                'grammar': grammar_score,
                'complexity': complexity_score,
                'flow': flow_score,
                'overall': overall_score
            },
            'current_level': await self.get_level(user_id),
            'suggested_level': suggested_level,
            'assessment_details': {
                'vocabulary_range': self.estimate_vocabulary_range(conversation_data),
                'common_errors': conversation_data.get('errors', []),
                'strengths': self.identify_strengths(vocabulary_score, grammar_score, complexity_score, flow_score),
                'areas_for_improvement': self.identify_improvement_areas(vocabulary_score, grammar_score, complexity_score, flow_score)
            }
        }
        
        # 평가 히스토리 저장
        if user_id not in self.user_levels:
            self.user_levels[user_id] = {'assessment_history': []}
        
        self.user_levels[user_id]['assessment_history'].append(assessment)
        
        return assessment
    
    async def update_level(self, user_id: str, feedback: Dict):
        """피드백을 기반으로 레벨 업데이트"""
        
        current_level = await self.get_level(user_id)
        performance_score = feedback.get('performance_score', 0)
        difficulty_rating = feedback.get('difficulty_rating', 3)
        
        # 레벨 조정 로직
        level_change = None
        
        if performance_score >= 85 and difficulty_rating <= 2:
            # 레벨 업 고려
            if current_level == 'beginner':
                level_change = 'intermediate'
            elif current_level == 'intermediate':
                level_change = 'advanced'
        elif performance_score <= 50 and difficulty_rating >= 4:
            # 레벨 다운 고려
            if current_level == 'advanced':
                level_change = 'intermediate'
            elif current_level == 'intermediate':
                level_change = 'beginner'
        
        if level_change and level_change != current_level:
            self.user_levels[user_id]['current_level'] = level_change
            self.user_levels[user_id]['last_updated'] = datetime.now()
            
            logger.info(f"User {user_id} level updated from {current_level} to {level_change}")
            
            return {
                'level_changed': True,
                'old_level': current_level,
                'new_level': level_change,
                'reason': f"Performance: {performance_score}%, Difficulty: {difficulty_rating}/5"
            }
        
        return {'level_changed': False, 'current_level': current_level}
    
    async def get_recommendations(self, user_id: str) -> List[Dict]:
        """레벨별 학습 추천"""
        
        current_level = await self.get_level(user_id)
        user_data = self.user_levels.get(user_id, {})
        
        recommendations = []
        
        if current_level == 'beginner':
            recommendations = [
                {
                    'type': 'vocabulary',
                    'title': 'Build Basic Vocabulary',
                    'description': 'Focus on learning 500 most common English words',
                    'activities': ['flashcards', 'word games', 'simple sentences']
                },
                {
                    'type': 'grammar',
                    'title': 'Master Basic Grammar',
                    'description': 'Learn present tense and basic sentence structure',
                    'activities': ['grammar exercises', 'pattern practice', 'simple conversations']
                },
                {
                    'type': 'speaking',
                    'title': 'Practice Basic Conversations',
                    'description': 'Start with greetings and everyday topics',
                    'activities': ['role play', 'shadowing', 'basic Q&A']
                }
            ]
        
        elif current_level == 'intermediate':
            recommendations = [
                {
                    'type': 'vocabulary',
                    'title': 'Expand Vocabulary Range',
                    'description': 'Learn 2000+ words including academic vocabulary',
                    'activities': ['reading practice', 'context learning', 'synonym practice']
                },
                {
                    'type': 'grammar',
                    'title': 'Complex Sentence Structures',
                    'description': 'Master past/future tenses and conditional sentences',
                    'activities': ['advanced grammar', 'writing practice', 'error correction']
                },
                {
                    'type': 'fluency',
                    'title': 'Improve Speaking Fluency',
                    'description': 'Practice longer conversations and express opinions',
                    'activities': ['debate practice', 'presentation skills', 'discussion topics']
                }
            ]
        
        else:  # advanced
            recommendations = [
                {
                    'type': 'nuance',
                    'title': 'Master Language Nuances',
                    'description': 'Learn idiomatic expressions and cultural contexts',
                    'activities': ['idiom practice', 'cultural discussions', 'advanced reading']
                },
                {
                    'type': 'specialization',
                    'title': 'Specialized Vocabulary',
                    'description': 'Focus on professional or academic English',
                    'activities': ['business English', 'academic writing', 'technical discussions']
                },
                {
                    'type': 'native_like',
                    'title': 'Native-like Proficiency',
                    'description': 'Achieve natural speech patterns and pronunciation',
                    'activities': ['accent training', 'natural conversation', 'advanced topics']
                }
            ]
        
        # 개인화된 추천 추가
        assessment_history = user_data.get('assessment_history', [])
        if assessment_history:
            latest_assessment = assessment_history[-1]
            weak_areas = latest_assessment.get('assessment_details', {}).get('areas_for_improvement', [])
            
            for area in weak_areas:
                recommendations.append({
                    'type': 'improvement',
                    'title': f'Improve {area.title()}',
                    'description': f'Focus on strengthening your {area} skills',
                    'activities': self.get_improvement_activities(area),
                    'priority': 'high'
                })
        
        return recommendations[:5]  # 최대 5개 추천
    
    async def analyze_vocabulary(self, conversation_data: Dict) -> float:
        """어휘 수준 분석"""
        user_messages = conversation_data.get('user_messages', [])
        
        if not user_messages:
            return 0.5
        
        # 단어 다양성 계산
        all_words = []
        for message in user_messages:
            words = message.get('text', '').lower().split()
            all_words.extend(words)
        
        unique_words = len(set(all_words))
        total_words = len(all_words)
        
        if total_words == 0:
            return 0.5
        
        vocabulary_diversity = unique_words / total_words
        
        # 고급 어휘 사용 확인
        advanced_words = ['fascinating', 'significant', 'remarkable', 'consequently', 'nevertheless']
        intermediate_words = ['interesting', 'important', 'because', 'however', 'although']
        
        advanced_count = sum(1 for word in all_words if word in advanced_words)
        intermediate_count = sum(1 for word in all_words if word in intermediate_words)
        
        # 점수 계산 (0.0 - 1.0)
        score = min(1.0, vocabulary_diversity * 2 + (advanced_count * 0.1) + (intermediate_count * 0.05))
        
        return score
    
    async def analyze_grammar(self, conversation_data: Dict) -> float:
        """문법 정확도 분석"""
        errors = conversation_data.get('total_errors', 0)
        total_sentences = conversation_data.get('total_sentences', 1)
        
        accuracy = 1.0 - (errors / max(total_sentences, 1))
        return max(0.0, accuracy)
    
    async def analyze_sentence_complexity(self, conversation_data: Dict) -> float:
        """문장 복잡도 분석"""
        user_messages = conversation_data.get('user_messages', [])
        
        if not user_messages:
            return 0.5
        
        total_complexity = 0
        
        for message in user_messages:
            text = message.get('text', '')
            
            # 문장 길이
            sentence_length = len(text.split())
            
            # 복합문 지시자
            complex_indicators = ['because', 'although', 'however', 'therefore', 'meanwhile', 'furthermore']
            complexity_count = sum(1 for indicator in complex_indicators if indicator in text.lower())
            
            # 복잡도 점수 (길이 + 복합문 사용)
            complexity_score = min(1.0, (sentence_length / 20) + (complexity_count * 0.2))
            total_complexity += complexity_score
        
        return total_complexity / len(user_messages)
    
    async def analyze_conversation_flow(self, conversation_data: Dict) -> float:
        """대화 흐름 분석"""
        user_messages = conversation_data.get('user_messages', [])
        ai_responses = conversation_data.get('ai_responses', [])
        
        if len(user_messages) < 2:
            return 0.5
        
        # 응답 적절성 (길이 기반)
        appropriate_responses = 0
        
        for i, message in enumerate(user_messages):
            if i < len(ai_responses):
                user_length = len(message.get('text', '').split())
                
                # 적절한 길이의 응답인지 확인 (너무 짧지도 길지도 않은)
                if 3 <= user_length <= 25:
                    appropriate_responses += 1
        
        flow_score = appropriate_responses / len(user_messages)
        return flow_score
    
    def determine_level_from_score(self, score: float) -> str:
        """점수를 기반으로 레벨 결정"""
        if score >= 0.8:
            return 'advanced'
        elif score >= 0.6:
            return 'intermediate'
        else:
            return 'beginner'
    
    def estimate_vocabulary_range(self, conversation_data: Dict) -> tuple:
        """어휘 범위 추정"""
        # 간단한 추정 로직
        vocabulary_score = conversation_data.get('vocabulary_score', 0.5)
        
        if vocabulary_score >= 0.8:
            return (3000, 6000)
        elif vocabulary_score >= 0.6:
            return (1000, 3000)
        else:
            return (0, 1000)
    
    def identify_strengths(self, vocab: float, grammar: float, complexity: float, flow: float) -> List[str]:
        """강점 식별"""
        strengths = []
        scores = {
            'vocabulary': vocab,
            'grammar': grammar,
            'sentence complexity': complexity,
            'conversation flow': flow
        }
        
        # 0.7 이상인 영역을 강점으로 식별
        for area, score in scores.items():
            if score >= 0.7:
                strengths.append(area)
        
        return strengths
    
    def identify_improvement_areas(self, vocab: float, grammar: float, complexity: float, flow: float) -> List[str]:
        """개선 영역 식별"""
        improvement_areas = []
        scores = {
            'vocabulary': vocab,
            'grammar': grammar,
            'sentence complexity': complexity,
            'conversation flow': flow
        }
        
        # 0.6 미만인 영역을 개선 필요 영역으로 식별
        for area, score in scores.items():
            if score < 0.6:
                improvement_areas.append(area)
        
        return improvement_areas
    
    def get_improvement_activities(self, area: str) -> List[str]:
        """개선 영역별 활동 추천"""
        activities = {
            'vocabulary': ['flashcard practice', 'reading comprehension', 'word association games'],
            'grammar': ['grammar exercises', 'sentence correction', 'pattern drills'],
            'sentence complexity': ['complex sentence practice', 'writing exercises', 'advanced grammar'],
            'conversation flow': ['dialogue practice', 'turn-taking exercises', 'topic discussions']
        }
        
        return activities.get(area, ['general practice', 'focused exercises'])
    
    async def get_level_progress(self, user_id: str) -> Dict:
        """레벨 진행 상황 조회"""
        user_data = self.user_levels.get(user_id, {})
        current_level = user_data.get('current_level', 'intermediate')
        target_level = user_data.get('target_level', 'advanced')
        assessment_history = user_data.get('assessment_history', [])
        
        progress_data = {
            'current_level': current_level,
            'target_level': target_level,
            'progress_percentage': self.calculate_progress_percentage(current_level, target_level),
            'recent_assessments': assessment_history[-5:],  # 최근 5개
            'next_milestone': self.get_next_milestone(current_level, target_level)
        }
        
        return progress_data
    
    def calculate_progress_percentage(self, current: str, target: str) -> float:
        """진행률 계산"""
        level_order = ['beginner', 'intermediate', 'advanced']
        
        try:
            current_index = level_order.index(current)
            target_index = level_order.index(target)
            
            if current_index >= target_index:
                return 100.0
            
            progress = (current_index / target_index) * 100
            return round(progress, 1)
        except ValueError:
            return 50.0  # 기본값
    
    def get_next_milestone(self, current: str, target: str) -> str:
        """다음 목표 설정"""
        if current == target:
            return "Target level achieved! Consider setting a new goal."
        
        level_order = ['beginner', 'intermediate', 'advanced']
        
        try:
            current_index = level_order.index(current)
            if current_index < len(level_order) - 1:
                next_level = level_order[current_index + 1]
                return f"Progress to {next_level} level"
        except ValueError:
            pass
        
        return "Continue improving current level skills"