"""
목표 에이전트 - 학습 목표 설정 및 진도 관리
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LearningGoal:
    id: str
    user_id: str
    goal_type: str  # 'vocabulary', 'grammar', 'speaking', 'listening', 'custom'
    title: str
    description: str
    target_value: int
    current_progress: int
    deadline: datetime
    created_at: datetime
    priority: str  # 'high', 'medium', 'low'
    status: str  # 'active', 'completed', 'paused', 'expired'
    milestones: List[Dict]
    completion_criteria: Dict

class GoalAgent:
    def __init__(self):
        # 사용자별 목표 데이터 (실제로는 데이터베이스 사용)
        self.user_goals = {}
        self.goal_templates = self.initialize_goal_templates()
        
    async def initialize(self):
        """목표 에이전트 초기화"""
        logger.info("Goal Agent initialized")
    
    def initialize_goal_templates(self) -> Dict:
        """목표 템플릿 초기화"""
        return {
            'beginner': {
                'vocabulary': {
                    'title': 'Build Basic Vocabulary',
                    'description': 'Learn and master 500 essential English words',
                    'target_value': 500,
                    'duration_days': 60,
                    'milestones': [
                        {'target': 100, 'reward': 'vocabulary_certificate_1'},
                        {'target': 250, 'reward': 'vocabulary_certificate_2'},
                        {'target': 500, 'reward': 'vocabulary_master_badge'}
                    ]
                },
                'conversation': {
                    'title': 'Master Basic Conversations',
                    'description': 'Complete 30 successful basic conversations',
                    'target_value': 30,
                    'duration_days': 45,
                    'milestones': [
                        {'target': 10, 'reward': 'conversation_starter'},
                        {'target': 20, 'reward': 'confident_speaker'},
                        {'target': 30, 'reward': 'conversation_master'}
                    ]
                },
                'pronunciation': {
                    'title': 'Improve Pronunciation',
                    'description': 'Achieve 80% accuracy in pronunciation exercises',
                    'target_value': 80,
                    'duration_days': 30,
                    'milestones': [
                        {'target': 60, 'reward': 'pronunciation_improver'},
                        {'target': 70, 'reward': 'clear_speaker'},
                        {'target': 80, 'reward': 'pronunciation_master'}
                    ]
                }
            },
            'intermediate': {
                'vocabulary': {
                    'title': 'Expand Advanced Vocabulary',
                    'description': 'Learn 1500+ intermediate-level words',
                    'target_value': 1500,
                    'duration_days': 90,
                    'milestones': [
                        {'target': 500, 'reward': 'vocabulary_expander'},
                        {'target': 1000, 'reward': 'word_collector'},
                        {'target': 1500, 'reward': 'vocabulary_expert'}
                    ]
                },
                'fluency': {
                    'title': 'Achieve Speaking Fluency',
                    'description': 'Maintain 60+ conversations without major errors',
                    'target_value': 60,
                    'duration_days': 75,
                    'milestones': [
                        {'target': 20, 'reward': 'fluency_beginner'},
                        {'target': 40, 'reward': 'smooth_talker'},
                        {'target': 60, 'reward': 'fluent_speaker'}
                    ]
                },
                'grammar': {
                    'title': 'Master Complex Grammar',
                    'description': 'Achieve 90% accuracy in complex grammar usage',
                    'target_value': 90,
                    'duration_days': 60,
                    'milestones': [
                        {'target': 70, 'reward': 'grammar_student'},
                        {'target': 80, 'reward': 'grammar_practitioner'},
                        {'target': 90, 'reward': 'grammar_master'}
                    ]
                }
            },
            'advanced': {
                'nuance': {
                    'title': 'Master Language Nuances',
                    'description': 'Successfully use idioms and cultural expressions in 50 conversations',
                    'target_value': 50,
                    'duration_days': 90,
                    'milestones': [
                        {'target': 15, 'reward': 'nuance_explorer'},
                        {'target': 30, 'reward': 'cultural_communicator'},
                        {'target': 50, 'reward': 'native_like_speaker'}
                    ]
                },
                'professional': {
                    'title': 'Professional Communication',
                    'description': 'Complete 25 business-level conversations successfully',
                    'target_value': 25,
                    'duration_days': 60,
                    'milestones': [
                        {'target': 8, 'reward': 'business_communicator'},
                        {'target': 15, 'reward': 'professional_speaker'},
                        {'target': 25, 'reward': 'business_expert'}
                    ]
                },
                'teaching': {
                    'title': 'Language Teaching Skills',
                    'description': 'Help other learners in 20 peer-teaching sessions',
                    'target_value': 20,
                    'duration_days': 75,
                    'milestones': [
                        {'target': 5, 'reward': 'helpful_peer'},
                        {'target': 12, 'reward': 'peer_mentor'},
                        {'target': 20, 'reward': 'language_coach'}
                    ]
                }
            }
        }
    
    async def set_user_goals(self, user_id: str, user_level: str, preferences: Dict) -> List[LearningGoal]:
        """사용자 목표 설정"""
        
        if user_id not in self.user_goals:
            self.user_goals[user_id] = []
        
        # 기존 목표가 있으면 상태 업데이트
        await self.update_existing_goals_status(user_id)
        
        # 레벨별 추천 목표 생성
        recommended_goals = await self.generate_recommended_goals(user_id, user_level, preferences)
        
        # 사용자별 맞춤 목표 추가
        if preferences.get('focus_areas'):
            custom_goals = await self.create_custom_goals(user_id, preferences['focus_areas'])
            recommended_goals.extend(custom_goals)
        
        # 목표 저장
        for goal in recommended_goals:
            self.user_goals[user_id].append(goal)
        
        logger.info(f"Set {len(recommended_goals)} goals for user {user_id}")
        
        return recommended_goals
    
    async def generate_recommended_goals(
        self,
        user_id: str,
        user_level: str,
        preferences: Dict
    ) -> List[LearningGoal]:
        """추천 목표 생성"""
        
        level_templates = self.goal_templates.get(user_level, self.goal_templates['intermediate'])
        goals = []
        
        # 우선순위가 높은 목표들
        priority_areas = preferences.get('priority_areas', ['vocabulary', 'conversation'])
        
        for area in priority_areas:
            if area in level_templates:
                template = level_templates[area]
                goal = self.create_goal_from_template(
                    user_id=user_id,
                    goal_type=area,
                    template=template,
                    priority='high'
                )
                goals.append(goal)
        
        # 추가 목표 (우선순위 medium)
        remaining_areas = [area for area in level_templates.keys() if area not in priority_areas]
        for area in remaining_areas[:2]:  # 최대 2개 추가
            template = level_templates[area]
            goal = self.create_goal_from_template(
                user_id=user_id,
                goal_type=area,
                template=template,
                priority='medium'
            )
            goals.append(goal)
        
        return goals
    
    def create_goal_from_template(
        self,
        user_id: str,
        goal_type: str,
        template: Dict,
        priority: str = 'medium'
    ) -> LearningGoal:
        """템플릿으로부터 목표 생성"""
        
        goal_id = f"{user_id}_{goal_type}_{datetime.now().timestamp()}"
        deadline = datetime.now() + timedelta(days=template['duration_days'])
        
        return LearningGoal(
            id=goal_id,
            user_id=user_id,
            goal_type=goal_type,
            title=template['title'],
            description=template['description'],
            target_value=template['target_value'],
            current_progress=0,
            deadline=deadline,
            created_at=datetime.now(),
            priority=priority,
            status='active',
            milestones=template.get('milestones', []),
            completion_criteria={
                'type': 'target_value',
                'value': template['target_value'],
                'measurement': self.get_measurement_type(goal_type)
            }
        )
    
    def get_measurement_type(self, goal_type: str) -> str:
        """목표 유형별 측정 방법"""
        measurement_types = {
            'vocabulary': 'words_learned',
            'conversation': 'successful_conversations',
            'grammar': 'accuracy_percentage',
            'pronunciation': 'accuracy_percentage',
            'fluency': 'conversation_count',
            'nuance': 'advanced_expressions_used',
            'professional': 'business_conversations',
            'teaching': 'peer_sessions_completed'
        }
        
        return measurement_types.get(goal_type, 'count')
    
    async def update_progress(self, user_id: str, user_input: str, ai_response: Dict):
        """사용자 진도 업데이트"""
        
        if user_id not in self.user_goals:
            return
        
        active_goals = [goal for goal in self.user_goals[user_id] if goal.status == 'active']
        
        for goal in active_goals:
            progress_update = await self.calculate_progress_update(
                goal, user_input, ai_response
            )
            
            if progress_update > 0:
                old_progress = goal.current_progress
                goal.current_progress += progress_update
                
                # 목표 달성 확인
                if goal.current_progress >= goal.target_value:
                    goal.status = 'completed'
                    await self.handle_goal_completion(goal)
                
                # 마일스톤 달성 확인
                await self.check_milestone_achievements(goal, old_progress)
                
                logger.info(f"Updated progress for goal {goal.id}: +{progress_update}")
    
    async def calculate_progress_update(
        self,
        goal: LearningGoal,
        user_input: str,
        ai_response: Dict
    ) -> int:
        """진도 업데이트 계산"""
        
        progress = 0
        
        if goal.goal_type == 'vocabulary':
            # 새로운 어휘 사용 확인
            vocabulary_used = ai_response.get('vocabulary_suggestions', [])
            progress = len(vocabulary_used)
            
        elif goal.goal_type == 'conversation':
            # 성공적인 대화인지 확인
            errors = ai_response.get('errors', [])
            if len(errors) <= 2 and len(user_input.split()) >= 5:  # 기준: 오류 2개 이하, 5단어 이상
                progress = 1
                
        elif goal.goal_type == 'grammar':
            # 문법 정확도 기반
            errors = ai_response.get('errors', [])
            words_count = len(user_input.split())
            if words_count > 0:
                accuracy = max(0, 100 - (len(errors) / words_count * 100))
                progress = int(accuracy)  # 정확도를 진도로 사용
                
        elif goal.goal_type == 'pronunciation':
            # 발음 관련 피드백 확인
            pronunciation_tips = ai_response.get('pronunciation_tips', [])
            if not pronunciation_tips:  # 발음 지적이 없으면 좋은 발음
                progress = 5
                
        elif goal.goal_type == 'fluency':
            # 대화 연속성 확인
            if len(user_input.split()) >= 10 and len(ai_response.get('errors', [])) <= 1:
                progress = 1
                
        elif goal.goal_type == 'nuance':
            # 고급 표현 사용 확인
            advanced_expressions = ['however', 'nevertheless', 'furthermore', 'consequently']
            used_expressions = [expr for expr in advanced_expressions if expr in user_input.lower()]
            progress = len(used_expressions)
        
        return progress
    
    async def check_milestone_achievements(self, goal: LearningGoal, old_progress: int):
        """마일스톤 달성 확인"""
        
        for milestone in goal.milestones:
            target = milestone['target']
            
            # 새로 달성한 마일스톤인지 확인
            if old_progress < target <= goal.current_progress:
                await self.handle_milestone_achievement(goal, milestone)
    
    async def handle_milestone_achievement(self, goal: LearningGoal, milestone: Dict):
        """마일스톤 달성 처리"""
        
        reward = milestone.get('reward', 'achievement_badge')
        
        # 보상 지급 로직
        await self.award_achievement(goal.user_id, reward, milestone['target'])
        
        logger.info(f"Milestone achieved: {goal.user_id} reached {milestone['target']} in {goal.goal_type}")
    
    async def handle_goal_completion(self, goal: LearningGoal):
        """목표 완료 처리"""
        
        # 완료 보상
        completion_reward = f"{goal.goal_type}_master"
        await self.award_achievement(goal.user_id, completion_reward, goal.target_value)
        
        # 다음 레벨 목표 제안
        next_goals = await self.suggest_next_level_goals(goal)
        
        logger.info(f"Goal completed: {goal.user_id} completed {goal.title}")
    
    async def award_achievement(self, user_id: str, achievement_type: str, value: int):
        """업적 보상 지급"""
        # 실제로는 사용자 프로필에 업적 추가
        logger.info(f"Achievement awarded to {user_id}: {achievement_type} (value: {value})")
    
    async def get_progress(self, user_id: str) -> Dict:
        """사용자 진도 조회"""
        
        if user_id not in self.user_goals:
            return {
                'active_goals': [],
                'completed_goals': [],
                'overall_progress': 0.0,
                'achievements': [],
                'next_milestones': []
            }
        
        user_goals = self.user_goals[user_id]
        
        active_goals = [goal for goal in user_goals if goal.status == 'active']
        completed_goals = [goal for goal in user_goals if goal.status == 'completed']
        
        # 전체 진도율 계산
        total_progress = 0
        if active_goals:
            for goal in active_goals:
                goal_progress = min(100, (goal.current_progress / goal.target_value) * 100)
                total_progress += goal_progress
            total_progress = total_progress / len(active_goals)
        
        # 다음 마일스톤
        next_milestones = []
        for goal in active_goals:
            next_milestone = self.get_next_milestone(goal)
            if next_milestone:
                next_milestones.append({
                    'goal_title': goal.title,
                    'milestone_target': next_milestone['target'],
                    'current_progress': goal.current_progress,
                    'progress_needed': next_milestone['target'] - goal.current_progress
                })
        
        return {
            'active_goals': [self.goal_to_dict(goal) for goal in active_goals],
            'completed_goals': [self.goal_to_dict(goal) for goal in completed_goals],
            'overall_progress': round(total_progress, 1),
            'achievements': await self.get_user_achievements(user_id),
            'next_milestones': next_milestones[:3],  # 상위 3개
            'recommendations': await self.generate_progress_recommendations(user_id)
        }
    
    def get_next_milestone(self, goal: LearningGoal) -> Optional[Dict]:
        """다음 마일스톤 조회"""
        
        for milestone in goal.milestones:
            if milestone['target'] > goal.current_progress:
                return milestone
        
        return None
    
    async def get_recommendations(self, user_id: str) -> List[Dict]:
        """목표 기반 추천"""
        
        if user_id not in self.user_goals:
            return []
        
        active_goals = [goal for goal in self.user_goals[user_id] if goal.status == 'active']
        recommendations = []
        
        for goal in active_goals:
            # 진도가 느린 목표 식별
            progress_ratio = goal.current_progress / goal.target_value
            days_passed = (datetime.now() - goal.created_at).days
            expected_days = (goal.deadline - goal.created_at).days
            expected_progress = days_passed / expected_days if expected_days > 0 else 0
            
            if progress_ratio < expected_progress * 0.8:  # 예상보다 20% 이하 진도
                recommendations.append({
                    'type': 'catch_up',
                    'goal_title': goal.title,
                    'message': f"You're behind schedule on '{goal.title}'. Consider focusing more time on this goal.",
                    'suggested_activities': self.get_goal_activities(goal.goal_type),
                    'priority': 'high'
                })
            
            # 마일스톤 근접 알림
            next_milestone = self.get_next_milestone(goal)
            if next_milestone:
                remaining = next_milestone['target'] - goal.current_progress
                if remaining <= 5:  # 5 단위 이내
                    recommendations.append({
                        'type': 'milestone_close',
                        'goal_title': goal.title,
                        'message': f"You're close to achieving a milestone in '{goal.title}'! Just {remaining} more to go.",
                        'suggested_activities': self.get_goal_activities(goal.goal_type),
                        'priority': 'medium'
                    })
        
        return recommendations[:5]  # 상위 5개 추천
    
    def get_goal_activities(self, goal_type: str) -> List[str]:
        """목표 유형별 활동 추천"""
        activities = {
            'vocabulary': [
                'Practice new words in conversations',
                'Use flashcard exercises',
                'Read articles and note new words'
            ],
            'conversation': [
                'Engage in longer discussions',
                'Practice with different topics',
                'Focus on clear expression'
            ],
            'grammar': [
                'Review grammar rules',
                'Practice sentence construction',
                'Get correction feedback'
            ],
            'pronunciation': [
                'Practice shadowing exercises',
                'Record and compare pronunciation',
                'Focus on difficult sounds'
            ],
            'fluency': [
                'Practice speaking without pauses',
                'Engage in rapid response exercises',
                'Focus on natural conversation flow'
            ]
        }
        
        return activities.get(goal_type, ['Continue regular practice'])
    
    async def update_goals(self, user_id: str, feedback: Dict):
        """피드백을 기반으로 목표 업데이트"""
        
        if user_id not in self.user_goals:
            return
        
        # 성능에 따른 목표 조정
        performance_score = feedback.get('performance_score', 70)
        difficulty_rating = feedback.get('difficulty_rating', 3)
        
        active_goals = [goal for goal in self.user_goals[user_id] if goal.status == 'active']
        
        for goal in active_goals:
            # 목표가 너무 어려운 경우 조정
            if performance_score < 60 and difficulty_rating >= 4:
                # 목표 값을 20% 감소
                goal.target_value = int(goal.target_value * 0.8)
                goal.deadline = goal.deadline + timedelta(days=14)  # 2주 연장
                
            # 목표가 너무 쉬운 경우 조정
            elif performance_score > 85 and difficulty_rating <= 2:
                # 목표 값을 15% 증가
                goal.target_value = int(goal.target_value * 1.15)
    
    async def create_custom_goals(self, user_id: str, focus_areas: List[str]) -> List[LearningGoal]:
        """맞춤 목표 생성"""
        
        custom_goals = []
        
        for area in focus_areas:
            if area not in ['vocabulary', 'grammar', 'conversation', 'pronunciation']:
                # 사용자 정의 영역
                goal_id = f"{user_id}_custom_{area}_{datetime.now().timestamp()}"
                
                custom_goal = LearningGoal(
                    id=goal_id,
                    user_id=user_id,
                    goal_type='custom',
                    title=f"Improve {area.title()}",
                    description=f"Focus on improving {area} skills through targeted practice",
                    target_value=20,  # 기본값
                    current_progress=0,
                    deadline=datetime.now() + timedelta(days=45),
                    created_at=datetime.now(),
                    priority='medium',
                    status='active',
                    milestones=[
                        {'target': 7, 'reward': f'{area}_beginner'},
                        {'target': 14, 'reward': f'{area}_intermediate'},
                        {'target': 20, 'reward': f'{area}_advanced'}
                    ],
                    completion_criteria={
                        'type': 'target_value',
                        'value': 20,
                        'measurement': 'custom_activities'
                    }
                )
                
                custom_goals.append(custom_goal)
        
        return custom_goals
    
    async def suggest_next_level_goals(self, completed_goal: LearningGoal) -> List[Dict]:
        """다음 레벨 목표 제안"""
        
        suggestions = []
        goal_type = completed_goal.goal_type
        
        # 상위 레벨 목표 제안
        if goal_type == 'vocabulary':
            if completed_goal.target_value <= 500:  # beginner 완료
                suggestions.append({
                    'type': 'vocabulary',
                    'title': 'Advanced Vocabulary Building',
                    'target': 1500,
                    'level': 'intermediate'
                })
            else:  # intermediate 완료
                suggestions.append({
                    'type': 'vocabulary',
                    'title': 'Professional Vocabulary Mastery',
                    'target': 3000,
                    'level': 'advanced'
                })
        
        elif goal_type == 'conversation':
            suggestions.append({
                'type': 'fluency',
                'title': 'Speaking Fluency Development',
                'target': 60,
                'level': 'intermediate'
            })
        
        return suggestions
    
    def goal_to_dict(self, goal: LearningGoal) -> Dict:
        """목표를 딕셔너리로 변환"""
        
        progress_percentage = min(100, (goal.current_progress / goal.target_value) * 100)
        
        return {
            'id': goal.id,
            'type': goal.goal_type,
            'title': goal.title,
            'description': goal.description,
            'current_progress': goal.current_progress,
            'target_value': goal.target_value,
            'progress_percentage': round(progress_percentage, 1),
            'deadline': goal.deadline.isoformat(),
            'priority': goal.priority,
            'status': goal.status,
            'next_milestone': self.get_next_milestone(goal),
            'days_remaining': (goal.deadline - datetime.now()).days
        }
    
    async def get_user_achievements(self, user_id: str) -> List[Dict]:
        """사용자 업적 조회"""
        # 실제로는 별도 테이블에서 조회
        achievements = [
            {
                'type': 'vocabulary_learner',
                'title': 'Word Collector',
                'description': 'Learned 100 new words',
                'earned_date': datetime.now() - timedelta(days=5),
                'badge_color': 'gold'
            }
        ]
        
        return achievements
    
    async def generate_progress_recommendations(self, user_id: str) -> List[str]:
        """진도 기반 추천 생성"""
        
        progress_data = await self.get_progress(user_id)
        recommendations = []
        
        overall_progress = progress_data['overall_progress']
        
        if overall_progress < 30:
            recommendations.append("Set aside more time for daily English practice")
            recommendations.append("Focus on completing easier goals first")
        elif overall_progress < 70:
            recommendations.append("You're making good progress! Keep up the consistency")
            recommendations.append("Consider setting slightly more challenging goals")
        else:
            recommendations.append("Excellent progress! You're doing great")
            recommendations.append("Ready for more advanced challenges")
        
        # 활성 목표가 많은 경우
        if len(progress_data['active_goals']) > 4:
            recommendations.append("Consider focusing on fewer goals for better results")
        
        return recommendations[:3]
    
    async def update_existing_goals_status(self, user_id: str):
        """기존 목표 상태 업데이트"""
        
        if user_id not in self.user_goals:
            return
        
        current_time = datetime.now()
        
        for goal in self.user_goals[user_id]:
            # 만료된 목표 처리
            if goal.status == 'active' and goal.deadline < current_time:
                goal.status = 'expired'
                logger.info(f"Goal expired: {goal.title} for user {user_id}")