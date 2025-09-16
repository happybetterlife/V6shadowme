"""
트렌드 에이전트 - 실시간 트렌드 토픽 관리
"""

from typing import Dict, List
from datetime import datetime, timedelta
import logging
import random

logger = logging.getLogger(__name__)

class TrendAgent:
    def __init__(self):
        # 트렌드 데이터 캐시
        self.trending_topics = []
        self.topic_relevance_cache = {}
        self.last_update = None
        self.update_interval = timedelta(hours=6)  # 6시간마다 업데이트
        
    async def initialize(self):
        """트렌드 에이전트 초기화"""
        await self.update_trending_topics()
        logger.info("Trend Agent initialized")
    
    async def get_relevant_trends(
        self,
        user_input: str,
        context: Dict
    ) -> Dict:
        """사용자 입력과 관련된 트렌드 조회"""
        
        # 트렌드 데이터 업데이트 확인
        if self.should_update_trends():
            await self.update_trending_topics()
        
        # 사용자 입력에서 키워드 추출
        input_keywords = self.extract_keywords(user_input)
        
        # 컨텍스트에서 관심사 추출
        user_interests = context.get('interests', [])
        recent_topics = context.get('recent_topics', [])
        
        # 관련 트렌드 찾기
        relevant_trends = await self.find_relevant_trends(
            input_keywords,
            user_interests,
            recent_topics
        )
        
        return {
            'topics': relevant_trends,
            'trending_keywords': self.get_trending_keywords(),
            'conversation_starters': self.generate_trend_based_starters(relevant_trends),
            'educational_value': self.assess_educational_value(relevant_trends)
        }
    
    async def update_trending_topics(self):
        """트렌딩 토픽 업데이트"""
        
        # 실제 환경에서는 Google Trends API나 뉴스 API 사용
        # 여기서는 샘플 데이터 사용
        
        current_trends = [
            {
                'title': 'AI in Education',
                'category': 'technology',
                'keywords': ['artificial intelligence', 'learning', 'education', 'students'],
                'popularity': 0.9,
                'educational_value': 0.85,
                'difficulty_level': 'intermediate',
                'discussion_points': [
                    'How AI is changing classroom learning',
                    'Benefits and challenges of AI tutors',
                    'Future of personalized education'
                ]
            },
            {
                'title': 'Sustainable Living',
                'category': 'lifestyle',
                'keywords': ['environment', 'sustainability', 'green living', 'climate'],
                'popularity': 0.8,
                'educational_value': 0.9,
                'difficulty_level': 'intermediate',
                'discussion_points': [
                    'Simple ways to reduce environmental impact',
                    'Sustainable shopping habits',
                    'Renewable energy in daily life'
                ]
            },
            {
                'title': 'Remote Work Culture',
                'category': 'business',
                'keywords': ['remote work', 'work from home', 'productivity', 'team collaboration'],
                'popularity': 0.75,
                'educational_value': 0.8,
                'difficulty_level': 'intermediate',
                'discussion_points': [
                    'Advantages and disadvantages of remote work',
                    'Maintaining work-life balance',
                    'Future of office spaces'
                ]
            },
            {
                'title': 'Mental Health Awareness',
                'category': 'health',
                'keywords': ['mental health', 'wellness', 'stress management', 'self-care'],
                'popularity': 0.85,
                'educational_value': 0.9,
                'difficulty_level': 'beginner',
                'discussion_points': [
                    'Importance of mental health',
                    'Stress management techniques',
                    'Supporting others with mental health issues'
                ]
            },
            {
                'title': 'Space Exploration',
                'category': 'science',
                'keywords': ['space', 'mars', 'astronaut', 'NASA', 'SpaceX'],
                'popularity': 0.7,
                'educational_value': 0.85,
                'difficulty_level': 'advanced',
                'discussion_points': [
                    'Recent space missions and discoveries',
                    'Future of human space exploration',
                    'Technology innovations from space research'
                ]
            },
            {
                'title': 'Digital Art and NFTs',
                'category': 'art',
                'keywords': ['digital art', 'NFT', 'blockchain', 'cryptocurrency', 'creativity'],
                'popularity': 0.6,
                'educational_value': 0.7,
                'difficulty_level': 'advanced',
                'discussion_points': [
                    'What are NFTs and how do they work?',
                    'Impact on traditional art markets',
                    'Future of digital ownership'
                ]
            },
            {
                'title': 'Plant-Based Diet',
                'category': 'food',
                'keywords': ['vegan', 'vegetarian', 'plant-based', 'nutrition', 'health'],
                'popularity': 0.65,
                'educational_value': 0.8,
                'difficulty_level': 'beginner',
                'discussion_points': [
                    'Health benefits of plant-based eating',
                    'Environmental impact of food choices',
                    'Easy plant-based meal ideas'
                ]
            },
            {
                'title': 'Gaming and Esports',
                'category': 'entertainment',
                'keywords': ['gaming', 'esports', 'video games', 'competition', 'streaming'],
                'popularity': 0.8,
                'educational_value': 0.6,
                'difficulty_level': 'beginner',
                'discussion_points': [
                    'Growth of competitive gaming',
                    'Gaming as a career option',
                    'Social aspects of online gaming'
                ]
            }
        ]
        
        self.trending_topics = current_trends
        self.last_update = datetime.now()
        
        # 관련성 캐시 초기화
        self.topic_relevance_cache = {}
        
        logger.info(f"Updated {len(current_trends)} trending topics")
    
    def should_update_trends(self) -> bool:
        """트렌드 업데이트 필요 여부 확인"""
        if self.last_update is None:
            return True
        
        return datetime.now() - self.last_update > self.update_interval
    
    def extract_keywords(self, text: str) -> List[str]:
        """텍스트에서 키워드 추출"""
        # 간단한 키워드 추출 (실제로는 NLP 라이브러리 사용)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'can', 'may', 'might', 'must', 'shall', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        
        words = text.lower().split()
        keywords = [word.strip('.,!?;:"()[]') for word in words 
                    if len(word) > 3 and word.lower() not in stop_words]
        
        return keywords[:10]  # 상위 10개 키워드
    
    async def find_relevant_trends(
        self,
        input_keywords: List[str],
        user_interests: List[Dict],
        recent_topics: List[Dict]
    ) -> List[Dict]:
        """관련 트렌드 찾기"""
        
        scored_trends = []
        
        for trend in self.trending_topics:
            relevance_score = await self.calculate_relevance_score(
                trend,
                input_keywords,
                user_interests,
                recent_topics
            )
            
            if relevance_score > 0.3:  # 임계값 이상인 트렌드만 포함
                trend_copy = trend.copy()
                trend_copy['relevance_score'] = relevance_score
                scored_trends.append(trend_copy)
        
        # 관련성 점수로 정렬
        scored_trends.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return scored_trends[:5]  # 상위 5개 트렌드
    
    async def calculate_relevance_score(
        self,
        trend: Dict,
        input_keywords: List[str],
        user_interests: List[Dict],
        recent_topics: List[Dict]
    ) -> float:
        """트렌드 관련성 점수 계산"""
        
        cache_key = f"{trend['title']}_{hash(tuple(input_keywords))}"
        if cache_key in self.topic_relevance_cache:
            return self.topic_relevance_cache[cache_key]
        
        score = 0.0
        
        # 1. 키워드 매칭 (40%)
        trend_keywords = trend['keywords']
        keyword_matches = 0
        
        for keyword in input_keywords:
            for trend_keyword in trend_keywords:
                if keyword.lower() in trend_keyword.lower() or trend_keyword.lower() in keyword.lower():
                    keyword_matches += 1
                    break
        
        keyword_score = min(1.0, keyword_matches / max(len(input_keywords), 1))
        score += keyword_score * 0.4
        
        # 2. 사용자 관심사 매칭 (30%)
        interest_score = 0.0
        if user_interests:
            for interest in user_interests:
                interest_area = interest.get('area', '')
                if interest_area == trend['category']:
                    interest_score = interest.get('interest_level', 0.5)
                    break
                elif any(keyword in trend['keywords'] for keyword in [interest_area]):
                    interest_score = interest.get('interest_level', 0.5) * 0.7
        
        score += interest_score * 0.3
        
        # 3. 최근 주제와의 연관성 (20%)
        recent_topic_score = 0.0
        if recent_topics:
            for recent_topic in recent_topics[:3]:  # 최근 3개 주제만 고려
                topic_name = recent_topic.get('topic', '')
                if topic_name == trend['category']:
                    recent_topic_score = 0.8
                    break
                elif any(keyword in trend['keywords'] for keyword in [topic_name]):
                    recent_topic_score = 0.5
        
        score += recent_topic_score * 0.2
        
        # 4. 트렌드 인기도 (10%)
        popularity_score = trend.get('popularity', 0.5)
        score += popularity_score * 0.1
        
        # 캐시에 저장
        self.topic_relevance_cache[cache_key] = score
        
        return min(1.0, score)
    
    def get_trending_keywords(self) -> List[str]:
        """현재 트렌딩 키워드 목록"""
        all_keywords = []
        
        for trend in self.trending_topics:
            all_keywords.extend(trend['keywords'])
        
        # 중복 제거 및 인기도순 정렬 (간단한 방식)
        unique_keywords = list(set(all_keywords))
        
        return unique_keywords[:20]  # 상위 20개 키워드
    
    def generate_trend_based_starters(self, relevant_trends: List[Dict]) -> List[str]:
        """트렌드 기반 대화 시작 문구 생성"""
        
        if not relevant_trends:
            return [
                "What's something interesting you've heard about recently?",
                "Are there any current topics you'd like to discuss?",
                "What trends have you noticed lately?"
            ]
        
        starters = []
        
        for trend in relevant_trends[:3]:  # 상위 3개 트렌드
            title = trend['title']
            discussion_points = trend.get('discussion_points', [])
            
            # 일반적인 시작 문구
            starters.append(f"Have you heard about {title}? What do you think about it?")
            
            # 토론 포인트 기반 문구
            if discussion_points:
                point = random.choice(discussion_points)
                starters.append(f"I'd like to hear your thoughts on {point.lower()}.")
        
        return starters[:5]
    
    def assess_educational_value(self, trends: List[Dict]) -> Dict:
        """트렌드의 교육적 가치 평가"""
        
        if not trends:
            return {
                'average_educational_value': 0.7,
                'vocabulary_potential': 'medium',
                'discussion_depth': 'medium',
                'cultural_relevance': 'medium'
            }
        
        avg_educational_value = sum(trend.get('educational_value', 0.7) for trend in trends) / len(trends)
        
        # 어휘 학습 잠재력 평가
        total_keywords = sum(len(trend['keywords']) for trend in trends)
        vocabulary_potential = 'high' if total_keywords > 15 else 'medium' if total_keywords > 8 else 'low'
        
        # 토론 깊이 평가
        total_discussion_points = sum(len(trend.get('discussion_points', [])) for trend in trends)
        discussion_depth = 'high' if total_discussion_points > 10 else 'medium' if total_discussion_points > 5 else 'low'
        
        # 문화적 관련성 평가
        categories = set(trend['category'] for trend in trends)
        cultural_relevance = 'high' if len(categories) > 2 else 'medium'
        
        return {
            'average_educational_value': round(avg_educational_value, 2),
            'vocabulary_potential': vocabulary_potential,
            'discussion_depth': discussion_depth,
            'cultural_relevance': cultural_relevance,
            'recommended_for_discussion': len(trends) > 0
        }
    
    async def get_trending_by_category(self, category: str) -> List[Dict]:
        """카테고리별 트렌딩 토픽 조회"""
        
        if self.should_update_trends():
            await self.update_trending_topics()
        
        category_trends = [
            trend for trend in self.trending_topics
            if trend['category'].lower() == category.lower()
        ]
        
        return sorted(category_trends, key=lambda x: x['popularity'], reverse=True)
    
    async def get_trending_by_difficulty(self, difficulty: str) -> List[Dict]:
        """난이도별 트렌딩 토픽 조회"""
        
        if self.should_update_trends():
            await self.update_trending_topics()
        
        difficulty_trends = [
            trend for trend in self.trending_topics
            if trend['difficulty_level'].lower() == difficulty.lower()
        ]
        
        return sorted(difficulty_trends, key=lambda x: x['educational_value'], reverse=True)
    
    def get_conversation_prompts(self, trend: Dict, user_level: str) -> List[str]:
        """트렌드 기반 대화 프롬프트 생성"""
        
        title = trend['title']
        discussion_points = trend.get('discussion_points', [])
        
        prompts = []
        
        if user_level == 'beginner':
            prompts = [
                f"Do you know anything about {title}?",
                f"What do you think about {title}?",
                f"Have you heard about {title} before?"
            ]
        elif user_level == 'intermediate':
            prompts = [
                f"What's your opinion on {title}?",
                f"How do you think {title} affects our daily lives?",
                f"What are the pros and cons of {title}?"
            ]
        else:  # advanced
            prompts = [
                f"What are the broader implications of {title}?",
                f"How might {title} evolve in the next decade?",
                f"What ethical considerations surround {title}?"
            ]
        
        # 토론 포인트 추가
        if discussion_points:
            selected_points = random.sample(discussion_points, min(2, len(discussion_points)))
            for point in selected_points:
                prompts.append(f"Let's discuss {point.lower()}.")
        
        return prompts[:5]
    
    async def analyze_trend_engagement(self, user_id: str, trend_interactions: List[Dict]) -> Dict:
        """사용자의 트렌드 참여도 분석"""
        
        if not trend_interactions:
            return {
                'engagement_level': 'low',
                'preferred_categories': [],
                'discussion_quality': 0.5,
                'learning_progress': 'stable'
            }
        
        # 카테고리별 참여도
        category_count = {}
        total_interactions = len(trend_interactions)
        
        for interaction in trend_interactions:
            category = interaction.get('category', 'general')
            category_count[category] = category_count.get(category, 0) + 1
        
        # 선호 카테고리 (상위 3개)
        preferred_categories = sorted(category_count.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # 참여 수준 평가
        if total_interactions >= 10:
            engagement_level = 'high'
        elif total_interactions >= 5:
            engagement_level = 'medium'
        else:
            engagement_level = 'low'
        
        # 토론 품질 평가
        quality_scores = [interaction.get('quality_score', 0.5) for interaction in trend_interactions]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        return {
            'engagement_level': engagement_level,
            'total_interactions': total_interactions,
            'preferred_categories': [cat for cat, count in preferred_categories],
            'discussion_quality': round(avg_quality, 2),
            'learning_progress': 'improving' if avg_quality > 0.7 else 'stable',
            'recommendations': self.generate_engagement_recommendations(
                engagement_level, preferred_categories, avg_quality
            )
        }
    
    def generate_engagement_recommendations(
        self,
        engagement_level: str,
        preferred_categories: List[tuple],
        avg_quality: float
    ) -> List[str]:
        """참여도 기반 추천 생성"""
        
        recommendations = []
        
        if engagement_level == 'low':
            recommendations.append("Try discussing more current topics to stay engaged")
            recommendations.append("Start with trending topics you're interested in")
        
        if avg_quality < 0.6:
            recommendations.append("Focus on expressing your opinions more clearly")
            recommendations.append("Ask follow-up questions to deepen discussions")
        
        if preferred_categories:
            top_category = preferred_categories[0][0]
            recommendations.append(f"Explore more {top_category}-related topics for deeper learning")
        
        if not recommendations:
            recommendations.append("Continue engaging with diverse trending topics")
            recommendations.append("Challenge yourself with more complex discussions")
        
        return recommendations[:3]