"""
실시간 대화형 음성 응답 시스템 (Interactive Voice Response)
"""

import asyncio
import random
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from core.voice_engine import VoiceEngine
from agents.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

class ConversationManager:
    """대화 관리자"""
    
    def __init__(self):
        self.conversation_state = {}
        self.topic_transitions = {}
        self.engagement_levels = {}
    
    async def analyze_user_input(self, user_input: str, context: Dict) -> Dict:
        """사용자 입력 분석"""
        
        analysis = {
            'length': len(user_input.split()),
            'question_count': user_input.count('?'),
            'excitement_level': self.detect_excitement(user_input),
            'topic_shift': await self.detect_topic_shift(user_input, context),
            'completion_type': self.detect_completion_type(user_input),
            'emotional_tone': self.detect_emotional_tone(user_input)
        }
        
        return analysis
    
    def detect_excitement(self, text: str) -> float:
        """흥분도 감지"""
        excitement_indicators = ['!', 'wow', 'amazing', 'incredible', 'awesome', 'fantastic']
        score = 0
        
        for indicator in excitement_indicators:
            if indicator in text.lower():
                score += 1
        
        return min(score / 3, 1.0)  # 0-1 범위로 정규화
    
    async def detect_topic_shift(self, user_input: str, context: Dict) -> bool:
        """주제 전환 감지"""
        shift_indicators = ['by the way', 'speaking of', 'that reminds me', 'oh, and', 'also']
        return any(indicator in user_input.lower() for indicator in shift_indicators)
    
    def detect_completion_type(self, text: str) -> str:
        """문장 완성도 타입"""
        if text.strip().endswith('...') or text.strip().endswith(','):
            return 'incomplete'
        elif text.strip().endswith(('?', '!')):
            return 'emphatic'
        else:
            return 'complete'
    
    def detect_emotional_tone(self, text: str) -> str:
        """감정 톤 감지"""
        positive_words = ['happy', 'good', 'great', 'awesome', 'love', 'like', 'enjoy']
        negative_words = ['sad', 'bad', 'terrible', 'hate', 'dislike', 'boring']
        confused_words = ['confused', 'don\'t understand', 'what', 'how', 'why']
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in confused_words):
            return 'confused'
        elif any(word in text_lower for word in positive_words):
            return 'positive'
        elif any(word in text_lower for word in negative_words):
            return 'negative'
        else:
            return 'neutral'

class VoiceGenerator:
    """음성 생성기"""
    
    def __init__(self, voice_engine: VoiceEngine):
        self.voice_engine = voice_engine
        self.filler_cache = {}
        
    async def generate_quick_response(
        self, 
        text: str, 
        voice_settings: Dict,
        priority: str = 'normal'
    ) -> bytes:
        """빠른 응답 생성"""
        
        # 짧은 응답은 캐시 확인
        cache_key = f"{text}_{voice_settings.get('emotion', 'neutral')}"
        
        if cache_key in self.filler_cache and len(text.split()) <= 3:
            return self.filler_cache[cache_key]
        
        # 음성 생성
        audio = await self.voice_engine.generate_response_audio(
            text=text,
            user_id=voice_settings.get('user_id'),
            emotion=voice_settings.get('emotion', 'neutral'),
            use_bone_conduction=voice_settings.get('use_bone_conduction', True),
            speed=voice_settings.get('speed', 1.1)  # 약간 빠르게
        )
        
        # 짧은 응답은 캐시에 저장
        if len(text.split()) <= 3:
            self.filler_cache[cache_key] = audio
        
        return audio
    
    async def generate_filler_audio(self, text: str, voice_settings: Dict) -> bytes:
        """필러 오디오 생성"""
        
        return await self.generate_quick_response(
            text=text,
            voice_settings={
                **voice_settings,
                'emotion': 'thoughtful',
                'speed': 0.9  # 조금 느리게
            }
        )
    
    def concatenate_audio(self, audio_segments: List[bytes]) -> bytes:
        """오디오 세그먼트 연결"""
        
        if not audio_segments:
            return b""
        
        if len(audio_segments) == 1:
            return audio_segments[0]
        
        # 간단한 연결 (실제로는 더 정교한 오디오 처리 필요)
        # 각 세그먼트 사이에 짧은 휴지 추가
        pause_duration = 0.3  # 300ms 휴지
        sample_rate = 22050
        pause_samples = int(pause_duration * sample_rate)
        pause_audio = np.zeros(pause_samples, dtype=np.float32).tobytes()
        
        result = audio_segments[0]
        for segment in audio_segments[1:]:
            result += pause_audio + segment
        
        return result

class ResponseCache:
    """응답 캐시"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[bytes]:
        """캐시에서 가져오기"""
        if key in self.cache:
            self.access_times[key] = datetime.now()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: bytes):
        """캐시에 저장"""
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = value
        self.access_times[key] = datetime.now()
    
    def _evict_oldest(self):
        """가장 오래된 항목 제거"""
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

class InteractiveVoiceResponse:
    """실시간 대화형 음성 응답"""
    
    def __init__(self, voice_engine: VoiceEngine, agent_orchestrator: AgentOrchestrator):
        self.conversation_manager = ConversationManager()
        self.voice_generator = VoiceGenerator(voice_engine)
        self.response_cache = ResponseCache()
        self.agent_orchestrator = agent_orchestrator
        self.voice_engine = voice_engine
        
        # 응답 타이밍 설정
        self.response_timings = {
            'immediate': 0.2,  # 200ms
            'thinking': 1.5,   # 1.5초
            'clarification': 0.8,  # 800ms
            'normal': 0.5      # 500ms
        }
        
    async def generate_conversational_response(
        self,
        user_input: str,
        context: Dict,
        voice_settings: Dict
    ) -> Dict:
        """대화형 응답 생성"""
        
        # 1. 사용자 입력 분석
        input_analysis = await self.conversation_manager.analyze_user_input(user_input, context)
        
        # 2. 응답 타입 결정
        response_type = await self.determine_response_type(user_input, input_analysis)
        
        # 3. 응답 생성
        start_time = datetime.now()
        
        if response_type == 'immediate':
            # 즉각적인 반응 (Uh-huh, I see, Really? 등)
            response_text = await self.generate_backchannel_response(input_analysis)
            audio = await self.voice_generator.generate_quick_response(
                response_text, 
                voice_settings,
                priority='high'
            )
            
        elif response_type == 'thinking':
            # 생각하는 중 표현
            filler_text = await self.generate_thinking_filler(input_analysis)
            
            # 필러와 메인 응답을 병렬로 생성
            filler_task = asyncio.create_task(
                self.voice_generator.generate_filler_audio(filler_text, voice_settings)
            )
            
            main_response_task = asyncio.create_task(
                self.generate_main_response(user_input, context, voice_settings)
            )
            
            filler_audio, main_response_data = await asyncio.gather(
                filler_task, 
                main_response_task
            )
            
            # 오디오 연결
            audio = self.voice_generator.concatenate_audio([
                filler_audio, 
                main_response_data['audio']
            ])
            
            response_text = f"{filler_text} {main_response_data['text']}"
            
        elif response_type == 'clarification':
            # 명확화 요청
            response_text = await self.generate_clarification_request(user_input, input_analysis)
            audio = await self.voice_generator.generate_quick_response(
                response_text,
                {**voice_settings, 'emotion': 'confused'},
                priority='high'
            )
            
        elif response_type == 'encouraging':
            # 격려 응답
            response_text = await self.generate_encouraging_response(input_analysis)
            audio = await self.voice_generator.generate_quick_response(
                response_text,
                {**voice_settings, 'emotion': 'encouraging'},
                priority='high'
            )
            
        else:  # 'normal'
            main_response_data = await self.generate_main_response(user_input, context, voice_settings)
            response_text = main_response_data['text']
            audio = main_response_data['audio']
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'text': response_text,
            'audio': audio,
            'response_type': response_type,
            'timing': await self.calculate_response_timing(response_type),
            'generation_time': generation_time,
            'input_analysis': input_analysis,
            'conversation_flow': await self.assess_conversation_flow(input_analysis)
        }
    
    async def determine_response_type(self, user_input: str, analysis: Dict) -> str:
        """응답 타입 결정"""
        
        # 매우 짧은 입력 → immediate
        if analysis['length'] <= 2:
            return 'immediate'
        
        # 감정적인 톤이 confused → clarification
        if analysis['emotional_tone'] == 'confused':
            return 'clarification'
        
        # 감정적인 톤이 negative → encouraging
        if analysis['emotional_tone'] == 'negative':
            return 'encouraging'
        
        # 매우 긴 입력이나 복잡한 질문 → thinking
        if analysis['length'] > 20 or analysis['question_count'] > 2:
            return 'thinking'
        
        # 흥분도가 높음 → immediate
        if analysis['excitement_level'] > 0.7:
            return 'immediate'
        
        # 일반적인 경우
        return 'normal'
    
    async def generate_backchannel_response(self, analysis: Dict) -> str:
        """즉각적인 반응 생성"""
        
        excitement = analysis['excitement_level']
        tone = analysis['emotional_tone']
        
        if excitement > 0.7:
            responses = [
                "Oh wow!",
                "That's amazing!",
                "Really?!",
                "No way!",
                "Incredible!",
                "Awesome!"
            ]
        elif tone == 'positive':
            responses = [
                "That's great!",
                "Wonderful!",
                "I'm glad to hear that!",
                "Nice!",
                "Excellent!"
            ]
        elif tone == 'negative':
            responses = [
                "I understand.",
                "I see how that could be difficult.",
                "That sounds challenging.",
                "Hmm, I get it."
            ]
        else:
            responses = [
                "I see.",
                "Uh-huh.",
                "Right.",
                "Got it!",
                "Okay.",
                "Interesting.",
                "Tell me more.",
                "Go on..."
            ]
        
        return random.choice(responses)
    
    async def generate_thinking_filler(self, analysis: Dict) -> str:
        """생각하는 중 필러 생성"""
        
        complexity_based_fillers = [
            "Hmm, let me think about that...",
            "That's a really good question...",
            "Well, let's see...",
            "Oh, that's interesting...",
            "You know what?",
            "Actually...",
            "Let me consider that...",
            "That's a thoughtful point..."
        ]
        
        return random.choice(complexity_based_fillers)
    
    async def generate_clarification_request(self, user_input: str, analysis: Dict) -> str:
        """명확화 요청 생성"""
        
        clarification_requests = [
            "I'm not sure I understand. Could you explain that a bit more?",
            "Can you help me understand what you mean?",
            "I want to make sure I get this right. Can you clarify?",
            "Could you tell me more about that?",
            "I'm following you, but could you elaborate?",
            "What do you mean by that exactly?",
            "Can you give me an example?"
        ]
        
        return random.choice(clarification_requests)
    
    async def generate_encouraging_response(self, analysis: Dict) -> str:
        """격려 응답 생성"""
        
        encouraging_responses = [
            "Don't worry, that's completely normal!",
            "Hey, everyone struggles with that sometimes.",
            "That's okay! Let's work through it together.",
            "No problem at all! Let's try a different approach.",
            "It's perfectly fine to feel that way.",
            "Don't be too hard on yourself!",
            "That's actually a common challenge.",
            "Let's take it step by step."
        ]
        
        return random.choice(encouraging_responses)
    
    async def generate_main_response(
        self, 
        user_input: str, 
        context: Dict, 
        voice_settings: Dict
    ) -> Dict:
        """메인 응답 생성"""
        
        # 에이전트 오케스트레이터를 통해 응답 생성
        agent_response = await self.agent_orchestrator.process_conversation(
            user_id=voice_settings.get('user_id'),
            user_input=user_input,
            session_id=context.get('session_id', 'default')
        )
        
        # 음성 생성
        audio = await self.voice_engine.generate_response_audio(
            text=agent_response['text'],
            user_id=voice_settings.get('user_id'),
            emotion=agent_response.get('emotion', 'neutral'),
            use_bone_conduction=voice_settings.get('use_bone_conduction', True)
        )
        
        return {
            'text': agent_response['text'],
            'audio': audio,
            'agent_data': agent_response
        }
    
    async def calculate_response_timing(self, response_type: str) -> Dict:
        """응답 타이밍 계산"""
        
        base_delay = self.response_timings.get(response_type, 0.5)
        
        # 자연스러운 변동 추가 (±20%)
        variation = random.uniform(-0.2, 0.2)
        actual_delay = max(0.1, base_delay * (1 + variation))
        
        return {
            'type': response_type,
            'base_delay': base_delay,
            'actual_delay': actual_delay,
            'should_interrupt': response_type in ['immediate', 'clarification']
        }
    
    async def assess_conversation_flow(self, analysis: Dict) -> Dict:
        """대화 흐름 평가"""
        
        flow_quality = 'good'
        
        # 흐름 품질 평가
        if analysis['emotional_tone'] == 'confused':
            flow_quality = 'needs_clarification'
        elif analysis['completion_type'] == 'incomplete':
            flow_quality = 'interrupted'
        elif analysis['excitement_level'] > 0.8:
            flow_quality = 'highly_engaged'
        elif analysis['length'] < 3:
            flow_quality = 'minimal_engagement'
        
        return {
            'quality': flow_quality,
            'engagement_level': analysis['excitement_level'],
            'coherence_score': await self.calculate_coherence_score(analysis),
            'suggested_followup': await self.suggest_followup_strategy(analysis)
        }
    
    async def calculate_coherence_score(self, analysis: Dict) -> float:
        """응집성 점수 계산"""
        
        score = 0.5  # 기본 점수
        
        # 길이 기반 점수
        if 5 <= analysis['length'] <= 25:
            score += 0.2
        elif analysis['length'] < 3:
            score -= 0.2
        
        # 감정 톤 기반
        if analysis['emotional_tone'] != 'confused':
            score += 0.2
        else:
            score -= 0.3
        
        # 완성도 기반
        if analysis['completion_type'] == 'complete':
            score += 0.1
        elif analysis['completion_type'] == 'incomplete':
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    async def suggest_followup_strategy(self, analysis: Dict) -> str:
        """후속 전략 제안"""
        
        if analysis['emotional_tone'] == 'confused':
            return 'provide_examples'
        elif analysis['excitement_level'] > 0.7:
            return 'maintain_energy'
        elif analysis['length'] < 3:
            return 'encourage_elaboration'
        elif analysis['completion_type'] == 'incomplete':
            return 'wait_for_completion'
        else:
            return 'continue_naturally'
    
    async def handle_interruption(self, current_response: Dict, interruption: str) -> Dict:
        """사용자 중단 처리"""
        
        # 현재 응답 중단하고 새로운 응답 생성
        logger.info(f"Handling interruption: {interruption}")
        
        # 중단에 대한 즉각적인 반응
        acknowledgment = await self.generate_interruption_acknowledgment(interruption)
        
        return {
            'acknowledgment': acknowledgment,
            'should_restart': True,
            'interrupt_type': 'user_initiated'
        }
    
    async def generate_interruption_acknowledgment(self, interruption: str) -> str:
        """중단 인정 응답"""
        
        acknowledgments = [
            "Oh, sorry about that!",
            "My apologies, go ahead.",
            "Of course, what were you saying?",
            "Sure, I'm listening.",
            "Yes, please continue.",
            "Sorry, you were saying?"
        ]
        
        return random.choice(acknowledgments)