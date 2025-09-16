"""
완전한 챗봇 음성 시스템
실시간 스트리밍, 적응형 제어, 액센트 다양성, 최적화 시스템 통합
"""

import asyncio
import numpy as np
import torch
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import time
import threading
from collections import deque
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor

# Import will be done dynamically to avoid circular imports

logger = logging.getLogger(__name__)

class StreamingStatus(Enum):
    """스트리밍 상태"""
    IDLE = "idle"
    BUFFERING = "buffering"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

class VoiceQuality(Enum):
    """음성 품질 레벨"""
    LOW = "low"        # 8kHz, 빠른 생성
    MEDIUM = "medium"  # 16kHz, 균형
    HIGH = "high"      # 22kHz, 고품질
    ULTRA = "ultra"    # 44kHz, 최고품질

class EmotionState(Enum):
    """감정 상태 (로컬 정의)"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    ENCOURAGING = "encouraging"
    PATIENT = "patient"
    SURPRISED = "surprised"
    CONCERNED = "concerned"

@dataclass
class LevelConfig:
    """레벨별 음성 설정"""
    name: str
    speed: float
    clarity: float
    pause_duration: float
    repetition_chance: float
    vocabulary_complexity: str
    sentence_length: str

@dataclass 
class StreamingChunk:
    """스트리밍 오디오 청크"""
    chunk_id: str
    audio_data: bytes
    sample_rate: int
    duration: float
    text_segment: str
    metadata: Dict[str, Any]

class RealTimeVoiceStreaming:
    """실시간 음성 스트리밍 시스템"""
    
    def __init__(self, buffer_size: int = 4096, quality: VoiceQuality = VoiceQuality.MEDIUM):
        self.buffer_size = buffer_size
        self.quality = quality
        self.active_streams = {}
        self.stream_buffers = {}
        self.quality_configs = {
            VoiceQuality.LOW: {'sample_rate': 8000, 'chunk_duration': 0.5},
            VoiceQuality.MEDIUM: {'sample_rate': 16000, 'chunk_duration': 0.3},
            VoiceQuality.HIGH: {'sample_rate': 22050, 'chunk_duration': 0.2},
            VoiceQuality.ULTRA: {'sample_rate': 44100, 'chunk_duration': 0.1}
        }
        
    async def create_stream(self, user_id: str, session_id: str) -> str:
        """새로운 스트림 생성"""
        stream_id = f"{user_id}_{session_id}_{uuid.uuid4().hex[:8]}"
        
        self.active_streams[stream_id] = {
            'user_id': user_id,
            'session_id': session_id,
            'status': StreamingStatus.IDLE,
            'created_at': time.time(),
            'total_chunks': 0,
            'bytes_streamed': 0
        }
        
        self.stream_buffers[stream_id] = deque(maxlen=50)  # 최대 50개 청크 버퍼
        
        logger.info(f"Created voice stream: {stream_id}")
        return stream_id
    
    async def stream_chatbot_voice(
        self,
        stream_id: str,
        text: str,
        persona: str,
        emotion: str = "neutral",
        speed: float = 1.0
    ) -> asyncio.Queue:
        """챗봇 음성 스트리밍"""
        
        if stream_id not in self.active_streams:
            raise ValueError(f"Stream {stream_id} not found")
        
        # 스트림 상태 업데이트
        self.active_streams[stream_id]['status'] = StreamingStatus.BUFFERING
        
        # 텍스트를 청크로 분할
        text_chunks = self._split_text_for_streaming(text)
        
        # 출력 큐 생성
        output_queue = asyncio.Queue()
        
        # 백그라운드에서 오디오 생성 및 스트리밍
        asyncio.create_task(self._generate_and_stream(
            stream_id, text_chunks, persona, emotion, speed, output_queue
        ))
        
        return output_queue
    
    async def _generate_and_stream(
        self,
        stream_id: str,
        text_chunks: List[str],
        persona: str,
        emotion: str,
        speed: float,
        output_queue: asyncio.Queue
    ):
        """오디오 생성 및 스트리밍 (백그라운드)"""
        
        try:
            self.active_streams[stream_id]['status'] = StreamingStatus.STREAMING
            
            config = self.quality_configs[self.quality]
            chunk_duration = config['chunk_duration'] / speed
            
            for i, text_chunk in enumerate(text_chunks):
                # 텍스트 청크를 오디오로 변환 (시뮬레이션)
                audio_data = await self._generate_chunk_audio(
                    text_chunk, persona, emotion, config['sample_rate']
                )
                
                # 청크 생성
                chunk = StreamingChunk(
                    chunk_id=f"{stream_id}_{i:04d}",
                    audio_data=audio_data,
                    sample_rate=config['sample_rate'],
                    duration=chunk_duration,
                    text_segment=text_chunk,
                    metadata={
                        'persona': persona,
                        'emotion': emotion,
                        'chunk_index': i,
                        'total_chunks': len(text_chunks)
                    }
                )
                
                # 버퍼에 추가
                self.stream_buffers[stream_id].append(chunk)
                
                # 큐에 전송
                await output_queue.put(chunk)
                
                # 스트리밍 통계 업데이트
                self.active_streams[stream_id]['total_chunks'] += 1
                self.active_streams[stream_id]['bytes_streamed'] += len(audio_data)
                
                # 지연 시간 조절
                await asyncio.sleep(chunk_duration * 0.8)  # 80% 중첩
            
            # 완료 표시
            self.active_streams[stream_id]['status'] = StreamingStatus.COMPLETED
            await output_queue.put(None)  # 종료 신호
            
        except Exception as e:
            logger.error(f"Streaming error for {stream_id}: {e}")
            self.active_streams[stream_id]['status'] = StreamingStatus.ERROR
            await output_queue.put(None)
    
    def _split_text_for_streaming(self, text: str, max_chunk_length: int = 100) -> List[str]:
        """텍스트를 스트리밍용 청크로 분할"""
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    async def _generate_chunk_audio(
        self,
        text: str,
        persona: str,
        emotion: str,
        sample_rate: int
    ) -> bytes:
        """청크 오디오 생성 (시뮬레이션)"""
        
        # 시뮬레이션: 실제로는 OpenVoice 사용
        duration = len(text) * 0.08  # 대략적인 읽기 시간
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # 기본 사인파 생성
        frequency = 200 + hash(persona) % 100  # 페르소나별 다른 주파수
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # 감정별 변조
        if emotion == "happy":
            audio *= 1.2  # 더 밝게
            frequency *= 1.1
        elif emotion == "calm":
            audio *= 0.8  # 더 조용하게
        
        # 노이즈 추가 (자연스러움)
        noise = 0.05 * np.random.random(len(t)) - 0.025
        audio += noise
        
        # 정규화
        audio = np.clip(audio, -0.95, 0.95)
        
        return audio.astype(np.float32).tobytes()
    
    async def pause_stream(self, stream_id: str):
        """스트림 일시정지"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['status'] = StreamingStatus.PAUSED
            logger.info(f"Paused stream: {stream_id}")
    
    async def resume_stream(self, stream_id: str):
        """스트림 재개"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['status'] = StreamingStatus.STREAMING
            logger.info(f"Resumed stream: {stream_id}")
    
    async def stop_stream(self, stream_id: str):
        """스트림 중지"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            if stream_id in self.stream_buffers:
                del self.stream_buffers[stream_id]
            logger.info(f"Stopped stream: {stream_id}")
    
    def get_stream_status(self, stream_id: str) -> Optional[Dict]:
        """스트림 상태 조회"""
        return self.active_streams.get(stream_id)

class AdaptiveVoiceController:
    """적응형 음성 제어기"""
    
    def __init__(self):
        self.level_presets = {
            'beginner': LevelConfig(
                name='beginner',
                speed=0.8,
                clarity=1.0,
                pause_duration=1.5,
                repetition_chance=0.3,
                vocabulary_complexity='simple',
                sentence_length='short'
            ),
            'elementary': LevelConfig(
                name='elementary', 
                speed=0.85,
                clarity=0.95,
                pause_duration=1.3,
                repetition_chance=0.2,
                vocabulary_complexity='simple',
                sentence_length='short'
            ),
            'intermediate': LevelConfig(
                name='intermediate',
                speed=1.0,
                clarity=0.9,
                pause_duration=1.0,
                repetition_chance=0.15,
                vocabulary_complexity='medium',
                sentence_length='medium'
            ),
            'upper_intermediate': LevelConfig(
                name='upper_intermediate',
                speed=1.05,
                clarity=0.85,
                pause_duration=0.8,
                repetition_chance=0.1,
                vocabulary_complexity='medium',
                sentence_length='medium'
            ),
            'advanced': LevelConfig(
                name='advanced',
                speed=1.1,
                clarity=0.8,
                pause_duration=0.6,
                repetition_chance=0.05,
                vocabulary_complexity='complex',
                sentence_length='long'
            )
        }
        self.user_adaptations = {}
        
    async def adjust_voice_for_level(
        self,
        text: str,
        level: str,
        persona: str,
        user_id: str = None
    ) -> Dict[str, Any]:
        """레벨에 맞춘 음성 조정"""
        
        config = self.level_presets.get(level, self.level_presets['intermediate'])
        
        # 사용자별 적응 적용
        if user_id and user_id in self.user_adaptations:
            adaptation = self.user_adaptations[user_id]
            config = self._apply_user_adaptation(config, adaptation)
        
        # 텍스트 수정 (레벨에 따라)
        adjusted_text = await self._adjust_text_for_level(text, config)
        
        return {
            'adjusted_text': adjusted_text,
            'voice_config': {
                'speed': config.speed,
                'clarity': config.clarity,
                'pause_duration': config.pause_duration,
                'persona': persona
            },
            'repetition_needed': np.random.random() < config.repetition_chance,
            'level_info': {
                'level': config.name,
                'vocabulary_complexity': config.vocabulary_complexity,
                'sentence_length': config.sentence_length
            }
        }
    
    async def _adjust_text_for_level(self, text: str, config: LevelConfig) -> str:
        """레벨에 따른 텍스트 조정"""
        
        if config.vocabulary_complexity == 'simple':
            # 간단한 단어로 치환
            replacements = {
                'utilize': 'use',
                'demonstrate': 'show',
                'comprehend': 'understand',
                'approximately': 'about',
                'extremely': 'very'
            }
            for complex_word, simple_word in replacements.items():
                text = text.replace(complex_word, simple_word)
        
        if config.sentence_length == 'short':
            # 긴 문장을 짧게 분할
            sentences = text.split('.')
            short_sentences = []
            for sentence in sentences:
                if len(sentence.split()) > 10:
                    # 긴 문장을 두 개로 분할 (간단한 방법)
                    words = sentence.split()
                    mid = len(words) // 2
                    short_sentences.append(' '.join(words[:mid]) + '.')
                    short_sentences.append(' '.join(words[mid:]) + '.')
                else:
                    short_sentences.append(sentence + '.')
            text = ' '.join(short_sentences).replace('..', '.')
        
        return text
    
    def _apply_user_adaptation(self, config: LevelConfig, adaptation: Dict) -> LevelConfig:
        """사용자 적응 적용"""
        adapted_config = LevelConfig(
            name=config.name,
            speed=config.speed * adaptation.get('speed_multiplier', 1.0),
            clarity=min(1.0, config.clarity * adaptation.get('clarity_multiplier', 1.0)),
            pause_duration=config.pause_duration * adaptation.get('pause_multiplier', 1.0),
            repetition_chance=config.repetition_chance * adaptation.get('repetition_multiplier', 1.0),
            vocabulary_complexity=config.vocabulary_complexity,
            sentence_length=config.sentence_length
        )
        return adapted_config
    
    async def learn_from_feedback(self, user_id: str, feedback: Dict[str, Any]):
        """피드백을 통한 학습"""
        
        if user_id not in self.user_adaptations:
            self.user_adaptations[user_id] = {
                'speed_multiplier': 1.0,
                'clarity_multiplier': 1.0,
                'pause_multiplier': 1.0,
                'repetition_multiplier': 1.0,
                'feedback_count': 0
            }
        
        adaptation = self.user_adaptations[user_id]
        adaptation['feedback_count'] += 1
        
        # 피드백에 따른 조정
        if feedback.get('too_fast'):
            adaptation['speed_multiplier'] = max(0.7, adaptation['speed_multiplier'] - 0.05)
        elif feedback.get('too_slow'):
            adaptation['speed_multiplier'] = min(1.3, adaptation['speed_multiplier'] + 0.05)
        
        if feedback.get('unclear'):
            adaptation['clarity_multiplier'] = min(1.2, adaptation['clarity_multiplier'] + 0.02)
        
        if feedback.get('need_more_pauses'):
            adaptation['pause_multiplier'] = min(2.0, adaptation['pause_multiplier'] + 0.1)
        
        if feedback.get('need_repetition'):
            adaptation['repetition_multiplier'] = min(2.0, adaptation['repetition_multiplier'] + 0.1)
        
        logger.info(f"Updated adaptation for user {user_id}: {adaptation}")

class AccentVarietySystem:
    """액센트 다양성 시스템"""
    
    def __init__(self):
        self.accent_rotation = {
            'daily': ['american', 'british'],
            'weekly': ['american', 'british', 'australian'],  
            'monthly': ['american', 'british', 'australian', 'canadian']
        }
        self.user_accent_exposure = {}
        
    async def get_today_accent(self, user_id: str, user_level: str) -> str:
        """오늘의 액센트 결정"""
        
        import datetime
        today = datetime.date.today()
        day_of_year = today.timetuple().tm_yday
        
        # 레벨별 액센트 노출 전략
        if user_level in ['beginner', 'elementary']:
            # 초급자는 American 위주, 가끔 British
            accents = ['american'] * 4 + ['british'] * 1
            selected = accents[day_of_year % len(accents)]
        elif user_level in ['intermediate', 'upper_intermediate']:
            # 중급자는 American, British 균등, 가끔 다른 액센트
            accents = ['american'] * 2 + ['british'] * 2 + ['australian'] * 1
            selected = accents[day_of_year % len(accents)]
        else:  # advanced
            # 고급자는 모든 액센트 균등 노출
            accents = ['american', 'british', 'australian', 'canadian']
            selected = accents[day_of_year % len(accents)]
        
        # 사용자 노출 기록 업데이트
        if user_id not in self.user_accent_exposure:
            self.user_accent_exposure[user_id] = {}
        
        exposure = self.user_accent_exposure[user_id]
        exposure[selected] = exposure.get(selected, 0) + 1
        
        logger.info(f"Selected accent for {user_id} ({user_level}): {selected}")
        return selected
    
    async def get_accent_diversity_report(self, user_id: str) -> Dict[str, Any]:
        """액센트 다양성 리포트"""
        
        if user_id not in self.user_accent_exposure:
            return {'message': 'No accent exposure data found'}
        
        exposure = self.user_accent_exposure[user_id]
        total_exposure = sum(exposure.values())
        
        return {
            'total_sessions': total_exposure,
            'accent_breakdown': {
                accent: {
                    'count': count,
                    'percentage': round(count / total_exposure * 100, 1)
                }
                for accent, count in exposure.items()
            },
            'most_exposed': max(exposure, key=exposure.get),
            'least_exposed': min(exposure, key=exposure.get) if len(exposure) > 1 else None,
            'diversity_score': len(exposure) / 4.0  # 4가지 액센트 대비
        }
    
    async def suggest_accent_for_context(self, context: str, user_level: str) -> str:
        """컨텍스트별 액센트 추천"""
        
        context_accent_map = {
            'business': 'british',  # 비즈니스는 영국식
            'casual': 'american',   # 캐주얼은 미국식
            'travel': 'australian', # 여행은 호주식
            'academic': 'british',  # 학술적은 영국식
            'technology': 'american', # 기술은 미국식
            'sports': 'australian'  # 스포츠는 호주식
        }
        
        suggested = context_accent_map.get(context, 'american')
        
        # 레벨이 낮으면 American으로 고정
        if user_level in ['beginner', 'elementary'] and suggested != 'american':
            suggested = 'american'
        
        return suggested

class VoiceOptimizationSystem:
    """음성 최적화 시스템"""
    
    def __init__(self, cache_size_mb: int = 100):
        self.cache_size_bytes = cache_size_mb * 1024 * 1024
        self.audio_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.common_phrases = {}
        self.pregenerated_audio = {}
        
    async def initialize_common_phrases(self, personas: List[str]):
        """공통 구문 사전 생성 시스템 초기화"""
        
        common_phrases_list = [
            # 인사
            "Hello! How are you today?",
            "Good morning! Ready to learn?",
            "Hi there! Let's get started.",
            
            # 격려
            "Great job!",
            "You're doing well!",
            "Excellent progress!",
            "Keep it up!",
            
            # 질문
            "Could you repeat that?",
            "What do you think?",
            "How was that?",
            "Any questions?",
            
            # 설명
            "Let me explain.",
            "Here's how it works.",
            "The correct way is...",
            "Try this instead.",
            
            # 마무리
            "Great session today!",
            "See you next time!",
            "Well done!",
            "That's all for now."
        ]
        
        # 각 페르소나별 공통 구문 음성 사전 생성
        for persona in personas:
            self.common_phrases[persona] = {}
            for phrase in common_phrases_list:
                # 실제로는 여기서 음성 생성
                cache_key = self._generate_cache_key(phrase, persona, 'neutral')
                audio_data = await self._generate_phrase_audio(phrase, persona)
                self.pregenerated_audio[cache_key] = {
                    'audio': audio_data,
                    'generated_at': time.time(),
                    'usage_count': 0
                }
        
        logger.info(f"Initialized {len(common_phrases_list)} common phrases for {len(personas)} personas")
    
    def _generate_cache_key(self, text: str, persona: str, emotion: str = 'neutral') -> str:
        """캐시 키 생성"""
        content = f"{text}_{persona}_{emotion}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def get_cached_or_generate(
        self,
        text: str,
        persona: str,
        emotion: str = 'neutral'
    ) -> bytes:
        """캐시된 오디오 반환 또는 생성"""
        
        cache_key = self._generate_cache_key(text, persona, emotion)
        
        # 캐시 확인
        if cache_key in self.audio_cache:
            self.cache_stats['hits'] += 1
            self.audio_cache[cache_key]['usage_count'] += 1
            self.audio_cache[cache_key]['last_used'] = time.time()
            return self.audio_cache[cache_key]['audio']
        
        # 사전 생성된 오디오 확인
        if cache_key in self.pregenerated_audio:
            audio_data = self.pregenerated_audio[cache_key]['audio']
            self.pregenerated_audio[cache_key]['usage_count'] += 1
            
            # 캐시에도 추가
            self._add_to_cache(cache_key, audio_data)
            self.cache_stats['hits'] += 1
            return audio_data
        
        # 새로 생성
        self.cache_stats['misses'] += 1
        audio_data = await self._generate_phrase_audio(text, persona, emotion)
        self._add_to_cache(cache_key, audio_data)
        
        return audio_data
    
    async def _generate_phrase_audio(self, text: str, persona: str, emotion: str = 'neutral') -> bytes:
        """구문 오디오 생성 (시뮬레이션)"""
        
        # 시뮬레이션 오디오 생성
        duration = len(text) * 0.08
        sample_rate = 22050
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # 페르소나별 기본 주파수
        base_freq = 200 + hash(persona) % 100
        
        # 감정별 조정
        if emotion == 'happy':
            base_freq *= 1.1
        elif emotion == 'calm':
            base_freq *= 0.95
        
        audio = 0.3 * np.sin(2 * np.pi * base_freq * t)
        audio += 0.05 * np.random.random(len(t)) - 0.025
        audio = np.clip(audio, -0.95, 0.95)
        
        return audio.astype(np.float32).tobytes()
    
    def _add_to_cache(self, cache_key: str, audio_data: bytes):
        """캐시에 오디오 추가"""
        
        # 캐시 크기 확인
        current_size = sum(len(item['audio']) for item in self.audio_cache.values())
        
        if current_size + len(audio_data) > self.cache_size_bytes:
            self._evict_cache_items()
        
        self.audio_cache[cache_key] = {
            'audio': audio_data,
            'created_at': time.time(),
            'last_used': time.time(),
            'usage_count': 1,
            'size': len(audio_data)
        }
    
    def _evict_cache_items(self):
        """캐시 아이템 제거 (LRU)"""
        
        # 사용량과 최근 사용 시간 기반으로 점수 계산
        scored_items = []
        current_time = time.time()
        
        for key, item in self.audio_cache.items():
            age = current_time - item['last_used']
            score = item['usage_count'] / (age + 1)  # 높을수록 유지
            scored_items.append((score, key))
        
        # 점수가 낮은 순으로 정렬하여 하위 25% 제거
        scored_items.sort()
        items_to_remove = len(scored_items) // 4
        
        for _, key in scored_items[:items_to_remove]:
            del self.audio_cache[key]
        
        logger.info(f"Evicted {items_to_remove} cache items")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0
        
        cache_size = sum(item['size'] for item in self.audio_cache.values())
        
        return {
            'hit_rate': round(hit_rate * 100, 2),
            'total_requests': total_requests,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'cache_items': len(self.audio_cache),
            'cache_size_mb': round(cache_size / (1024 * 1024), 2),
            'pregenerated_phrases': len(self.pregenerated_audio)
        }

class CompleteChatbotVoiceSystem:
    """완전한 챗봇 음성 시스템"""
    
    def __init__(self):
        # Initialize persona system dynamically to avoid circular imports
        self.persona_system = None
        self.emotion_engine = None
        self._initialize_persona_system()
        
        self.streaming = RealTimeVoiceStreaming()
        self.adaptive_controller = AdaptiveVoiceController()
        self.accent_system = AccentVarietySystem()
        self.optimization = VoiceOptimizationSystem()
    
    def _initialize_persona_system(self):
        """페르소나 시스템 동적 초기화"""
        try:
            from .chatbot_voice_persona import ChatbotVoicePersonaSystem, EmotionExpressionEngine
            self.persona_system = ChatbotVoicePersonaSystem()
            self.emotion_engine = EmotionExpressionEngine()
        except ImportError as e:
            logger.warning(f"Could not import persona system: {e}")
            self.persona_system = None
            self.emotion_engine = None
        
        # 시스템 상태 초기화
        self.active_sessions = {}
        self.system_stats = {
            'total_sessions': 0,
            'total_audio_generated': 0,
            'total_streaming_time': 0.0
        }
        
    async def initialize_for_user(self, user_id: str) -> Dict[str, Any]:
        """사용자별 초기화"""
        
        user_profile = await self.load_user_profile(user_id)
        
        # 1. 선호 음성 페르소나 설정
        preferred_persona = user_profile.get('preferred_voice', 'emily_us')
        
        # 2. 학습 레벨 확인
        user_level = user_profile.get('level', 'intermediate')
        
        # 3. 오늘의 액센트 결정
        today_accent = await self.accent_system.get_today_accent(user_id, user_level)
        
        # 4. 음성 설정 구성
        voice_settings = {
            'primary_persona': preferred_persona,
            'alternative_personas': self.select_alternative_personas(user_level),
            'level_config': self.adaptive_controller.level_presets[user_level],
            'accent_exposure': user_profile.get('accent_variety', True),
            'emotion_expression': user_profile.get('use_emotions', True),
            'today_accent': today_accent,
            'streaming_quality': VoiceQuality.MEDIUM
        }
        
        # 5. 공통 구문 사전 로드
        await self.optimization.initialize_common_phrases(
            [preferred_persona] + voice_settings['alternative_personas']
        )
        
        # 6. 스트림 생성
        session_id = f"session_{int(time.time())}"
        stream_id = await self.streaming.create_stream(user_id, session_id)
        voice_settings['stream_id'] = stream_id
        
        # 세션 기록
        self.active_sessions[user_id] = voice_settings
        self.system_stats['total_sessions'] += 1
        
        logger.info(f"Initialized voice system for user {user_id}: {preferred_persona}, {today_accent} accent")
        return voice_settings
    
    async def speak_response(
        self,
        text: str,
        user_id: str,
        emotion: str = 'neutral',
        immediate: bool = False,
        context: str = ''
    ) -> asyncio.Queue:
        """응답 음성 재생"""
        
        voice_settings = await self.initialize_for_user(user_id)
        
        # 1. 즉시 응답이 필요한 경우
        if immediate:
            # 캐시된 음성 사용
            audio = await self.optimization.get_cached_or_generate(
                text,
                voice_settings['primary_persona'],
                emotion
            )
            # 즉시 재생 큐 생성
            immediate_queue = asyncio.Queue()
            await immediate_queue.put({
                'type': 'immediate_audio',
                'audio': audio,
                'text': text
            })
            return immediate_queue
        
        # 2. 레벨에 맞춘 음성 조정
        adjusted_response = await self.adaptive_controller.adjust_voice_for_level(
            text,
            voice_settings['level_config'].name,
            voice_settings['primary_persona'],
            user_id
        )
        
        # 3. 컨텍스트별 액센트 결정
        if context:
            suggested_accent = await self.accent_system.suggest_accent_for_context(
                context, voice_settings['level_config'].name
            )
            # 액센트에 맞는 페르소나 선택
            persona = self._select_persona_for_accent(suggested_accent)
        else:
            persona = voice_settings['primary_persona']
        
        # 4. 스트리밍 음성 생성
        stream_queue = await self.streaming.stream_chatbot_voice(
            voice_settings['stream_id'],
            adjusted_response['adjusted_text'],
            persona,
            emotion,
            adjusted_response['voice_config']['speed']
        )
        
        # 5. 통계 업데이트
        self.system_stats['total_audio_generated'] += 1
        
        return stream_queue
    
    async def create_multi_speaker_dialogue(
        self,
        dialogue_script: List[Dict[str, Any]],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """다중 화자 대화 생성"""
        
        audio_segments = []
        voice_settings = await self.initialize_for_user(user_id)
        
        for turn in dialogue_script:
            speaker = turn['speaker']
            text = turn['text']
            emotion = turn.get('emotion', 'neutral')
            context = turn.get('context', '')
            
            # 화자별 다른 음성 사용
            if speaker == 'teacher':
                persona = 'teacher_anna'
            elif speaker == 'student':
                persona = 'buddy_tom'
            elif speaker == 'native_speaker':
                # 오늘의 액센트에 맞는 네이티브 스피커
                accent = voice_settings['today_accent']
                persona = self._select_persona_for_accent(accent)
            else:
                persona = turn.get('persona', voice_settings['primary_persona'])
            
            # 레벨 조정
            adjusted_response = await self.adaptive_controller.adjust_voice_for_level(
                text,
                voice_settings['level_config'].name,
                persona,
                user_id
            )
            
            # 음성 생성
            audio = await self.optimization.get_cached_or_generate(
                adjusted_response['adjusted_text'],
                persona,
                emotion
            )
            
            audio_segments.append({
                'speaker': speaker,
                'persona': persona,
                'audio': audio,
                'duration': self.calculate_duration(audio),
                'text': adjusted_response['adjusted_text'],
                'original_text': text,
                'emotion': emotion,
                'level_adjusted': adjusted_response['voice_config'],
                'metadata': {
                    'accent': self._get_persona_accent(persona),
                    'voice_characteristics': adjusted_response['voice_config']
                }
            })
        
        logger.info(f"Generated multi-speaker dialogue with {len(audio_segments)} segments")
        return audio_segments
    
    def calculate_duration(self, audio_bytes: bytes, sample_rate: int = 22050) -> float:
        """오디오 지속 시간 계산"""
        
        # bytes를 numpy array로 변환
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        duration = len(audio_array) / sample_rate
        return duration
    
    def _select_persona_for_accent(self, accent: str) -> str:
        """액센트에 맞는 페르소나 선택"""
        
        accent_persona_map = {
            'american': 'emily_us',
            'british': 'james_uk', 
            'australian': 'sarah_au',
            'canadian': 'michael_ca'
        }
        
        return accent_persona_map.get(accent, 'emily_us')
    
    def _get_persona_accent(self, persona: str) -> str:
        """페르소나의 액센트 반환"""
        
        persona_accent_map = {
            'emily_us': 'american',
            'james_uk': 'british',
            'sarah_au': 'australian', 
            'michael_ca': 'canadian',
            'teacher_anna': 'american',
            'buddy_tom': 'canadian'
        }
        
        return persona_accent_map.get(persona, 'american')
    
    def select_alternative_personas(self, user_level: str) -> List[str]:
        """레벨별 대안 페르소나 선택"""
        
        if user_level in ['beginner', 'elementary']:
            return ['teacher_anna', 'emily_us']
        elif user_level in ['intermediate', 'upper_intermediate']:
            return ['emily_us', 'james_uk', 'buddy_tom']
        else:  # advanced
            return ['emily_us', 'james_uk', 'sarah_au', 'michael_ca']
    
    async def load_user_profile(self, user_id: str) -> Dict[str, Any]:
        """사용자 프로필 로드 (시뮬레이션)"""
        
        # 실제로는 데이터베이스에서 로드
        return {
            'user_id': user_id,
            'preferred_voice': 'emily_us',
            'level': 'intermediate',
            'accent_variety': True,
            'use_emotions': True,
            'learning_goals': ['conversation', 'pronunciation'],
            'session_history': []
        }
    
    async def process_user_feedback(
        self,
        user_id: str,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """사용자 피드백 처리"""
        
        # 적응형 컨트롤러에 학습시키기
        await self.adaptive_controller.learn_from_feedback(user_id, feedback)
        
        # 페르소나 시스템에도 피드백 전달
        if user_id in self.active_sessions:
            session_history = []  # 실제로는 세션 히스토리 로드
            adaptations = await self.persona_system.adapt_persona_to_user(
                feedback, session_history
            )
            
            return {
                'message': 'Feedback processed successfully',
                'adaptive_changes': adaptations,
                'session_updated': True
            }
        
        return {'message': 'No active session found'}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        
        return {
            'system_stats': self.system_stats,
            'cache_stats': self.optimization.get_cache_stats(),
            'active_sessions': len(self.active_sessions),
            'streaming_stats': {
                'active_streams': len(self.streaming.active_streams),
                'total_streams_created': self.system_stats['total_sessions']
            },
            'persona_info': {
                'available_personas': len(self.persona_system.voice_personas),
                'current_selections': {
                    user_id: session.get('primary_persona')
                    for user_id, session in self.active_sessions.items()
                }
            }
        }
    
    async def cleanup_user_session(self, user_id: str):
        """사용자 세션 정리"""
        
        if user_id in self.active_sessions:
            session = self.active_sessions[user_id]
            
            # 스트림 정리
            if 'stream_id' in session:
                await self.streaming.stop_stream(session['stream_id'])
            
            # 세션 제거
            del self.active_sessions[user_id]
            
            logger.info(f"Cleaned up session for user {user_id}")
    
    async def shutdown(self):
        """시스템 종료"""
        
        # 모든 활성 스트림 중지
        for stream_id in list(self.streaming.active_streams.keys()):
            await self.streaming.stop_stream(stream_id)
        
        # 모든 세션 정리
        for user_id in list(self.active_sessions.keys()):
            await self.cleanup_user_session(user_id)
        
        logger.info("Complete voice system shutdown completed")