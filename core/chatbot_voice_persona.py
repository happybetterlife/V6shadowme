"""
챗봇 음성 페르소나 관리 시스템
다양한 네이티브 스피커 음성과 학습 도우미 특화 음성 제공
"""

import asyncio
import numpy as np
import torch
import librosa
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class PersonalityType(Enum):
    """음성 페르소나 성격 유형"""
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional" 
    ENERGETIC = "energetic"
    CALM = "calm"
    PATIENT = "patient"
    ENCOURAGING = "encouraging"

class AccentType(Enum):
    """영어 액센트 유형"""
    AMERICAN = "american"
    BRITISH = "british"
    AUSTRALIAN = "australian"
    CANADIAN = "canadian"
    INDIAN = "indian"
    SOUTH_AFRICAN = "south_african"

class EmotionState(Enum):
    """감정 상태"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    EXCITED = "excited"
    ENCOURAGING = "encouraging"
    PATIENT = "patient"
    SURPRISED = "surprised"
    CONCERNED = "concerned"

@dataclass
class VoiceCharacteristics:
    """음성 특성 데이터 클래스"""
    pitch: float  # Hz
    speed: float  # 배속 (0.5-2.0)
    energy: float  # 에너지 (0.0-1.0)
    clarity: float  # 명료도 (0.0-1.0)
    warmth: float = 0.8  # 따뜻함 (0.0-1.0)
    pause_duration: float = 1.0  # 문장 간 멈춤 시간
    breath_frequency: float = 0.3  # 숨 쉬는 빈도
    intonation_variance: float = 0.8  # 억양 변화

@dataclass
class VoicePersona:
    """음성 페르소나 정의"""
    name: str
    accent: AccentType
    age_group: str
    gender: str
    personality: PersonalityType
    role: str
    voice_model_path: str
    characteristics: VoiceCharacteristics
    emotional_range: List[EmotionState]
    specialty_contexts: List[str]
    sample_phrases: Dict[str, List[str]]

class EmotionExpressionEngine:
    """감정 표현 엔진"""
    
    def __init__(self):
        self.emotion_configs = self._initialize_emotion_configs()
        
    def _initialize_emotion_configs(self) -> Dict[EmotionState, Dict]:
        """감정별 음성 변조 설정"""
        return {
            EmotionState.HAPPY: {
                'pitch_shift': 1.15,
                'speed_modifier': 1.05,
                'energy_boost': 1.2,
                'vibrato_intensity': 0.3,
                'formant_shift': 1.1
            },
            EmotionState.EXCITED: {
                'pitch_shift': 1.25,
                'speed_modifier': 1.15,
                'energy_boost': 1.4,
                'vibrato_intensity': 0.4,
                'volume_variance': 1.3
            },
            EmotionState.ENCOURAGING: {
                'pitch_shift': 1.08,
                'speed_modifier': 0.95,
                'energy_boost': 1.1,
                'warmth_boost': 1.3,
                'clarity_boost': 1.05
            },
            EmotionState.PATIENT: {
                'pitch_shift': 0.95,
                'speed_modifier': 0.85,
                'energy_boost': 0.8,
                'pause_extension': 1.5,
                'clarity_boost': 1.2
            },
            EmotionState.CONCERNED: {
                'pitch_shift': 0.9,
                'speed_modifier': 0.9,
                'energy_boost': 0.7,
                'formant_shift': 0.95,
                'breathiness': 0.2
            },
            EmotionState.SURPRISED: {
                'pitch_shift': 1.3,
                'speed_modifier': 1.1,
                'energy_boost': 1.3,
                'attack_time': 0.8,
                'volume_spike': 1.4
            }
        }
    
    async def apply_emotion_to_audio(self, audio: np.ndarray, emotion: EmotionState, 
                                   sample_rate: int = 22050) -> np.ndarray:
        """오디오에 감정 표현 적용"""
        if emotion == EmotionState.NEUTRAL:
            return audio
            
        config = self.emotion_configs.get(emotion, {})
        modified_audio = audio.copy()
        
        # 피치 변조
        if 'pitch_shift' in config:
            modified_audio = librosa.effects.pitch_shift(
                modified_audio, sr=sample_rate, 
                n_steps=12 * np.log2(config['pitch_shift'])
            )
        
        # 속도 변조
        if 'speed_modifier' in config:
            modified_audio = librosa.effects.time_stretch(
                modified_audio, rate=config['speed_modifier']
            )
        
        # 에너지 부스트
        if 'energy_boost' in config:
            modified_audio = modified_audio * config['energy_boost']
        
        # 진폭 정규화
        max_val = np.max(np.abs(modified_audio))
        if max_val > 1.0:
            modified_audio = modified_audio / max_val
            
        return modified_audio
    
    def get_emotion_text_markers(self, emotion: EmotionState) -> Dict[str, str]:
        """감정에 따른 텍스트 마커 반환"""
        markers = {
            EmotionState.HAPPY: {
                'prefix': '*cheerfully* ',
                'intonation': '↗',
                'emphasis': '**'
            },
            EmotionState.EXCITED: {
                'prefix': '*excitedly* ',
                'intonation': '↗↗',
                'emphasis': '***'
            },
            EmotionState.ENCOURAGING: {
                'prefix': '*encouragingly* ',
                'intonation': '→↗',
                'emphasis': '*'
            },
            EmotionState.PATIENT: {
                'prefix': '*patiently* ',
                'intonation': '→',
                'pause_markers': '... '
            },
            EmotionState.CONCERNED: {
                'prefix': '*with concern* ',
                'intonation': '↘',
                'tone': 'gentle'
            },
            EmotionState.SURPRISED: {
                'prefix': '*surprised* ',
                'intonation': '↗!',
                'emphasis': '!'
            }
        }
        return markers.get(emotion, {})

class AccentLibrary:
    """액센트 라이브러리"""
    
    def __init__(self):
        self.accent_configs = self._initialize_accent_configs()
        self.phoneme_mappings = self._load_phoneme_mappings()
    
    def _initialize_accent_configs(self) -> Dict[AccentType, Dict]:
        """액센트별 음성 특성 설정"""
        return {
            AccentType.AMERICAN: {
                'r_coloring': True,
                'vowel_shifts': {'æ': 'ɛ', 'ɑ': 'ɔ'},
                'consonant_features': {'t_flapping': True},
                'rhythm': 'stress_timed',
                'intonation_pattern': 'falling_final'
            },
            AccentType.BRITISH: {
                'r_coloring': False,
                'vowel_shifts': {'æ': 'a', 'ɑ': 'ɒ'},
                'consonant_features': {'h_dropping': False, 'glottal_stop': True},
                'rhythm': 'stress_timed',
                'intonation_pattern': 'high_rise_terminal'
            },
            AccentType.AUSTRALIAN: {
                'r_coloring': False,
                'vowel_shifts': {'eɪ': 'aɪ', 'aɪ': 'ɔɪ'},
                'consonant_features': {'l_vocalization': True},
                'rhythm': 'stress_timed',
                'intonation_pattern': 'uptalk'
            },
            AccentType.CANADIAN: {
                'r_coloring': True,
                'vowel_shifts': {'aʊ': 'ʌʊ', 'aɪ': 'ʌɪ'},
                'consonant_features': {'canadian_raising': True},
                'rhythm': 'stress_timed',
                'intonation_pattern': 'eh_particle'
            }
        }
    
    def _load_phoneme_mappings(self) -> Dict[AccentType, Dict]:
        """음소 매핑 테이블 로드"""
        return {
            AccentType.AMERICAN: {
                'vowels': {
                    'DANCE': 'æ',  # /dæns/
                    'BATH': 'æ',   # /bæθ/
                    'PALM': 'ɑ',   # /pɑm/
                    'LOT': 'ɑ',    # /lɑt/
                    'THOUGHT': 'ɔ' # /θɔt/
                },
                'consonants': {
                    'r_final': True,    # car /kɑr/
                    't_flap': True,     # better /bɛɾər/
                    'ny_coalescence': True  # tune /tun/
                }
            },
            AccentType.BRITISH: {
                'vowels': {
                    'DANCE': 'ɑː',  # /dɑːns/
                    'BATH': 'ɑː',   # /bɑːθ/
                    'PALM': 'ɑː',   # /pɑːm/
                    'LOT': 'ɒ',     # /lɒt/
                    'THOUGHT': 'ɔː' # /θɔːt/
                },
                'consonants': {
                    'r_final': False,   # car /kɑː/
                    't_glottal': True,  # better /beʔə/
                    'yod_retention': True  # tune /tjuːn/
                }
            }
        }
    
    async def apply_accent_to_text(self, text: str, accent: AccentType) -> str:
        """텍스트에 액센트 특성 적용"""
        accent_config = self.accent_configs.get(accent, {})
        modified_text = text
        
        # 액센트별 발음 가이드 추가
        if accent == AccentType.BRITISH:
            modified_text = self._apply_british_pronunciation(modified_text)
        elif accent == AccentType.AUSTRALIAN:
            modified_text = self._apply_australian_pronunciation(modified_text)
        elif accent == AccentType.CANADIAN:
            modified_text = self._apply_canadian_pronunciation(modified_text)
            
        return modified_text
    
    def _apply_british_pronunciation(self, text: str) -> str:
        """영국식 발음 적용"""
        # R-dropping 시뮬레이션
        text = text.replace('car ', 'cah ')
        text = text.replace('here ', 'heah ')
        # BATH 단어들의 장모음화
        text = text.replace('dance', 'dahnce')
        text = text.replace('bath', 'bahth')
        return text
    
    def _apply_australian_pronunciation(self, text: str) -> str:
        """호주식 발음 적용"""
        # 이중모음 변화 시뮬레이션
        text = text.replace('day', 'die')
        text = text.replace('mate', 'mite')
        return text
    
    def _apply_canadian_pronunciation(self, text: str) -> str:
        """캐나다식 발음 적용"""
        # Canadian raising 시뮬레이션
        text = text.replace('about', 'aboot')
        text = text.replace('house', 'hoose')
        return text

class ChatbotVoicePersonaSystem:
    """챗봇 음성 페르소나 관리 시스템"""
    
    def __init__(self):
        self.voice_personas = self._initialize_voice_personas()
        self.emotion_engine = EmotionExpressionEngine()
        self.accent_library = AccentLibrary()
        self.current_persona: Optional[VoicePersona] = None
        self.context_history = []
        
    def _initialize_voice_personas(self) -> Dict[str, VoicePersona]:
        """음성 페르소나 초기화"""
        personas = {}
        
        # 네이티브 스피커 음성들
        personas['emily_us'] = VoicePersona(
            name='Emily',
            accent=AccentType.AMERICAN,
            age_group='young_adult',
            gender='female',
            personality=PersonalityType.FRIENDLY,
            role='conversation_partner',
            voice_model_path='models/openvoice_v2_emily.pth',
            characteristics=VoiceCharacteristics(
                pitch=220.0, speed=1.0, energy=0.8, clarity=0.95,
                warmth=0.9, pause_duration=0.8
            ),
            emotional_range=[
                EmotionState.HAPPY, EmotionState.ENCOURAGING, 
                EmotionState.SURPRISED, EmotionState.NEUTRAL
            ],
            specialty_contexts=['casual_conversation', 'daily_life', 'hobbies'],
            sample_phrases={
                'greeting': [
                    "Hey there! How's it going?",
                    "Hi! What's new with you today?",
                    "Hello! Ready for some English practice?"
                ],
                'encouragement': [
                    "You're doing great!",
                    "That's awesome progress!",
                    "Keep it up, you've got this!"
                ],
                'clarification': [
                    "Could you say that again?",
                    "I'm not sure I caught that.",
                    "Can you rephrase that for me?"
                ]
            }
        )
        
        personas['james_uk'] = VoicePersona(
            name='James',
            accent=AccentType.BRITISH,
            age_group='adult',
            gender='male',
            personality=PersonalityType.PROFESSIONAL,
            role='formal_instructor',
            voice_model_path='models/openvoice_v2_james.pth',
            characteristics=VoiceCharacteristics(
                pitch=120.0, speed=0.95, energy=0.7, clarity=0.98,
                warmth=0.6, pause_duration=1.2
            ),
            emotional_range=[
                EmotionState.NEUTRAL, EmotionState.ENCOURAGING,
                EmotionState.PATIENT, EmotionState.PROFESSIONAL
            ],
            specialty_contexts=['business_english', 'presentations', 'formal_writing'],
            sample_phrases={
                'greeting': [
                    "Good day! Shall we begin our session?",
                    "Right then, let's get started.",
                    "Excellent! I'm delighted to work with you today."
                ],
                'correction': [
                    "I beg your pardon, but there's a small error there.",
                    "Actually, the correct form would be...",
                    "May I suggest a slight adjustment?"
                ]
            }
        )
        
        personas['sarah_au'] = VoicePersona(
            name='Sarah',
            accent=AccentType.AUSTRALIAN,
            age_group='young_adult',
            gender='female',
            personality=PersonalityType.ENERGETIC,
            role='enthusiastic_coach',
            voice_model_path='models/openvoice_v2_sarah.pth',
            characteristics=VoiceCharacteristics(
                pitch=210.0, speed=1.05, energy=0.9, clarity=0.93,
                warmth=0.95, pause_duration=0.7
            ),
            emotional_range=[
                EmotionState.EXCITED, EmotionState.HAPPY,
                EmotionState.ENCOURAGING, EmotionState.ENERGETIC
            ],
            specialty_contexts=['sports', 'travel', 'adventure', 'motivation'],
            sample_phrases={
                'motivation': [
                    "No worries, mate! You'll get it!",
                    "Fair dinkum, that was brilliant!",
                    "You're a real champion!"
                ]
            }
        )
        
        # 학습 도우미 특화 음성들
        personas['teacher_anna'] = VoicePersona(
            name='Teacher Anna',
            accent=AccentType.AMERICAN,
            age_group='adult',
            gender='female',
            personality=PersonalityType.PATIENT,
            role='patient_teacher',
            voice_model_path='models/openvoice_v2_teacher.pth',
            characteristics=VoiceCharacteristics(
                pitch=200.0, speed=0.85, energy=0.7, clarity=1.0,
                warmth=1.0, pause_duration=1.5, breath_frequency=0.4
            ),
            emotional_range=[
                EmotionState.PATIENT, EmotionState.ENCOURAGING,
                EmotionState.NEUTRAL, EmotionState.CARING
            ],
            specialty_contexts=['grammar_lessons', 'pronunciation', 'beginner_level'],
            sample_phrases={
                'instruction': [
                    "Let's take this step by step.",
                    "Don't worry, we'll practice until you get it.",
                    "That's a great question. Let me explain..."
                ],
                'pronunciation_guide': [
                    "Listen carefully to the sound...",
                    "Now try to repeat after me...",
                    "Feel how your tongue moves..."
                ]
            }
        )
        
        personas['buddy_tom'] = VoicePersona(
            name='Buddy Tom',
            accent=AccentType.CANADIAN,
            age_group='young_adult',
            gender='male',
            personality=PersonalityType.ENCOURAGING,
            role='practice_partner',
            voice_model_path='models/openvoice_v2_buddy.pth',
            characteristics=VoiceCharacteristics(
                pitch=140.0, speed=1.0, energy=0.85, clarity=0.9,
                warmth=0.9, pause_duration=1.0
            ),
            emotional_range=[
                EmotionState.ENCOURAGING, EmotionState.FRIENDLY,
                EmotionState.HAPPY, EmotionState.SUPPORTIVE
            ],
            specialty_contexts=['conversation_practice', 'role_play', 'confidence_building'],
            sample_phrases={
                'practice': [
                    "Alright, let's give it a shot, eh?",
                    "You're getting better every time!",
                    "That's the spirit! Keep going!"
                ]
            }
        )
        
        return personas
    
    def get_available_personas(self) -> List[Dict[str, Any]]:
        """사용 가능한 페르소나 목록 반환"""
        return [
            {
                'id': persona_id,
                'name': persona.name,
                'accent': persona.accent.value,
                'personality': persona.personality.value,
                'role': persona.role,
                'gender': persona.gender,
                'age_group': persona.age_group,
                'specialty_contexts': persona.specialty_contexts,
                'sample_voice_url': f'/api/voice/sample/{persona_id}'
            }
            for persona_id, persona in self.voice_personas.items()
        ]
    
    async def select_persona(self, persona_id: str) -> bool:
        """페르소나 선택"""
        if persona_id in self.voice_personas:
            self.current_persona = self.voice_personas[persona_id]
            logger.info(f"Selected voice persona: {self.current_persona.name}")
            return True
        return False
    
    async def recommend_persona(self, context: str, user_level: str, 
                             learning_goals: List[str]) -> str:
        """상황에 맞는 페르소나 추천"""
        scores = {}
        
        for persona_id, persona in self.voice_personas.items():
            score = 0
            
            # 학습 레벨에 따른 점수
            if user_level == 'beginner' and persona.role == 'patient_teacher':
                score += 10
            elif user_level == 'intermediate' and persona.role in ['conversation_partner', 'practice_partner']:
                score += 8
            elif user_level == 'advanced' and persona.role == 'formal_instructor':
                score += 9
            
            # 컨텍스트 매칭
            for specialty in persona.specialty_contexts:
                if specialty in context.lower():
                    score += 5
            
            # 학습 목표 매칭
            for goal in learning_goals:
                if goal in persona.specialty_contexts:
                    score += 3
            
            scores[persona_id] = score
        
        # 최고 점수 페르소나 반환
        recommended = max(scores, key=scores.get)
        await self.select_persona(recommended)
        return recommended
    
    async def generate_persona_audio(self, text: str, emotion: EmotionState = EmotionState.NEUTRAL,
                                   context: str = "", user_id: str = None) -> Tuple[bytes, Dict[str, Any]]:
        """페르소나 기반 음성 생성"""
        if not self.current_persona:
            raise ValueError("No persona selected")
        
        # 액센트 적용
        accent_text = await self.accent_library.apply_accent_to_text(
            text, self.current_persona.accent
        )
        
        # 감정 마커 적용
        emotion_markers = self.emotion_engine.get_emotion_text_markers(emotion)
        if emotion_markers.get('prefix'):
            accent_text = emotion_markers['prefix'] + accent_text
        
        # 페르소나별 샘플 문구 추가 (컨텍스트에 따라)
        if context in self.current_persona.sample_phrases:
            sample_phrases = self.current_persona.sample_phrases[context]
            if random.random() < 0.3:  # 30% 확률로 샘플 문구 추가
                selected_phrase = random.choice(sample_phrases)
                accent_text = f"{selected_phrase} {accent_text}"
        
        # 음성 생성 (실제 OpenVoice 모델 호출은 voice_engine에서)
        # 여기서는 페르소나 메타데이터만 반환
        persona_metadata = {
            'persona_id': self.current_persona.name.lower().replace(' ', '_'),
            'persona_name': self.current_persona.name,
            'accent': self.current_persona.accent.value,
            'personality': self.current_persona.personality.value,
            'voice_characteristics': {
                'pitch': self.current_persona.characteristics.pitch,
                'speed': self.current_persona.characteristics.speed,
                'energy': self.current_persona.characteristics.energy,
                'clarity': self.current_persona.characteristics.clarity,
                'warmth': self.current_persona.characteristics.warmth,
                'pause_duration': self.current_persona.characteristics.pause_duration
            },
            'emotion_applied': emotion.value,
            'processed_text': accent_text,
            'original_text': text
        }
        
        return accent_text.encode(), persona_metadata
    
    async def get_persona_sample_phrases(self, context: str) -> List[str]:
        """현재 페르소나의 상황별 샘플 문구 반환"""
        if not self.current_persona:
            return []
        
        return self.current_persona.sample_phrases.get(context, [])
    
    async def adapt_persona_to_user(self, user_feedback: Dict[str, Any], 
                                  session_history: List[Dict]) -> Dict[str, Any]:
        """사용자 피드백에 따른 페르소나 적응"""
        if not self.current_persona:
            return {}
        
        adaptations = {}
        
        # 속도 조정
        if user_feedback.get('too_fast'):
            new_speed = max(0.7, self.current_persona.characteristics.speed - 0.1)
            self.current_persona.characteristics.speed = new_speed
            adaptations['speed'] = new_speed
        elif user_feedback.get('too_slow'):
            new_speed = min(1.3, self.current_persona.characteristics.speed + 0.1)
            self.current_persona.characteristics.speed = new_speed
            adaptations['speed'] = new_speed
        
        # 명료도 조정
        if user_feedback.get('unclear'):
            new_clarity = min(1.0, self.current_persona.characteristics.clarity + 0.05)
            self.current_persona.characteristics.clarity = new_clarity
            adaptations['clarity'] = new_clarity
        
        # 에너지 레벨 조정
        if user_feedback.get('too_energetic'):
            new_energy = max(0.5, self.current_persona.characteristics.energy - 0.1)
            self.current_persona.characteristics.energy = new_energy
            adaptations['energy'] = new_energy
        elif user_feedback.get('too_flat'):
            new_energy = min(1.0, self.current_persona.characteristics.energy + 0.1)
            self.current_persona.characteristics.energy = new_energy
            adaptations['energy'] = new_energy
        
        logger.info(f"Adapted persona {self.current_persona.name}: {adaptations}")
        return adaptations
    
    def get_current_persona_info(self) -> Optional[Dict[str, Any]]:
        """현재 페르소나 정보 반환"""
        if not self.current_persona:
            return None
        
        return {
            'name': self.current_persona.name,
            'accent': self.current_persona.accent.value,
            'personality': self.current_persona.personality.value,
            'role': self.current_persona.role,
            'characteristics': {
                'pitch': self.current_persona.characteristics.pitch,
                'speed': self.current_persona.characteristics.speed,
                'energy': self.current_persona.characteristics.energy,
                'clarity': self.current_persona.characteristics.clarity,
                'warmth': self.current_persona.characteristics.warmth
            },
            'emotional_range': [e.value for e in self.current_persona.emotional_range],
            'specialty_contexts': self.current_persona.specialty_contexts
        }