"""
음성 엔진 - OpenVoice + 골전도 시뮬레이션
"""

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from scipy import signal
import asyncio
from typing import Optional, Dict, Tuple, List
import whisper
from datetime import datetime
import logging

from .chatbot_voice_persona import ChatbotVoicePersonaSystem, EmotionState
from .complete_voice_system import CompleteChatbotVoiceSystem

logger = logging.getLogger(__name__)

class VoiceEngine:
    def __init__(self):
        self.sample_rate = 22050
        self.openvoice_model = None
        self.whisper_model = None
        self.user_voices = {}
        self.bone_conduction_params = self.init_bone_conduction_params()
        
        # 챗봇 음성 페르소나 시스템 추가
        self.persona_system = ChatbotVoicePersonaSystem()
        
        # 완전한 챗봇 음성 시스템 추가
        self.complete_voice_system = CompleteChatbotVoiceSystem()
        
    async def load_models(self):
        """음성 모델 로드"""
        # OpenVoice 모델 로드
        self.openvoice_model = self.load_openvoice_model()
        
        # Whisper 모델 로드
        self.whisper_model = whisper.load_model("base")
        
        logger.info("Voice models loaded successfully")
    
    def load_openvoice_model(self):
        """OpenVoice 모델 로드"""
        # OpenVoice V2 모델 로드 로직
        # 실제 구현 시 OpenVoice 공식 문서 참조
        from openvoice import OpenVoiceV2
        
        model = OpenVoiceV2(
            config_path="configs/openvoice_v2.json",
            checkpoint_path="checkpoints/openvoice_v2.pth"
        )
        return model
    
    async def setup_voice_cloning(
        self, 
        user_id: str, 
        audio_data: bytes
    ) -> Dict:
        """사용자 음성 클로닝 설정"""
        
        # 오디오 데이터 처리
        audio_array = self.bytes_to_array(audio_data)
        
        # 30초 샘플 추출
        sample = self.extract_30_second_sample(audio_array)
        
        # OpenVoice로 음성 특징 추출
        tone_color = await self.extract_tone_color(sample)
        speaker_embedding = await self.create_speaker_embedding(sample)
        
        # 사용자 음성 프로필 저장
        voice_profile = {
            'id': f"voice_{user_id}",
            'user_id': user_id,
            'tone_color': tone_color,
            'speaker_embedding': speaker_embedding,
            'created_at': datetime.now()
        }
        
        self.user_voices[user_id] = voice_profile
        
        # DB에 저장
        await self.save_voice_profile(voice_profile)
        
        return voice_profile
    
    async def generate_response_audio(
        self,
        text: str,
        user_id: str,
        emotion: str = 'neutral',
        use_bone_conduction: bool = True
    ) -> bytes:
        """응답 음성 생성"""
        
        # 기본 TTS 생성 (챗봇 음성)
        base_audio = await self.generate_chatbot_voice(text, emotion)
        
        # 골전도 시뮬레이션 적용 (선택적)
        if use_bone_conduction:
            processed_audio = self.apply_bone_conduction(base_audio, user_id)
        else:
            processed_audio = base_audio
        
        # 바이트로 변환
        audio_bytes = self.array_to_bytes(processed_audio)
        
        return audio_bytes
    
    async def generate_shadowing_audio(
        self,
        text: str,
        user_id: str,
        speed: float = 1.0
    ) -> bytes:
        """Shadowing용 사용자 음성 생성"""
        
        # 사용자 음성 프로필 로드
        voice_profile = self.user_voices.get(user_id)
        if not voice_profile:
            raise ValueError(f"No voice profile for user {user_id}")
        
        # OpenVoice로 사용자 음성 생성
        base_audio = await self.openvoice_model.tts(
            text=text,
            speaker_embedding=voice_profile['speaker_embedding'],
            speed=speed
        )
        
        # Tone Color Conversion
        user_voice_audio = await self.apply_tone_color(
            base_audio,
            voice_profile['tone_color']
        )
        
        # 골전도 시뮬레이션 적용
        natural_audio = self.apply_bone_conduction(
            user_voice_audio,
            user_id
        )
        
        return self.array_to_bytes(natural_audio)
    
    def apply_bone_conduction(
        self, 
        audio: np.ndarray, 
        user_id: str
    ) -> np.ndarray:
        """골전도 시뮬레이션 적용"""
        
        params = self.get_user_bone_conduction_params(user_id)
        
        # 주파수 대역별 EQ 필터링
        # Low-shelf filter (50-200Hz: +6~8dB)
        sos_low = signal.butter(
            2, 200, 'low', 
            fs=self.sample_rate, 
            output='sos'
        )
        audio_low = signal.sosfilt(sos_low, audio)
        audio_low *= self.db_to_amplitude(params['bass_gain'])
        
        # High-cut filter (2kHz 이상 감쇄)
        sos_high = signal.butter(
            2, params['high_cut_freq'], 
            'low', 
            fs=self.sample_rate, 
            output='sos'
        )
        audio_filtered = signal.sosfilt(sos_high, audio)
        
        # 다중 레이어 믹싱
        # Track A: 원본 (70%)
        track_a = audio * (1 - params['mix_ratio'])
        
        # Track B: 골전도 시뮬레이션 (30%)
        track_b = audio_filtered * params['mix_ratio']
        
        # 최종 믹싱
        final_audio = track_a + track_b
        
        # Normalize
        final_audio = self.normalize_audio(final_audio)
        
        return final_audio
    
    async def transcribe(self, audio_data: bytes) -> str:
        """음성을 텍스트로 변환"""
        
        # 바이트를 numpy 배열로 변환
        audio_array = self.bytes_to_array(audio_data)
        
        # Whisper로 전사
        result = self.whisper_model.transcribe(
            audio_array,
            language='en'
        )
        
        return result['text']
    
    def init_bone_conduction_params(self) -> Dict:
        """골전도 파라미터 초기화"""
        return {
            'bass_gain': 6.0,
            'warmth': 3.0,
            'high_cut_freq': 4000,
            'mix_ratio': 0.3,
            'resonance_freq': (100, 300)
        }
    
    def get_user_bone_conduction_params(self, user_id: str) -> Dict:
        """사용자별 골전도 파라미터"""
        # DB에서 사용자 설정 로드
        # 기본값 반환
        return self.bone_conduction_params
    
    def db_to_amplitude(self, db: float) -> float:
        """dB를 amplitude로 변환"""
        return 10 ** (db / 20.0)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """오디오 정규화"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95
        return audio
    
    def bytes_to_array(self, audio_bytes: bytes) -> np.ndarray:
        """바이트를 numpy 배열로 변환"""
        return np.frombuffer(audio_bytes, dtype=np.float32)
    
    def array_to_bytes(self, audio_array: np.ndarray) -> bytes:
        """numpy 배열을 바이트로 변환"""
        return audio_array.astype(np.float32).tobytes()
    
    def extract_30_second_sample(self, audio_array: np.ndarray) -> np.ndarray:
        """30초 샘플 추출"""
        target_length = 30 * self.sample_rate
        
        if len(audio_array) > target_length:
            # 중간 부분에서 30초 추출
            start = (len(audio_array) - target_length) // 2
            return audio_array[start:start + target_length]
        else:
            # 30초보다 짧으면 패딩
            padded = np.zeros(target_length)
            padded[:len(audio_array)] = audio_array
            return padded
    
    async def extract_tone_color(self, audio_sample: np.ndarray) -> Dict:
        """음색 특징 추출"""
        # OpenVoice tone color extraction
        # 실제 구현은 OpenVoice API에 따라 수정
        tone_features = {
            'pitch': self.extract_pitch_features(audio_sample),
            'timbre': self.extract_timbre_features(audio_sample),
            'rhythm': self.extract_rhythm_features(audio_sample),
            'energy': self.extract_energy_features(audio_sample)
        }
        return tone_features
    
    async def create_speaker_embedding(self, audio_sample: np.ndarray) -> torch.Tensor:
        """스피커 임베딩 생성"""
        # OpenVoice speaker embedding
        # 실제 구현은 OpenVoice API에 따라 수정
        if self.openvoice_model:
            embedding = await self.openvoice_model.encode_speaker(audio_sample)
            return embedding
        else:
            # 더미 임베딩 반환 (실제 구현 시 제거)
            return torch.randn(256)
    
    async def generate_chatbot_voice(self, text: str, emotion: str) -> np.ndarray:
        """챗봇 기본 음성 생성"""
        # OpenVoice TTS로 챗봇 음성 생성
        # 감정 파라미터 적용
        emotion_params = self.get_emotion_params(emotion)
        
        if self.openvoice_model:
            audio = await self.openvoice_model.tts(
                text=text,
                speaker='chatbot',
                emotion=emotion_params
            )
            return audio
        else:
            # 더미 오디오 반환 (실제 구현 시 제거)
            duration = len(text) * 0.1 * self.sample_rate
            return np.random.randn(int(duration)) * 0.1
    
    async def apply_tone_color(
        self, 
        audio: np.ndarray, 
        tone_color: Dict
    ) -> np.ndarray:
        """톤 컬러 적용"""
        # OpenVoice tone color conversion
        if self.openvoice_model:
            converted = await self.openvoice_model.convert_tone_color(
                audio=audio,
                target_tone_color=tone_color
            )
            return converted
        else:
            # 간단한 필터링 시뮬레이션
            return audio * 0.9
    
    async def save_voice_profile(self, voice_profile: Dict):
        """음성 프로필 DB 저장"""
        # 실제 DB 저장 로직 구현
        logger.info(f"Voice profile saved for user {voice_profile['user_id']}")
    
    def extract_pitch_features(self, audio: np.ndarray) -> Dict:
        """피치 특징 추출"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        return {
            'mean_f0': np.nanmean(f0),
            'std_f0': np.nanstd(f0),
            'range_f0': (np.nanmin(f0), np.nanmax(f0)),
            'voiced_ratio': np.mean(voiced_flag)
        }
    
    def extract_timbre_features(self, audio: np.ndarray) -> Dict:
        """음색 특징 추출"""
        # MFCC 특징 추출
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13
        )
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate
        )
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio,
            sr=self.sample_rate
        )
        
        return {
            'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
            'mfcc_std': np.std(mfcc, axis=1).tolist(),
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_rolloff': np.mean(spectral_rolloff)
        }
    
    def extract_rhythm_features(self, audio: np.ndarray) -> Dict:
        """리듬 특징 추출"""
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(
            y=audio,
            sr=self.sample_rate
        )
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate
        )
        
        return {
            'tempo': tempo,
            'beat_times': beats.tolist() if isinstance(beats, np.ndarray) else beats,
            'onset_strength_mean': np.mean(onset_env),
            'onset_strength_std': np.std(onset_env)
        }
    
    def extract_energy_features(self, audio: np.ndarray) -> Dict:
        """에너지 특징 추출"""
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        return {
            'rms_mean': np.mean(rms),
            'rms_std': np.std(rms),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr),
            'dynamic_range': np.max(rms) - np.min(rms)
        }
    
    def get_emotion_params(self, emotion: str) -> Dict:
        """감정별 음성 파라미터"""
        emotion_map = {
            'neutral': {
                'pitch_shift': 0,
                'speed': 1.0,
                'energy': 1.0,
                'tone_variance': 0.5
            },
            'happy': {
                'pitch_shift': 2,
                'speed': 1.1,
                'energy': 1.2,
                'tone_variance': 0.8
            },
            'sad': {
                'pitch_shift': -2,
                'speed': 0.9,
                'energy': 0.8,
                'tone_variance': 0.3
            },
            'excited': {
                'pitch_shift': 3,
                'speed': 1.2,
                'energy': 1.3,
                'tone_variance': 0.9
            },
            'calm': {
                'pitch_shift': -1,
                'speed': 0.95,
                'energy': 0.9,
                'tone_variance': 0.4
            }
        }
        
        return emotion_map.get(emotion, emotion_map['neutral'])
    
    async def process_realtime_audio(
        self,
        audio_stream: asyncio.Queue,
        user_id: str
    ) -> asyncio.Queue:
        """실시간 오디오 처리"""
        output_queue = asyncio.Queue()
        
        while True:
            try:
                # 입력 오디오 청크 받기
                audio_chunk = await audio_stream.get()
                
                if audio_chunk is None:
                    break
                
                # 골전도 시뮬레이션 적용
                processed_chunk = self.apply_bone_conduction(
                    audio_chunk,
                    user_id
                )
                
                # 출력 큐에 전송
                await output_queue.put(processed_chunk)
                
            except Exception as e:
                logger.error(f"Error processing realtime audio: {e}")
                break
        
        return output_queue
    
    # ===== 페르소나 기반 음성 생성 메서드들 =====
    
    async def generate_persona_audio(
        self,
        text: str,
        user_id: str,
        persona_id: str = None,
        emotion: str = "neutral",
        context: str = "",
        use_bone_conduction: bool = True
    ) -> bytes:
        """페르소나 기반 음성 생성"""
        
        # 페르소나 선택 (추천 시스템 사용)
        if persona_id:
            await self.persona_system.select_persona(persona_id)
        else:
            # 자동 추천 (사용자 레벨과 컨텍스트 기반)
            user_level = await self.get_user_level(user_id)
            learning_goals = await self.get_user_learning_goals(user_id)
            await self.persona_system.recommend_persona(context, user_level, learning_goals)
        
        # 감정 상태 변환
        emotion_state = self._convert_to_emotion_state(emotion)
        
        # 페르소나 기반 텍스트 처리 및 메타데이터 생성
        processed_text_bytes, persona_metadata = await self.persona_system.generate_persona_audio(
            text=text,
            emotion=emotion_state,
            context=context,
            user_id=user_id
        )
        
        processed_text = processed_text_bytes.decode()
        
        # OpenVoice로 실제 음성 생성
        if self.openvoice_model:
            # 페르소나 특성 적용
            characteristics = persona_metadata['voice_characteristics']
            
            # 페르소나별 음성 생성 파라미터 설정
            generation_params = {
                'speed': characteristics['speed'],
                'pitch_shift': self._pitch_to_semitones(characteristics['pitch']),
                'energy': characteristics['energy'],
                'clarity': characteristics['clarity']
            }
            
            # OpenVoice 음성 생성 (시뮬레이션)
            audio = await self._generate_with_openvoice(
                text=processed_text,
                params=generation_params,
                persona_metadata=persona_metadata
            )
        else:
            # 시뮬레이션 오디오 (개발용)
            audio = self._generate_simulation_audio(processed_text)
        
        # 감정 효과 적용
        audio_with_emotion = await self.persona_system.emotion_engine.apply_emotion_to_audio(
            audio, emotion_state, self.sample_rate
        )
        
        # 골전도 시뮬레이션 적용
        if use_bone_conduction:
            final_audio = self.apply_bone_conduction(audio_with_emotion, user_id)
        else:
            final_audio = audio_with_emotion
        
        # 정규화
        final_audio = self.normalize_audio(final_audio)
        
        logger.info(f"Generated persona audio: {persona_metadata['persona_name']} ({emotion})")
        return self.array_to_bytes(final_audio)
    
    def _convert_to_emotion_state(self, emotion: str) -> EmotionState:
        """문자열 감정을 EmotionState로 변환"""
        emotion_mapping = {
            'neutral': EmotionState.NEUTRAL,
            'happy': EmotionState.HAPPY,
            'excited': EmotionState.EXCITED,
            'encouraging': EmotionState.ENCOURAGING,
            'patient': EmotionState.PATIENT,
            'surprised': EmotionState.SURPRISED,
            'concerned': EmotionState.CONCERNED
        }
        return emotion_mapping.get(emotion.lower(), EmotionState.NEUTRAL)
    
    def _pitch_to_semitones(self, pitch_hz: float) -> float:
        """주파수(Hz)를 반음 단위로 변환"""
        # 기준 주파수 (A4 = 440Hz)를 기준으로 반음 계산
        reference_hz = 220.0  # 중간 음성 주파수
        semitones = 12 * np.log2(pitch_hz / reference_hz)
        return semitones
    
    async def _generate_with_openvoice(
        self,
        text: str,
        params: Dict,
        persona_metadata: Dict
    ) -> np.ndarray:
        """OpenVoice로 음성 생성"""
        if self.openvoice_model:
            # 실제 OpenVoice 모델 호출
            # 페르소나별 voice model path 사용
            voice_model_path = persona_metadata.get('voice_model_path', 'default')
            
            # OpenVoice tone color 적용
            tone_color = await self.load_persona_tone_color(persona_metadata['persona_id'])
            
            audio = await self.openvoice_model.generate(
                text=text,
                tone_color=tone_color,
                speed=params['speed'],
                pitch_shift=params['pitch_shift']
            )
            return audio
        else:
            return self._generate_simulation_audio(text)
    
    def _generate_simulation_audio(self, text: str) -> np.ndarray:
        """시뮬레이션 오디오 생성 (개발용)"""
        # 간단한 톤 생성 (실제로는 OpenVoice 사용)
        duration = len(text) * 0.08  # 대략적인 읽기 시간
        t = np.linspace(0, duration, int(duration * self.sample_rate))
        
        # 기본 톤 생성 (사인파 + 노이즈)
        frequency = 200 + np.random.random() * 100  # 200-300Hz 범위
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        audio += 0.1 * np.random.random(len(t)) - 0.05  # 노이즈 추가
        
        return audio
    
    async def load_persona_tone_color(self, persona_id: str) -> Dict:
        """페르소나별 tone color 로드"""
        # 실제로는 사전 학습된 페르소나 tone color를 로드
        # 현재는 기본값 반환
        return {
            'speaker_embedding': np.random.random(256).tolist(),
            'tone_color_embedding': np.random.random(256).tolist()
        }
    
    async def get_user_level(self, user_id: str) -> str:
        """사용자 레벨 조회 (DB에서)"""
        # 실제로는 database에서 조회
        return "intermediate"
    
    async def get_user_learning_goals(self, user_id: str) -> List[str]:
        """사용자 학습 목표 조회"""
        # 실제로는 database에서 조회
        return ["conversation", "pronunciation", "business_english"]
    
    async def get_available_personas(self) -> List[Dict]:
        """사용 가능한 페르소나 목록 반환"""
        return self.persona_system.get_available_personas()
    
    async def select_voice_persona(self, persona_id: str) -> bool:
        """음성 페르소나 선택"""
        return await self.persona_system.select_persona(persona_id)
    
    async def get_current_persona_info(self) -> Optional[Dict]:
        """현재 페르소나 정보 반환"""
        return self.persona_system.get_current_persona_info()
    
    async def adapt_persona_to_feedback(self, user_feedback: Dict, session_history: List[Dict]) -> Dict:
        """사용자 피드백에 따른 페르소나 적응"""
        return await self.persona_system.adapt_persona_to_user(user_feedback, session_history)
    
    async def get_persona_sample_phrases(self, context: str) -> List[str]:
        """페르소나별 샘플 문구 반환"""
        return await self.persona_system.get_persona_sample_phrases(context)
    
    # ===== 완전한 음성 시스템 메서드들 =====
    
    async def initialize_complete_voice_system(self, user_id: str) -> Dict[str, Any]:
        """완전한 음성 시스템 초기화"""
        return await self.complete_voice_system.initialize_for_user(user_id)
    
    async def speak_response_streaming(
        self,
        text: str,
        user_id: str,
        emotion: str = 'neutral',
        immediate: bool = False,
        context: str = ''
    ) -> Any:  # asyncio.Queue
        """스트리밍 응답 음성 재생"""
        return await self.complete_voice_system.speak_response(
            text, user_id, emotion, immediate, context
        )
    
    async def create_multi_speaker_dialogue(
        self,
        dialogue_script: List[Dict[str, Any]],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """다중 화자 대화 생성"""
        return await self.complete_voice_system.create_multi_speaker_dialogue(
            dialogue_script, user_id
        )
    
    async def process_voice_feedback(
        self,
        user_id: str,
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """음성 피드백 처리"""
        return await self.complete_voice_system.process_user_feedback(user_id, feedback)
    
    async def get_voice_system_status(self) -> Dict[str, Any]:
        """음성 시스템 상태 조회"""
        return await self.complete_voice_system.get_system_status()
    
    async def get_accent_diversity_report(self, user_id: str) -> Dict[str, Any]:
        """액센트 다양성 리포트"""
        return await self.complete_voice_system.accent_system.get_accent_diversity_report(user_id)
    
    async def get_today_accent(self, user_id: str, user_level: str) -> str:
        """오늘의 액센트"""
        return await self.complete_voice_system.accent_system.get_today_accent(user_id, user_level)
    
    async def create_voice_stream(self, user_id: str, session_id: str) -> str:
        """음성 스트림 생성"""
        return await self.complete_voice_system.streaming.create_stream(user_id, session_id)
    
    async def pause_voice_stream(self, stream_id: str):
        """음성 스트림 일시정지"""
        return await self.complete_voice_system.streaming.pause_stream(stream_id)
    
    async def resume_voice_stream(self, stream_id: str):
        """음성 스트림 재개"""
        return await self.complete_voice_system.streaming.resume_stream(stream_id)
    
    async def stop_voice_stream(self, stream_id: str):
        """음성 스트림 중지"""
        return await self.complete_voice_system.streaming.stop_stream(stream_id)
    
    def get_voice_stream_status(self, stream_id: str) -> Optional[Dict]:
        """음성 스트림 상태 조회"""
        return self.complete_voice_system.streaming.get_stream_status(stream_id)
    
    async def cleanup_user_voice_session(self, user_id: str):
        """사용자 음성 세션 정리"""
        return await self.complete_voice_system.cleanup_user_session(user_id)
    
    def cleanup(self):
        """리소스 정리"""
        self.openvoice_model = None
        self.whisper_model = None
        self.user_voices.clear()
        logger.info("Voice engine resources cleaned up")