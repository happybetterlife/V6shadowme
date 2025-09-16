"""
AI 영어 학습 앱 메인 애플리케이션
"""

import asyncio
import logging
import base64
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import io

from config.settings import Settings
from core.conversation_engine import ConversationEngine
from core.voice_engine import VoiceEngine
from core.learning_engine import LearningEngine
from core.interactive_voice_response import InteractiveVoiceResponse
from agents.orchestrator import AgentOrchestrator
from database.manager import DatabaseManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(
    title="AI English Learning App",
    description="Voice Shadow - AI 영어 학습 앱",
    version="1.0.0"
)

# 전역 설정 및 엔진 초기화
settings = Settings()
db_manager = DatabaseManager()
conversation_engine = ConversationEngine()
voice_engine = VoiceEngine()
learning_engine = LearningEngine()
agent_orchestrator = AgentOrchestrator(openai_api_key=settings.openai_api_key)
# Interactive Voice Response 시스템
ivr_system = InteractiveVoiceResponse(voice_engine, agent_orchestrator)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic 모델
class UserRegistration(BaseModel):
    username: str
    email: str
    native_language: str = "Korean"
    target_level: str = "intermediate"

class ConversationRequest(BaseModel):
    user_id: str
    message: str
    audio_data: Optional[bytes] = None

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime
    learning_materials: dict
    estimated_duration: int

class VoiceSetupRequest(BaseModel):
    user_id: str
    audio_base64: str
    sample_rate: int = 22050

class TextToSpeechRequest(BaseModel):
    text: str
    user_id: str
    mode: str = "shadowing"  # "shadowing" or "response"
    emotion: str = "neutral"
    speed: float = 1.0
    use_bone_conduction: bool = True

class TranscriptionRequest(BaseModel):
    audio_base64: str
    language: str = "en"

class FeedbackRequest(BaseModel):
    session_id: str
    performance_score: float
    difficulty_rating: int  # 1-5 scale
    enjoyed_activities: list = []
    struggled_areas: list = []
    comments: str = ""

# 시작 시 초기화
@app.on_event("startup")
async def startup_event():
    """앱 시작 시 초기화"""
    logger.info("Initializing AI English Learning App...")
    
    try:
        # 데이터베이스 연결
        await db_manager.initialize()
        
        # 에이전트 시스템 초기화
        await agent_orchestrator.initialize()
        
        # 음성 모델 로드
        await voice_engine.load_models()
        
        # 학습 데이터베이스 로드
        await learning_engine.load_databases()
        
        # 대화 엔진 데이터베이스 초기화
        await conversation_engine.initialize_databases()
        
        logger.info("✅ Initialization complete!")
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료 시 정리"""
    await db_manager.close()
    voice_engine.cleanup()
    logger.info("Application shutdown complete")

# API 엔드포인트

# ===== 사용자 관리 =====
@app.post("/api/register")
async def register_user(user: UserRegistration):
    """새 사용자 등록"""
    try:
        user_id = await db_manager.create_user(
            username=user.username,
            email=user.email,
            native_language=user.native_language,
            target_level=user.target_level
        )
        
        logger.info(f"New user registered: {user_id}")
        return {
            "user_id": user_id, 
            "message": "Registration successful",
            "next_step": "voice_setup"
        }
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/user/{user_id}/profile")
async def get_user_profile(user_id: str):
    """사용자 프로필 조회"""
    try:
        profile = await db_manager.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        return profile
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ===== 음성 페르소나 관리 =====
@app.get("/api/voice/personas")
async def get_available_personas():
    """사용 가능한 음성 페르소나 목록 조회"""
    try:
        personas = await voice_engine.get_available_personas()
        return {"personas": personas}
    except Exception as e:
        logger.error(f"Persona list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/persona/select")
async def select_voice_persona(request: dict):
    """음성 페르소나 선택"""
    try:
        persona_id = request.get('persona_id')
        if not persona_id:
            raise HTTPException(status_code=400, detail="persona_id is required")
        
        success = await voice_engine.select_voice_persona(persona_id)
        if success:
            persona_info = await voice_engine.get_current_persona_info()
            return {
                "message": "Persona selected successfully",
                "persona": persona_info
            }
        else:
            raise HTTPException(status_code=404, detail="Persona not found")
    except Exception as e:
        logger.error(f"Persona selection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/voice/persona/current")
async def get_current_persona():
    """현재 선택된 페르소나 정보 조회"""
    try:
        persona_info = await voice_engine.get_current_persona_info()
        if persona_info:
            return {"persona": persona_info}
        else:
            return {"persona": None, "message": "No persona selected"}
    except Exception as e:
        logger.error(f"Current persona error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/persona/tts")
async def persona_text_to_speech(request: dict):
    """페르소나 기반 텍스트 음성 변환"""
    try:
        text = request.get('text')
        user_id = request.get('user_id')
        persona_id = request.get('persona_id')
        emotion = request.get('emotion', 'neutral')
        context = request.get('context', '')
        use_bone_conduction = request.get('use_bone_conduction', True)
        
        if not text or not user_id:
            raise HTTPException(status_code=400, detail="text and user_id are required")
        
        # 페르소나 기반 음성 생성
        audio_bytes = await voice_engine.generate_persona_audio(
            text=text,
            user_id=user_id,
            persona_id=persona_id,
            emotion=emotion,
            context=context,
            use_bone_conduction=use_bone_conduction
        )
        
        # 오디오 스트림 반환
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=persona_tts.wav"}
        )
        
    except Exception as e:
        logger.error(f"Persona TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voice/persona/samples/{context}")
async def get_persona_sample_phrases(context: str):
    """페르소나별 상황별 샘플 문구 조회"""
    try:
        phrases = await voice_engine.get_persona_sample_phrases(context)
        return {"context": context, "phrases": phrases}
    except Exception as e:
        logger.error(f"Sample phrases error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/persona/feedback")
async def submit_persona_feedback(request: dict):
    """페르소나 피드백 제출 (적응형 학습)"""
    try:
        user_feedback = request.get('feedback', {})
        session_history = request.get('session_history', [])
        
        adaptations = await voice_engine.adapt_persona_to_feedback(
            user_feedback, session_history
        )
        
        return {
            "message": "Feedback processed successfully",
            "adaptations": adaptations
        }
    except Exception as e:
        logger.error(f"Persona feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== 완전한 음성 시스템 API =====
@app.post("/api/voice/system/initialize")
async def initialize_complete_voice_system(request: dict):
    """완전한 음성 시스템 초기화"""
    try:
        user_id = request.get('user_id')
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        voice_settings = await voice_engine.initialize_complete_voice_system(user_id)
        
        return {
            "message": "Voice system initialized successfully",
            "voice_settings": voice_settings
        }
    except Exception as e:
        logger.error(f"Voice system initialization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/stream/speak")
async def streaming_voice_response(request: dict):
    """스트리밍 음성 응답"""
    try:
        text = request.get('text')
        user_id = request.get('user_id')
        emotion = request.get('emotion', 'neutral')
        immediate = request.get('immediate', False)
        context = request.get('context', '')
        
        if not text or not user_id:
            raise HTTPException(status_code=400, detail="text and user_id are required")
        
        # 스트리밍 큐 반환 (실제로는 WebSocket 또는 Server-Sent Events 사용)
        stream_queue = await voice_engine.speak_response_streaming(
            text, user_id, emotion, immediate, context
        )
        
        return {
            "message": "Streaming started",
            "stream_info": "Use WebSocket for real-time audio streaming"
        }
        
    except Exception as e:
        logger.error(f"Streaming voice error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/dialogue/create")
async def create_multi_speaker_dialogue(request: dict):
    """다중 화자 대화 생성"""
    try:
        dialogue_script = request.get('dialogue_script', [])
        user_id = request.get('user_id')
        
        if not dialogue_script or not user_id:
            raise HTTPException(status_code=400, detail="dialogue_script and user_id are required")
        
        audio_segments = await voice_engine.create_multi_speaker_dialogue(
            dialogue_script, user_id
        )
        
        return {
            "message": "Multi-speaker dialogue created successfully",
            "segments": audio_segments,
            "total_segments": len(audio_segments),
            "total_duration": sum(seg['duration'] for seg in audio_segments)
        }
        
    except Exception as e:
        logger.error(f"Multi-speaker dialogue error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/feedback/process")
async def process_voice_feedback(request: dict):
    """음성 피드백 처리 (완전한 시스템)"""
    try:
        user_id = request.get('user_id')
        feedback = request.get('feedback', {})
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        result = await voice_engine.process_voice_feedback(user_id, feedback)
        
        return result
        
    except Exception as e:
        logger.error(f"Voice feedback processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voice/system/status")
async def get_voice_system_status():
    """음성 시스템 상태 조회"""
    try:
        status = await voice_engine.get_voice_system_status()
        return status
    except Exception as e:
        logger.error(f"Voice system status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voice/accent/report/{user_id}")
async def get_accent_diversity_report(user_id: str):
    """액센트 다양성 리포트"""
    try:
        report = await voice_engine.get_accent_diversity_report(user_id)
        return {"user_id": user_id, "accent_report": report}
    except Exception as e:
        logger.error(f"Accent report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voice/accent/today/{user_id}")
async def get_today_accent(user_id: str, user_level: str = "intermediate"):
    """오늘의 액센트 조회"""
    try:
        accent = await voice_engine.get_today_accent(user_id, user_level)
        return {
            "user_id": user_id,
            "user_level": user_level,
            "today_accent": accent
        }
    except Exception as e:
        logger.error(f"Today accent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== 음성 스트림 관리 =====
@app.post("/api/voice/stream/create")
async def create_voice_stream(request: dict):
    """음성 스트림 생성"""
    try:
        user_id = request.get('user_id')
        session_id = request.get('session_id')
        
        if not user_id or not session_id:
            raise HTTPException(status_code=400, detail="user_id and session_id are required")
        
        stream_id = await voice_engine.create_voice_stream(user_id, session_id)
        
        return {
            "message": "Voice stream created successfully",
            "stream_id": stream_id
        }
    except Exception as e:
        logger.error(f"Voice stream creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/stream/{stream_id}/pause")
async def pause_voice_stream(stream_id: str):
    """음성 스트림 일시정지"""
    try:
        await voice_engine.pause_voice_stream(stream_id)
        return {"message": f"Stream {stream_id} paused successfully"}
    except Exception as e:
        logger.error(f"Voice stream pause error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/stream/{stream_id}/resume")
async def resume_voice_stream(stream_id: str):
    """음성 스트림 재개"""
    try:
        await voice_engine.resume_voice_stream(stream_id)
        return {"message": f"Stream {stream_id} resumed successfully"}
    except Exception as e:
        logger.error(f"Voice stream resume error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/stream/{stream_id}/stop")
async def stop_voice_stream(stream_id: str):
    """음성 스트림 중지"""
    try:
        await voice_engine.stop_voice_stream(stream_id)
        return {"message": f"Stream {stream_id} stopped successfully"}
    except Exception as e:
        logger.error(f"Voice stream stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voice/stream/{stream_id}/status")
async def get_voice_stream_status(stream_id: str):
    """음성 스트림 상태 조회"""
    try:
        status = voice_engine.get_voice_stream_status(stream_id)
        if status:
            return {"stream_id": stream_id, "status": status}
        else:
            raise HTTPException(status_code=404, detail="Stream not found")
    except Exception as e:
        logger.error(f"Voice stream status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/voice/session/{user_id}")
async def cleanup_voice_session(user_id: str):
    """음성 세션 정리"""
    try:
        await voice_engine.cleanup_user_voice_session(user_id)
        return {"message": f"Voice session for user {user_id} cleaned up successfully"}
    except Exception as e:
        logger.error(f"Voice session cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== 음성 관련 =====
@app.post("/api/voice/setup")
async def setup_voice_cloning(request: VoiceSetupRequest):
    """사용자 음성 클로닝 설정 (30초 녹음)"""
    try:
        # Base64 디코딩
        audio_data = base64.b64decode(request.audio_base64)
        
        # OpenVoice로 음성 클로닝
        voice_profile = await voice_engine.setup_voice_cloning(
            user_id=request.user_id,
            audio_data=audio_data
        )
        
        # 데이터베이스에 저장
        await db_manager.save_voice_profile(
            user_id=request.user_id,
            profile_id=voice_profile['id'],
            audio_sample_path=f"voice_samples/{request.user_id}.wav",
            voice_features=voice_profile['tone_color']
        )
        
        return {
            "message": "Voice cloning setup complete",
            "voice_profile_id": voice_profile['id'],
            "features_extracted": len(voice_profile.get('tone_color', {}))
        }
    except Exception as e:
        logger.error(f"Voice setup error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/voice/tts")
async def text_to_speech(request: TextToSpeechRequest):
    """텍스트를 음성으로 변환 (음성 클로닝 사용)"""
    try:
        # TTS 생성
        if request.mode == "shadowing":
            audio_bytes = await voice_engine.generate_shadowing_audio(
                text=request.text,
                user_id=request.user_id,
                speed=request.speed
            )
        else:  # response mode
            audio_bytes = await voice_engine.generate_response_audio(
                text=request.text,
                user_id=request.user_id,
                emotion=request.emotion,
                use_bone_conduction=request.use_bone_conduction
            )
        
        # 오디오 스트림 반환
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/transcribe")
async def transcribe_audio(request: TranscriptionRequest):
    """음성을 텍스트로 변환"""
    try:
        # Base64 디코딩
        audio_data = base64.b64decode(request.audio_base64)
        
        # Whisper로 전사
        start_time = datetime.now()
        text = await voice_engine.transcribe(audio_data)
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "text": text,
            "confidence": 0.95,  # Placeholder
            "duration": duration,
            "language": request.language
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== 학습 세션 관리 =====
@app.post("/api/session/create/{user_id}")
async def create_learning_session(user_id: str) -> SessionResponse:
    """일일 학습 세션 생성"""
    try:
        # 학습 세션 생성
        session = await learning_engine.create_daily_session(user_id)
        
        # 데이터베이스에 세션 기록
        db_session_id = await db_manager.create_learning_session(
            user_id=user_id,
            session_type="daily_practice"
        )
        
        session['db_session_id'] = db_session_id
        
        return SessionResponse(
            session_id=session['id'],
            user_id=user_id,
            created_at=session['created_at'],
            learning_materials=session['materials'],
            estimated_duration=session['duration']
        )
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/shadowing/{user_id}/{session_id}")
async def get_shadowing_materials(user_id: str, session_id: str):
    """Shadowing 연습 자료 가져오기"""
    try:
        materials = await learning_engine.get_shadowing_materials(
            user_id=user_id,
            session_id=session_id
        )
        
        return {
            "materials": materials,
            "total_sentences": len(materials),
            "difficulty_levels": ["beginner", "intermediate", "advanced"],
            "practice_modes": ["listen", "repeat", "shadow"]
        }
    except Exception as e:
        logger.error(f"Shadowing materials error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/feedback/{user_id}")
async def submit_feedback(user_id: str, feedback: FeedbackRequest):
    """학습 피드백 제출"""
    try:
        # 적응형 학습 엔진에서 처리
        await learning_engine.process_feedback(
            user_id=user_id,
            session_id=feedback.session_id,
            feedback=feedback.dict()
        )
        
        # 통계 업데이트
        await db_manager.update_daily_statistics(
            user_id=user_id,
            activities_completed=1,
            practice_minutes=20  # 평균 세션 시간
        )
        
        return {"message": "Feedback processed successfully"}
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ===== 실시간 대화 WebSocket =====
@app.websocket("/ws/conversation/{user_id}")
async def websocket_conversation(websocket: WebSocket, user_id: str):
    """실시간 대화 WebSocket"""
    await websocket.accept()
    
    try:
        # 대화 세션 초기화
        session = await conversation_engine.initialize_session(user_id)
        
        # 데이터베이스 세션 생성
        db_session_id = await db_manager.create_learning_session(
            user_id=user_id,
            session_type="conversation"
        )
        
        await websocket.send_json({
            "type": "session_started",
            "session_id": session['id'],
            "message": "Hello! I'm your AI English teacher. Let's practice English together!",
            "topic": session['topic']['title']
        })
        
        conversation_count = 0
        
        while True:
            # 사용자 입력 수신
            data = await websocket.receive_json()
            
            if data['type'] == 'audio':
                # 음성 입력 처리
                audio_base64 = data['audio']
                audio_data = base64.b64decode(audio_base64)
                user_input = await voice_engine.transcribe(audio_data)
                
                await websocket.send_json({
                    "type": "transcription",
                    "text": user_input
                })
            else:
                # 텍스트 입력
                user_input = data['message']
            
            # IVR 시스템으로 대화형 응답 생성
            voice_settings = {
                'user_id': user_id,
                'emotion': 'neutral',
                'use_bone_conduction': True,
                'speed': 1.0
            }
            
            context = {
                'session_id': session['id'],
                'topic': session['topic']['title'],
                'conversation_count': conversation_count
            }
            
            response = await ivr_system.generate_conversational_response(
                user_input=user_input,
                context=context,
                voice_settings=voice_settings
            )
            
            # 대화 로그 저장
            await db_manager.log_conversation(
                session_id=db_session_id,
                user_id=user_id,
                user_message=user_input,
                ai_response=response['text'],
                errors_detected=response.get('errors', []),
                corrections_made=response.get('corrections', [])
            )
            
            # IVR 시스템에서 이미 음성이 생성됨
            audio_base64 = response.get('audio_base64')
            
            # 응답 전송
            await websocket.send_json({
                "type": "response",
                "text": response['text'],
                "audio_base64": audio_base64,
                "response_type": response.get('response_type', 'full_response'),
                "thinking_audio": response.get('thinking_audio'),
                "backchannel_audio": response.get('backchannel_audio'),
                "errors": response.get('errors', []),
                "corrections": response.get('corrections', []),
                "vocabulary_suggestions": response.get('vocabulary_suggestions', []),
                "pronunciation_tips": response.get('pronunciation_tips', []),
                "progress": response.get('progress_update', {}),
                "suggestions": response.get('suggestions', []),
                "conversation_flow": response.get('conversation_flow', {}),
                "engagement_level": response.get('engagement_level', 'neutral')
            })
            
            conversation_count += 1
            
            # 통계 업데이트 (5턴마다)
            if conversation_count % 5 == 0:
                await db_manager.update_daily_statistics(
                    user_id=user_id,
                    interactions=5,
                    errors_made=len(response.get('errors', [])),
                    errors_corrected=len(response.get('corrections', [])),
                    practice_minutes=2  # 2분 추정
                )
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": "Connection error occurred"
        })
    finally:
        try:
            await websocket.close()
        except:
            pass

# ===== 통계 및 분석 =====
@app.get("/api/statistics/{user_id}")
async def get_user_statistics(user_id: str, days: int = 30):
    """사용자 학습 통계 조회"""
    try:
        stats = await db_manager.get_user_statistics(user_id, days)
        return stats
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/progress/{user_id}")
async def get_learning_progress(user_id: str):
    """학습 진도 조회"""
    try:
        profile = await db_manager.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user_id,
            "current_level": profile['current_level'],
            "target_level": profile['target_level'],
            "progress_by_skill": profile['progress'],
            "total_practice_time": profile['total_practice_time'],
            "streak_days": profile['streak_days']
        }
    except Exception as e:
        logger.error(f"Progress retrieval error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# ===== 건강 체크 =====
@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected" if db_manager.conn else "disconnected",
            "voice_engine": "loaded" if voice_engine.openvoice_model else "not_loaded",
            "conversation_engine": "ready",
            "learning_engine": "ready"
        }
    }

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Voice Shadow - AI English Learning App",
        "version": "1.0.0",
        "endpoints": {
            "register": "/api/register",
            "voice_setup": "/api/voice/setup", 
            "conversation": "/ws/conversation/{user_id}",
            "shadowing": "/api/shadowing/{user_id}/{session_id}",
            "personas": "/api/voice/personas",
            "persona_select": "/api/voice/persona/select",
            "persona_tts": "/api/voice/persona/tts",
            "voice_system_init": "/api/voice/system/initialize",
            "streaming_voice": "/api/voice/stream/speak",
            "multi_dialogue": "/api/voice/dialogue/create",
            "voice_feedback": "/api/voice/feedback/process",
            "system_status": "/api/voice/system/status",
            "accent_report": "/api/voice/accent/report/{user_id}",
            "health": "/health"
        }
    }

# 메인 실행
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )