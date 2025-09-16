#!/usr/bin/env python3
"""
연결 테스트 스크립트 - 모든 시스템 연결 확인
"""

import sys
import traceback

def test_imports():
    """모든 임포트 테스트"""
    print("🔍 Testing imports...")
    
    try:
        # 기본 모듈
        from config.settings import Settings
        print("✅ Settings import successful")
        
        # Core 모듈들
        from core.conversation_engine import ConversationEngine
        print("✅ ConversationEngine import successful")
        
        from core.voice_engine import VoiceEngine
        print("✅ VoiceEngine import successful")
        
        from core.learning_engine import LearningEngine
        print("✅ LearningEngine import successful")
        
        from core.interactive_voice_response import InteractiveVoiceResponse
        print("✅ InteractiveVoiceResponse import successful")
        
        from core.chatbot_voice_persona import ChatbotVoicePersonaSystem
        print("✅ ChatbotVoicePersonaSystem import successful")
        
        from core.complete_voice_system import CompleteChatbotVoiceSystem
        print("✅ CompleteChatbotVoiceSystem import successful")
        
        # Agent 모듈들
        from agents.orchestrator import AgentOrchestrator
        print("✅ AgentOrchestrator import successful")
        
        # Database 모듈
        from database.manager import DatabaseManager
        print("✅ DatabaseManager import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_class_initialization():
    """클래스 초기화 테스트"""
    print("\n🔧 Testing class initialization...")
    
    try:
        # Settings
        from config.settings import Settings
        settings = Settings()
        print("✅ Settings initialized")
        
        # Voice Engine
        from core.voice_engine import VoiceEngine
        voice_engine = VoiceEngine()
        print("✅ VoiceEngine initialized")
        
        # Voice Persona System
        from core.chatbot_voice_persona import ChatbotVoicePersonaSystem
        persona_system = ChatbotVoicePersonaSystem()
        print("✅ ChatbotVoicePersonaSystem initialized")
        
        # Complete Voice System
        from core.complete_voice_system import CompleteChatbotVoiceSystem
        complete_system = CompleteChatbotVoiceSystem()
        print("✅ CompleteChatbotVoiceSystem initialized")
        
        # Database Manager
        from database.manager import DatabaseManager
        db_manager = DatabaseManager()
        print("✅ DatabaseManager initialized")
        
        return True
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        traceback.print_exc()
        return False

def test_method_availability():
    """메서드 가용성 테스트"""
    print("\n🎯 Testing method availability...")
    
    try:
        from core.voice_engine import VoiceEngine
        voice_engine = VoiceEngine()
        
        # 핵심 메서드들 확인
        methods_to_check = [
            'setup_voice_cloning',
            'generate_response_audio',
            'generate_shadowing_audio',
            'transcribe',
            'generate_persona_audio',
            'get_available_personas',
            'select_voice_persona',
            'initialize_complete_voice_system',
            'speak_response_streaming',
            'create_multi_speaker_dialogue'
        ]
        
        for method_name in methods_to_check:
            if hasattr(voice_engine, method_name):
                print(f"✅ {method_name} method available")
            else:
                print(f"❌ {method_name} method missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Method check error: {e}")
        traceback.print_exc()
        return False

def test_api_endpoints_structure():
    """API 엔드포인트 구조 테스트"""
    print("\n🌐 Testing API endpoints structure...")
    
    try:
        # main.py에서 엔드포인트 정의 확인
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_endpoints = [
            '@app.get("/health")',
            '@app.post("/api/register")',
            '@app.post("/api/voice/setup")',
            '@app.get("/api/voice/personas")',
            '@app.post("/api/voice/persona/select")',
            '@app.post("/api/voice/persona/tts")',
            '@app.post("/api/voice/system/initialize")',
            '@app.post("/api/voice/stream/speak")',
            '@app.websocket("/ws/conversation/{user_id}")',
        ]
        
        for endpoint in required_endpoints:
            if endpoint in content:
                print(f"✅ {endpoint} endpoint defined")
            else:
                print(f"❌ {endpoint} endpoint missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ API endpoint check error: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Voice Shadow - Connection Test")
    print("=" * 50)
    
    all_passed = True
    
    # 임포트 테스트
    if not test_imports():
        all_passed = False
    
    # 초기화 테스트  
    if not test_class_initialization():
        all_passed = False
    
    # 메서드 가용성 테스트
    if not test_method_availability():
        all_passed = False
    
    # API 엔드포인트 구조 테스트
    if not test_api_endpoints_structure():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All connection tests PASSED!")
        print("✅ System is ready for deployment")
    else:
        print("❌ Some tests FAILED!")
        print("🔧 Please check the errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()