#!/bin/bash

echo "🚀 Running ShadowME App..."
echo "============================================"

# 프로젝트 디렉토리로 이동
cd "/Users/joy/Desktop/figma_local/v6 shadowme/voiceshadow/frontend"

# 의존성 확인 및 설치
echo "📦 Installing dependencies..."
flutter pub get

echo ""
echo "🎯 App Features:"
echo "  • 390x844 고정 레이아웃"
echo "  • 통일된 보라색 배경"
echo "  • 완전한 앱 플로우 (Launch → Sign Up → Welcome → Voice Setup → Dashboard)"
echo "  • SafeArea 레이아웃 래퍼"
echo "  • 실제 앱 네비게이션"
echo ""

echo "🔥 Starting app..."
echo "============================================"

# Flutter 앱 실행 (웹으로)
flutter run -d web-server --web-port 8080

echo ""
echo "✅ App started!"
echo "🌐 Open: http://localhost:8080"
echo ""
echo "💡 Navigation:"
echo "  - Launch Screen → 앱 시작"
echo "  - Sign Up → 회원가입"
echo "  - Welcome → 환영 화면"
echo "  - Voice Setup → 음성 설정"
echo "  - Dashboard → 메인 대시보드"