# ShadowME - Voice Shadowing Practice App

ShadowME는 AI 기반 음성 섀도잉 연습 앱입니다. Flutter 프론트엔드와 Python FastAPI 백엔드로 구성되어 있습니다.

## 🎯 주요 기능

- **음성 섀도잉 연습**: 30초 음성 녹음을 통한 개인화된 음성 클론
- **AI 코치**: 개인화된 피드백과 연습 가이드
- **진행 상황 추적**: 상세한 분석과 통계
- **반응형 디자인**: 모바일과 태블릿 지원
- **Firebase 통합**: 인증, Firestore, Storage

## 🏗️ 아키텍처

### 프론트엔드 (Flutter)
- **Flutter 3.x** 최신 버전
- **Riverpod** 상태 관리
- **Firebase** 인증 및 데이터베이스
- **반응형 디자인** (모바일/태블릿)

### 백엔드 (Python FastAPI)
- **FastAPI** 웹 프레임워크
- **Firebase Admin SDK** 통합
- **ML 모델** 음성 분석 및 클론
- **RESTful API** 설계

## 📱 화면 구성

1. **Launch Screen** - 앱 시작 화면
2. **Onboarding** - 3단계 온보딩
3. **Voice Setup** - 30초 음성 녹음
4. **Main Dashboard** - 메인 대시보드
5. **Shadowing Practice** - 섀도잉 연습
6. **Progress Analytics** - 진행 상황 분석

## 🚀 시작하기

### 전제 조건
- Flutter 3.x
- Python 3.8+
- Firebase 프로젝트
- Git

### 설치 및 실행

#### 1. 저장소 클론
```bash
git clone <repository-url>
cd V6shadowme
```

#### 2. Flutter 프론트엔드 설정
```bash
cd voiceshadow/frontend
flutter pub get
flutter run
```

#### 3. Python 백엔드 설정
```bash
cd voiceshadow/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### Firebase 설정

1. Firebase 콘솔에서 새 프로젝트 생성
2. `google-services.json` (Android) 및 `GoogleService-Info.plist` (iOS) 파일 추가
3. Firestore 및 Storage 규칙 설정

## 🎨 디자인 시스템

Figma V5-ShadowME 디자인 시스템을 기반으로 구현:
- **색상**: 보라색 그라데이션 테마 (#3A1AB0)
- **타이포그래피**: Inter 및 Poppins 폰트
- **컴포넌트**: 재사용 가능한 UI 컴포넌트

## 📦 주요 의존성

### Flutter
- `flutter_riverpod` - 상태 관리
- `firebase_core` - Firebase 통합
- `google_fonts` - 폰트
- `flutter_svg` - SVG 지원
- `lottie` - 애니메이션

### Python
- `fastapi` - 웹 프레임워크
- `firebase-admin` - Firebase 관리
- `uvicorn` - ASGI 서버
- `pydantic` - 데이터 검증

## 🔧 개발

### 코드 구조
```
voiceshadow/
├── frontend/          # Flutter 앱
│   ├── lib/
│   │   ├── core/      # 테마, 유틸리티
│   │   ├── features/  # 기능별 모듈
│   │   └── shared/    # 공통 컴포넌트
├── backend/           # Python API
│   ├── app/
│   │   ├── api/       # API 엔드포인트
│   │   ├── core/      # 설정, 보안
│   │   ├── models/    # 데이터 모델
│   │   └── services/  # 비즈니스 로직
└── shared/            # 공통 설정
```

## 📄 라이선스

MIT License

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 연락처

프로젝트 링크: [https://github.com/your-username/V6shadowme](https://github.com/your-username/V6shadowme)
