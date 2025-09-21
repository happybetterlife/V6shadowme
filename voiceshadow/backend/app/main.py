
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.security import get_current_user
from app.api.endpoints import auth, voice_cloning, speech_analysis, user_progress, personas
# Import scheduler separately to handle optional dependencies
try:
    from app.api.endpoints import scheduler
    SCHEDULER_AVAILABLE = True
except ImportError as e:
    print(f"Scheduler endpoint not available: {e}")
    scheduler = None
    SCHEDULER_AVAILABLE = False
from app.services.firebase.firebase_service import FirebaseService
from app.services.openvoice_service import get_openvoice_service

# Import scheduler service (optional)
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from services.scheduler_service import scheduler_service
    SCHEDULER_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"Scheduler service not available: {e}")
    scheduler_service = None
    SCHEDULER_SERVICE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting VoiceShadow API...")
    
    # Initialize Firebase
    try:
        FirebaseService.initialize()
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        raise

    # Initialize OpenVoice
    try:
        openvoice_service = get_openvoice_service()
        if openvoice_service.is_available():
            logger.info("OpenVoice initialized successfully")
        else:
            logger.warning("OpenVoice not available - voice cloning features may be limited")
    except Exception as e:
        logger.error(f"Failed to initialize OpenVoice: {e}")
        logger.warning("Continuing without OpenVoice - voice cloning features may be limited")

    # Start scheduler service for trend data collection
    if SCHEDULER_SERVICE_AVAILABLE and scheduler_service:
        try:
            await scheduler_service.start()
            logger.info("Scheduler service started successfully")
            logger.info("Trend data collection will run every 2 hours")
        except Exception as e:
            logger.error(f"Failed to start scheduler service: {e}")
            logger.warning("Continuing without automatic trend data collection")
    else:
        logger.info("Scheduler service not available - trend collection disabled")

    yield

    # Shutdown
    logger.info("Shutting down VoiceShadow API...")

    # Stop scheduler service
    if SCHEDULER_SERVICE_AVAILABLE and scheduler_service:
        try:
            await scheduler_service.stop()
            logger.info("Scheduler service stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler service: {e}")

# Create FastAPI app
app = FastAPI(
    title="VoiceShadow API",
    description="AI-powered voice cloning and speech shadowing platform",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include API routers
app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Authentication"]
)

app.include_router(
    voice_cloning.router,
    prefix="/api/v1/voice-cloning",
    tags=["Voice Cloning"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    speech_analysis.router,
    prefix="/api/v1/speech-analysis",
    tags=["Speech Analysis"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    user_progress.router,
    prefix="/api/v1/user-progress",
    tags=["User Progress"],
    dependencies=[Depends(get_current_user)]
)

app.include_router(
    personas.router,
    prefix="/api/v1/personas",
    tags=["Voice Personas"],
    dependencies=[Depends(get_current_user)]
)

# Include scheduler router only if available
if SCHEDULER_AVAILABLE and scheduler:
    app.include_router(
        scheduler.router,
        prefix="/api/v1/scheduler",
        tags=["Scheduler Management"],
        dependencies=[Depends(get_current_user)]
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to VoiceShadow API",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
