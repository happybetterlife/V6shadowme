"""OpenVoice integration service for voice cloning and speech synthesis."""

import os
import torch
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

logger = logging.getLogger(__name__)


class OpenVoiceService:
    """Service for OpenVoice voice cloning and generation."""

    def __init__(self, model_dir: str = "models/openvoice"):
        """Initialize OpenVoice service with model paths."""
        self.model_dir = Path(model_dir)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Model paths
        self.ckpt_base = self.model_dir / "checkpoints/base_speakers/EN"
        self.ckpt_converter = self.model_dir / "checkpoints/converter"

        # Initialize models
        self.base_speaker_tts = None
        self.tone_color_converter = None
        self.source_se = None

        self._initialize_models()

    def _initialize_models(self):
        """Initialize OpenVoice models."""
        try:
            # Load base speaker TTS
            self.base_speaker_tts = BaseSpeakerTTS(
                f'{self.ckpt_base}/config.json',
                device=self.device
            )
            self.base_speaker_tts.load_ckpt(f'{self.ckpt_base}/checkpoint.pth')

            # Load tone color converter
            self.tone_color_converter = ToneColorConverter(
                f'{self.ckpt_converter}/config.json',
                device=self.device
            )
            self.tone_color_converter.load_ckpt(f'{self.ckpt_converter}/checkpoint.pth')

            # Load default source speaker embedding
            self.source_se = torch.load(
                f'{self.ckpt_base}/en_default_se.pth'
            ).to(self.device)

            logger.info("OpenVoice models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OpenVoice models: {e}")
            raise

    async def extract_voice_embedding(
        self,
        audio_path: str,
        output_dir: str = "processed",
        use_vad: bool = True
    ) -> torch.Tensor:
        """Extract voice embedding from reference audio.

        Args:
            audio_path: Path to reference audio file
            output_dir: Directory to save processed audio
            use_vad: Whether to use voice activity detection

        Returns:
            Voice embedding tensor
        """
        loop = asyncio.get_event_loop()

        def _extract():
            target_se, audio_name = se_extractor.get_se(
                audio_path,
                self.tone_color_converter,
                target_dir=output_dir,
                vad=use_vad
            )
            return target_se

        target_se = await loop.run_in_executor(None, _extract)
        return target_se

    async def clone_voice(
        self,
        text: str,
        reference_audio: str,
        output_path: str,
        speaker: str = "default",
        language: str = "English",
        speed: float = 1.0,
        watermark: Optional[str] = "@VoiceShadow"
    ) -> Dict[str, Any]:
        """Generate speech with cloned voice.

        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio for voice cloning
            output_path: Path to save output audio
            speaker: Speaking style (default, friendly, cheerful, excited, sad, angry, terrified, shouting, whispering)
            language: Language for synthesis
            speed: Speech speed (0.5-2.0)
            watermark: Optional watermark message

        Returns:
            Generation result dictionary
        """
        try:
            # Extract target voice embedding
            target_se = await self.extract_voice_embedding(reference_audio)

            # Run generation in executor
            loop = asyncio.get_event_loop()

            def _generate():
                # Create output directory
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

                # Generate base TTS
                temp_path = Path(output_path).with_suffix('.tmp.wav')
                self.base_speaker_tts.tts(
                    text,
                    str(temp_path),
                    speaker=speaker,
                    language=language,
                    speed=speed
                )

                # Apply voice cloning
                self.tone_color_converter.convert(
                    audio_src_path=str(temp_path),
                    src_se=self.source_se,
                    tgt_se=target_se,
                    output_path=output_path,
                    message=watermark
                )

                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()

                return True

            await loop.run_in_executor(None, _generate)

            return {
                "status": "completed",
                "output_file": output_path,
                "text": text,
                "speaker": speaker,
                "language": language,
                "speed": speed
            }

        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def generate_multi_style(
        self,
        text: str,
        reference_audio: str,
        output_dir: str,
        styles: list = None,
        language: str = "English",
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """Generate speech in multiple styles.

        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio
            output_dir: Directory to save outputs
            styles: List of speaking styles to generate
            language: Language for synthesis
            speed: Speech speed

        Returns:
            Dictionary with paths to generated audio files
        """
        if styles is None:
            styles = ["default", "friendly", "cheerful", "excited"]

        # Extract voice embedding once
        target_se = await self.extract_voice_embedding(reference_audio)

        results = {}
        for style in styles:
            output_path = os.path.join(output_dir, f"{style}_{Path(text[:20]).stem}.wav")
            result = await self.clone_voice(
                text=text,
                reference_audio=reference_audio,
                output_path=output_path,
                speaker=style,
                language=language,
                speed=speed
            )
            results[style] = result

        return results

    def get_available_styles(self) -> list:
        """Get list of available speaking styles."""
        return [
            "default",
            "friendly",
            "cheerful",
            "excited",
            "sad",
            "angry",
            "terrified",
            "shouting",
            "whispering"
        ]

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return [
            "English",
            "Chinese",
            "Spanish",
            "French",
            "German",
            "Japanese",
            "Korean"
        ]


# Quality metrics placeholder
class VoiceQualityMetrics:
    """Placeholder for voice quality metrics - to be implemented with Firebase/DB."""

    @staticmethod
    async def calculate_similarity(original_audio: str, generated_audio: str) -> float:
        """Calculate similarity between original and generated audio.

        This is a placeholder that returns mock data.
        Real implementation would use speech embeddings comparison.
        """
        return 0.85

    @staticmethod
    async def calculate_clarity(audio_path: str) -> float:
        """Calculate audio clarity score.

        Placeholder returning mock data.
        Real implementation would analyze SNR, spectral clarity, etc.
        """
        return 0.90

    @staticmethod
    async def calculate_naturalness(audio_path: str) -> float:
        """Calculate naturalness score.

        Placeholder returning mock data.
        Real implementation would use MOS prediction models.
        """
        return 0.88


# Persistence placeholder
class VoiceModelPersistence:
    """Placeholder for voice model persistence - to be implemented with Firebase/DB."""

    @staticmethod
    async def save_model_metadata(
        user_id: str,
        model_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Save voice model metadata to database.

        Placeholder that logs the data.
        Real implementation would save to Firebase/PostgreSQL.
        """
        logger.info(f"Saving model metadata for user {user_id}, model {model_id}")
        logger.debug(f"Metadata: {metadata}")
        return True

    @staticmethod
    async def load_model_metadata(
        user_id: str,
        model_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load voice model metadata from database.

        Placeholder returning mock data.
        Real implementation would fetch from Firebase/PostgreSQL.
        """
        return {
            "user_id": user_id,
            "model_id": model_id,
            "created_at": "2024-01-01T00:00:00Z",
            "status": "active",
            "quality_scores": {
                "clarity": 0.9,
                "naturalness": 0.88,
                "similarity": 0.85
            }
        }