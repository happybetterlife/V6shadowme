import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import uuid

# Add OpenVoice to Python path
openvoice_path = Path(__file__).parent.parent.parent / "OpenVoice"
if str(openvoice_path) not in sys.path:
    sys.path.append(str(openvoice_path))

try:
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter, BaseSpeakerTTS
except ImportError as e:
    print(f"Warning: OpenVoice not available: {e}")
    se_extractor = None
    ToneColorConverter = None
    BaseSpeakerTTS = None

class OpenVoiceService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_path = Path(__file__).parent.parent.parent / "OpenVoice"

        # Model paths from environment
        self.ckpt_base = os.getenv(
            'OPENVOICE_V1_BASE_SPEAKER',
            str(self.base_path / "checkpoints" / "base_speakers" / "EN" / "checkpoint.pth")
        )
        self.ckpt_converter = os.getenv(
            'OPENVOICE_V1_TONE_COLOR_CONVERTER',
            str(self.base_path / "checkpoints" / "converter" / "checkpoint.pth")
        )
        self.default_speaker = os.getenv(
            'OPENVOICE_V1_BASE_SPEAKER',
            str(self.base_path / "checkpoints" / "base_speakers" / "EN" / "en_default_se.pth")
        )

        # Directories
        self.temp_dir = Path(tempfile.gettempdir()) / "openvoice"
        self.temp_dir.mkdir(exist_ok=True)

        # Initialize models
        self.base_speaker_tts = None
        self.tone_color_converter = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize OpenVoice models"""
        try:
            if not all([ToneColorConverter, BaseSpeakerTTS, se_extractor]):
                print("OpenVoice modules not available")
                return

            # Check if model files exist
            if not Path(self.ckpt_base).exists():
                print(f"Base speaker checkpoint not found: {self.ckpt_base}")
                return

            if not Path(self.ckpt_converter).exists():
                print(f"Converter checkpoint not found: {self.ckpt_converter}")
                return

            # Initialize base speaker TTS
            config_path = Path(self.ckpt_base).parent / "config.json"
            self.base_speaker_tts = BaseSpeakerTTS(
                str(config_path),
                device=self.device
            )
            self.base_speaker_tts.load_ckpt(self.ckpt_base)

            # Initialize tone color converter
            converter_config = Path(self.ckpt_converter).parent / "config.json"
            self.tone_color_converter = ToneColorConverter(
                str(converter_config),
                device=self.device
            )
            self.tone_color_converter.load_ckpt(self.ckpt_converter)

            print("OpenVoice models initialized successfully")

        except Exception as e:
            print(f"Error initializing OpenVoice models: {e}")
            self.base_speaker_tts = None
            self.tone_color_converter = None

    def is_available(self) -> bool:
        """Check if OpenVoice is available and initialized"""
        return (
            self.base_speaker_tts is not None and
            self.tone_color_converter is not None
        )

    def extract_speaker_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """Extract speaker embedding from audio file"""
        try:
            if not se_extractor or not Path(audio_path).exists():
                return None

            # Use OpenVoice speaker encoder
            speaker_embedding = se_extractor.get_se(
                audio_path,
                self.tone_color_converter,
                target_dir=str(self.temp_dir),
                vad=True
            )

            return speaker_embedding

        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")
            return None

    def create_voice_model(
        self,
        audio_files: list,
        model_name: str = None
    ) -> Dict[str, Any]:
        """Create a voice model from audio samples"""
        try:
            if not self.is_available():
                return {
                    "success": False,
                    "error": "OpenVoice not available"
                }

            model_id = model_name or str(uuid.uuid4())

            # Process audio files and extract embeddings
            embeddings = []
            for audio_file in audio_files:
                embedding = self.extract_speaker_embedding(audio_file)
                if embedding is not None:
                    embeddings.append(embedding)

            if not embeddings:
                return {
                    "success": False,
                    "error": "No valid embeddings extracted"
                }

            # Average the embeddings if multiple files
            final_embedding = np.mean(embeddings, axis=0) if len(embeddings) > 1 else embeddings[0]

            # Save embedding
            embedding_path = self.temp_dir / f"{model_id}_embedding.npy"
            np.save(embedding_path, final_embedding)

            return {
                "success": True,
                "model_id": model_id,
                "embedding_path": str(embedding_path),
                "num_samples": len(audio_files),
                "num_valid_embeddings": len(embeddings)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Voice model creation failed: {str(e)}"
            }

    def generate_speech(
        self,
        text: str,
        model_id: str = None,
        speaker_embedding: np.ndarray = None
    ) -> Dict[str, Any]:
        """Generate speech using OpenVoice"""
        try:
            if not self.is_available():
                return {
                    "success": False,
                    "error": "OpenVoice not available"
                }

            # Use provided embedding or load from model_id
            if speaker_embedding is None and model_id:
                embedding_path = self.temp_dir / f"{model_id}_embedding.npy"
                if embedding_path.exists():
                    speaker_embedding = np.load(embedding_path)
                else:
                    # Use default speaker if no custom embedding
                    if Path(self.default_speaker).exists():
                        speaker_embedding = torch.load(self.default_speaker, map_location=self.device)
                    else:
                        return {
                            "success": False,
                            "error": "No speaker embedding available"
                        }

            # Generate base speech
            output_id = str(uuid.uuid4())
            base_audio_path = self.temp_dir / f"{output_id}_base.wav"

            self.base_speaker_tts.tts(
                text,
                str(base_audio_path),
                speaker='default',
                language='English',
                speed=1.0
            )

            # Apply tone color conversion if custom embedding provided
            if speaker_embedding is not None:
                final_audio_path = self.temp_dir / f"{output_id}_final.wav"

                # Convert tone color
                encode_message = "@MyShell"
                self.tone_color_converter.convert(
                    audio_src_path=str(base_audio_path),
                    src_se=speaker_embedding,
                    tgt_se=speaker_embedding,
                    output_path=str(final_audio_path),
                    message=encode_message
                )

                return {
                    "success": True,
                    "audio_path": str(final_audio_path),
                    "text": text,
                    "model_id": model_id
                }
            else:
                return {
                    "success": True,
                    "audio_path": str(base_audio_path),
                    "text": text,
                    "model_id": "default"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Speech generation failed: {str(e)}"
            }

    def cleanup_temp_files(self, older_than_hours: int = 24):
        """Clean up temporary files older than specified hours"""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)

            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()

        except Exception as e:
            print(f"Error cleaning up temp files: {e}")

# Singleton instance
_openvoice_service = None

def get_openvoice_service() -> OpenVoiceService:
    """Get singleton OpenVoice service instance"""
    global _openvoice_service
    if _openvoice_service is None:
        _openvoice_service = OpenVoiceService()
    return _openvoice_service