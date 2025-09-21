"""Voice engine - OpenVoice and bone conduction simulation stubs."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

OpenVoiceV2 = None
whisper = None


class VoiceEngine:
    """High-level voice engine combining cloning and ASR.

    Heavy ML dependencies are optional; when missing, the engine falls back to
    lightweight stubs so the application can run in development environments
    without GPU resources.
    """

    def __init__(self) -> None:
        self.sample_rate = 22050
        self.openvoice_model = None
        self.whisper_model = None
        self.user_voices: Dict[str, Dict[str, object]] = {}
        self.bone_conduction_params = self.init_bone_conduction_params()
        self._numpy = None

    def _np(self):
        if self._numpy is None:
            try:
                import numpy as _np  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError("numpy is required for voice processing") from exc
            self._numpy = _np
        return self._numpy

    async def load_models(self) -> None:
        """Load OpenVoice and Whisper models when available."""
        if OpenVoiceV2 is not None:
            self.openvoice_model = self.load_openvoice_model()
        else:
            logger.warning("OpenVoiceV2 not installed; using stub responses")

        global whisper
        if whisper is None:
            try:
                import whisper as _whisper  # type: ignore
            except ModuleNotFoundError:  # pragma: no cover
                logger.warning("whisper not installed; transcription disabled")
                _whisper = None
            whisper = _whisper

        if whisper is not None:
            self.whisper_model = whisper.load_model("base")

        logger.info("Voice models loaded successfully")

    def load_openvoice_model(self):
        global OpenVoiceV2
        if OpenVoiceV2 is None:
            try:
                from openvoice import OpenVoiceV2 as _OpenVoiceV2  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError("OpenVoiceV2 dependency not available") from exc
            OpenVoiceV2 = _OpenVoiceV2

        model = OpenVoiceV2(
            config_path="configs/openvoice_v2.json",
            checkpoint_path="checkpoints/openvoice_v2.pth",
        )
        return model

    async def setup_voice_cloning(self, user_id: str, audio_data: bytes) -> Dict[str, object]:
        audio_array = self.bytes_to_array(audio_data)
        sample = self.extract_30_second_sample(audio_array)
        tone_color = await self.extract_tone_color(sample)
        speaker_embedding = await self.create_speaker_embedding(sample)

        voice_profile = {
            "id": f"voice_{user_id}",
            "user_id": user_id,
            "tone_color": tone_color,
            "speaker_embedding": speaker_embedding,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.user_voices[user_id] = voice_profile
        await self.save_voice_profile(voice_profile)
        return voice_profile

    async def generate_response_audio(
        self,
        text: str,
        user_id: str,
        emotion: str = "neutral",
        use_bone_conduction: bool = True,
    ) -> bytes:
        base_audio = await self.generate_chatbot_voice(text, emotion)
        processed = (
            self.apply_bone_conduction(base_audio, user_id)
            if use_bone_conduction
            else base_audio
        )
        return self.array_to_bytes(processed)

    async def generate_shadowing_audio(
        self,
        text: str,
        user_id: str,
        speed: float = 1.0,
    ) -> bytes:
        profile = self.user_voices.get(user_id)
        if not profile:
            raise ValueError(f"No voice profile for user {user_id}")

        if self.openvoice_model is not None:
            base_audio = await self.openvoice_model.tts(
                text=text,
                speaker_embedding=profile["speaker_embedding"],
                speed=speed,
            )
        else:
            logger.warning("OpenVoice model unavailable; returning synthetic sine wave")
            base_audio = self.synthetic_audio_from_text(text, speed)

        user_voice = await self.apply_tone_color(base_audio, profile["tone_color"])
        natural_audio = self.apply_bone_conduction(user_voice, user_id)
        return self.array_to_bytes(natural_audio)

    def apply_bone_conduction(self, audio, user_id: str):
        np = self._np()
        params = self.get_user_bone_conduction_params(user_id)
        low_gain = self.db_to_amplitude(params["bass_gain"])
        filtered = audio * low_gain * params["mix_ratio"] + audio * (1 - params["mix_ratio"])
        return self.normalize_audio(filtered)

    async def transcribe(self, audio_data: bytes) -> str:
        audio_array = self.bytes_to_array(audio_data)
        if self.whisper_model is None:
            logger.warning("Whisper model unavailable; returning empty transcription")
            return ""
        result = self.whisper_model.transcribe(audio_array, language="en")
        return result["text"]

    # ------------------------------------------------------------------
    # Utility helpers --------------------------------------------------
    # ------------------------------------------------------------------

    def init_bone_conduction_params(self) -> Dict[str, float]:
        return {
            "bass_gain": 6.0,
            "warmth": 3.0,
            "high_cut_freq": 4000.0,
            "mix_ratio": 0.3,
        }

    def get_user_bone_conduction_params(self, user_id: str) -> Dict[str, float]:
        return self.bone_conduction_params

    def db_to_amplitude(self, db: float) -> float:
        return 10 ** (db / 20.0)

    def normalize_audio(self, audio):
        np = self._np()
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.95
        return audio

    def bytes_to_array(self, audio_bytes: bytes):
        np = self._np()
        return np.frombuffer(audio_bytes, dtype=np.float32)

    def array_to_bytes(self, audio_array) -> bytes:
        np = self._np()
        return audio_array.astype(np.float32).tobytes()

    def extract_30_second_sample(self, audio_array):
        np = self._np()
        target_samples = self.sample_rate * 30
        if audio_array.shape[0] >= target_samples:
            return audio_array[:target_samples]
        padding = np.zeros(target_samples - audio_array.shape[0], dtype=np.float32)
        return np.concatenate([audio_array, padding])

    async def extract_tone_color(self, sample) -> Dict[str, float]:
        np = self._np()
        await asyncio.sleep(0)
        return {"brightness": float(np.mean(np.abs(sample))), "richness": float(np.std(sample))}

    async def create_speaker_embedding(self, sample) -> Dict[str, float]:
        np = self._np()
        await asyncio.sleep(0)
        return {"embedding": float(np.mean(sample))}

    async def save_voice_profile(self, profile: Dict[str, object]) -> None:
        logger.debug("Saving voice profile %s", profile["id"])
        await asyncio.sleep(0)

    async def generate_chatbot_voice(self, text: str, emotion: str):
        await asyncio.sleep(0)
        return self.synthetic_audio_from_text(text, 1.0)

    async def apply_tone_color(self, audio, tone_color: Dict[str, float]):
        await asyncio.sleep(0)
        return audio * (1 + tone_color.get("richness", 0.0))

    def synthetic_audio_from_text(self, text: str, speed: float):
        np = self._np()
        duration = max(1.0, len(text) / 50.0) / speed
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        return 0.1 * np.sin(2 * np.pi * 220 * t)


__all__ = ["VoiceEngine"]
