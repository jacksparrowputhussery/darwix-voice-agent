import os
import logging

logger = logging.getLogger(__name__)

# Shared singleton model across requests to prevent multi-second reload overhead
_SHARED_WHISPER_MODEL = None

class STTEngine:
    def __init__(self):
        self.provider = os.getenv("ASR_PROVIDER", "faster-whisper")
        self.model_name = os.getenv("ASR_MODEL", "tiny")  # tiny/base for ultra-fast CPU inference

    def _load_model(self):
        global _SHARED_WHISPER_MODEL
        if _SHARED_WHISPER_MODEL is None and self.provider == "faster-whisper":
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading faster-whisper model '{self.model_name}' on CPU (int8)...")
                _SHARED_WHISPER_MODEL = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4
                )
                logger.info("faster-whisper model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load faster-whisper: {e}. Using mock STT.")
                self.provider = "mock"
        return _SHARED_WHISPER_MODEL

    def warmup(self):
        """Pre-warm the STT model on server boot."""
        if self.provider == "faster-whisper":
            self._load_model()

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes the given audio file and returns the text.
        """
        if self.provider == "faster-whisper":
            model = self._load_model()
            if model:
                try:
                    segments, info = model.transcribe(
                        audio_path,
                        beam_size=1,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )
                    text = " ".join([segment.text for segment in segments])
                    return text.strip()
                except Exception as e:
                    logger.warning(f"STT Error: {e}")
                    return "[Unintelligible]"
                    
        # Fallback Mock
        logger.info(f"Mock STT: Transcribing {audio_path}")
        return "This is a mock transcription of the audio."
