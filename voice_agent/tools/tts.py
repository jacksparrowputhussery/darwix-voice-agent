import os
import time
import asyncio
import tempfile
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# In-memory LRU-like audio cache: hash(text, voice) -> audio_bytes
_AUDIO_CACHE: Dict[str, bytes] = {}

# Kokoro Singleton instance & lock to prevent repeated model reloads
_KOKORO_INSTANCE = None
_KOKORO_DEVICE = "uninitialized"
_KOKORO_LOCK = asyncio.Lock()


def get_kokoro_model_paths() -> Tuple[Path, Path]:
    """Resolves local paths for Kokoro ONNX model and voices binary."""
    base_dir = Path(__file__).parent.parent.parent / "models" / "kokoro"
    base_dir.mkdir(parents=True, exist_ok=True)
    model_path = base_dir / "kokoro-v1.0.onnx"
    voices_path = base_dir / "voices-v1.0.bin"
    return model_path, voices_path


def ensure_kokoro_assets():
    """Downloads Kokoro ONNX weights and voices if they are missing."""
    model_path, voices_path = get_kokoro_model_paths()
    
    urls = {
        model_path: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        voices_path: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    }
    
    for file_path, url in urls.items():
        if not file_path.exists() or file_path.stat().st_size < 10000:
            logger.info(f"Downloading Kokoro asset {file_path.name} from {url}...")
            try:
                import httpx
                with httpx.Client(follow_redirects=True, timeout=120.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
                    with client.stream("GET", url) as response:
                        response.raise_for_status()
                        with open(file_path, "wb") as f:
                            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                                f.write(chunk)
                logger.info(f"✓ Downloaded {file_path.name} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
            except Exception as e:
                logger.error(f"Failed to auto-download {file_path.name}: {e}")


def load_kokoro_engine():
    """
    Initializes Kokoro ONNX with dynamic Hardware Detection:
    1. If GPU (CUDA / ROCm / CoreML / DirectML) is available, runs on GPU.
    2. If GPU fails or is not available, automatically falls back to CPU.
    """
    global _KOKORO_INSTANCE, _KOKORO_DEVICE
    
    if _KOKORO_INSTANCE is not None:
        return _KOKORO_INSTANCE, _KOKORO_DEVICE

    ensure_kokoro_assets()
    model_path, voices_path = get_kokoro_model_paths()
    
    if not model_path.exists() or not voices_path.exists():
        raise FileNotFoundError(f"Kokoro model files not found at {model_path}")

    try:
        import onnxruntime as rt
        from kokoro_onnx import Kokoro
    except ImportError as e:
        logger.error(f"Kokoro dependencies missing: {e}. Please run `pip install kokoro-onnx soundfile`")
        raise

    available_providers = rt.get_available_providers()
    logger.info(f"Available ONNX Runtime execution providers: {available_providers}")

    preferred_device = os.getenv("KOKORO_DEVICE", "auto").lower()
    gpu_providers = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "TensorrtExecutionProvider",
        "CoreMLExecutionProvider",
        "DmlExecutionProvider"
    ]
    target_gpu = [p for p in gpu_providers if p in available_providers]

    kokoro = None
    active_device = "cpu"

    # 1. Attempt GPU if available and not explicitly disabled
    if target_gpu and preferred_device != "cpu":
        try:
            gpu_provider = target_gpu[0]
            logger.info(f"⚡ Attempting Kokoro initialization on GPU ({gpu_provider})...")
            os.environ["ONNX_PROVIDER"] = gpu_provider
            kokoro = Kokoro(str(model_path), str(voices_path))
            # Test a small inference to verify CUDA/cuDNN doesn't throw a driver error
            kokoro.create("warmup", voice="af_bella", speed=1.0)
            active_device = f"gpu ({gpu_provider})"
            logger.info(f"✓ Kokoro TTS successfully running on GPU ({gpu_provider})")
        except Exception as e:
            logger.warning(f"GPU initialization/inference failed ({e}). Falling back to CPU...")
            kokoro = None

    # 2. Fallback to CPU
    if kokoro is None:
        try:
            logger.info("Initializing Kokoro on CPU (CPUExecutionProvider)...")
            os.environ["ONNX_PROVIDER"] = "CPUExecutionProvider"
            kokoro = Kokoro(str(model_path), str(voices_path))
            active_device = "cpu (CPUExecutionProvider)"
            logger.info("✓ Kokoro TTS successfully running on CPU")
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro on CPU: {e}")
            raise

    _KOKORO_INSTANCE = kokoro
    _KOKORO_DEVICE = active_device
    return _KOKORO_INSTANCE, _KOKORO_DEVICE


class TTSEngine:
    """
    High-Performance Text-to-Speech Engine:
    - Primary: Kokoro-82M (Local Neural TTS with auto GPU/CPU fallback & sub-100ms inference)
    - Secondary: Microsoft Edge-TTS (Multi-lingual cloud neural voices)
    - Fallback: Mock / Instant WebSpeech
    """

    def __init__(self):
        self.provider = os.getenv("TTS_PROVIDER", "kokoro").lower()
        self.model = os.getenv("TTS_MODEL", "af_bella")
        self.timeout_sec = float(os.getenv("TTS_TIMEOUT", "4.0"))
        self.kokoro = None
        self.kokoro_device = "uninitialized"

    def warmup(self):
        """Pre-warms Kokoro or EdgeTTS in the background on startup."""
        if self.provider == "kokoro":
            try:
                self.kokoro, self.kokoro_device = load_kokoro_engine()
                # Run quick warmup synthesis
                self.kokoro.create("Hello", voice=self._resolve_kokoro_voice(self.model))
                logger.info(f"✓ Kokoro TTS pre-warmed on {self.kokoro_device}")
            except Exception as e:
                logger.warning(f"Kokoro warmup notice: {e}")

    def _resolve_kokoro_voice(self, requested_voice: str) -> str:
        """Maps EdgeTTS voice names or custom names to Kokoro voice keys."""
        if not requested_voice:
            return "af_bella"

        voice_lower = requested_voice.lower()
        
        # Exact Kokoro voice names
        known_voices = [
            "af_bella", "af_sarah", "af_heart", "af_nicole", "af_nova", "af_sky",
            "af_alloy", "af_aoede", "af_jessica", "af_kore", "af_river",
            "am_adam", "am_echo", "am_eric", "am_liam", "am_michael", "am_onyx", "am_puck",
            "bf_emma", "bf_alice", "bf_isabella", "bf_lily",
            "bm_george", "bm_daniel", "bm_fable", "bm_lewis"
        ]
        if requested_voice in known_voices:
            return requested_voice

        # Edge-TTS voice mappings
        if "male" in voice_lower or "angelo" in voice_lower or "ardi" in voice_lower or "guy" in voice_lower:
            return "am_adam"
        if "british" in voice_lower or "uk" in voice_lower or "george" in voice_lower:
            return "bf_emma"
        
        # Default high-quality conversational female voice
        return "af_bella"

    async def _generate_kokoro(self, text: str, output_path: str, voice: str) -> Optional[str]:
        """Generates audio using local Kokoro-82M ONNX."""
        try:
            import soundfile as sf
            
            if self.kokoro is None:
                self.kokoro, self.kokoro_device = load_kokoro_engine()

            resolved_voice = self._resolve_kokoro_voice(voice)
            loop = asyncio.get_running_loop()
            
            start_t = time.time()
            # Run Kokoro synthesis in thread pool so it does not block the async event loop
            samples, sample_rate = await loop.run_in_executor(
                None,
                lambda: self.kokoro.create(text, voice=resolved_voice, speed=1.0, lang="en-us")
            )
            
            # Write WAV / MP3 output
            await loop.run_in_executor(
                None,
                lambda: sf.write(output_path, samples, sample_rate)
            )
            
            elapsed = (time.time() - start_t) * 1000
            logger.info(f"⚡ Kokoro TTS generated ({self.kokoro_device}, voice={resolved_voice}) in {elapsed:.1f}ms")
            return output_path
        except Exception as e:
            logger.warning(f"Kokoro synthesis failed ({e}). Falling back to EdgeTTS...")
            return None

    async def _generate_edge_tts(self, text: str, output_path: str, voice: str) -> Optional[str]:
        """Generates audio using Microsoft Edge-TTS cloud service."""
        try:
            import edge_tts
            edge_voice = voice if ("-" in voice and "Neural" in voice) else "en-US-AriaNeural"
            communicate = edge_tts.Communicate(text, edge_voice)
            await asyncio.wait_for(communicate.save(output_path), timeout=self.timeout_sec)
            return output_path
        except asyncio.TimeoutError:
            logger.warning(f"EdgeTTS exceeded {self.timeout_sec}s timeout.")
            return None
        except Exception as e:
            logger.warning(f"EdgeTTS generation error: {e}")
            return None

    async def generate_audio(self, text: str, output_path: str = None) -> Optional[str]:
        """
        Generates TTS audio and saves it to output_path.
        Uses in-memory caching, Kokoro local synthesis, and EdgeTTS fallback.
        """
        if not text or not text.strip():
            return None

        cache_key = hashlib.md5(f"{self.provider}:{self.model}:{text.strip()}".encode("utf-8")).hexdigest()
        
        if not output_path:
            # Kokoro produces uncompressed WAV or MP3
            suffix = ".wav" if self.provider == "kokoro" else ".mp3"
            fd, output_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)

        # 1. Check in-memory audio cache
        if cache_key in _AUDIO_CACHE:
            with open(output_path, "wb") as f:
                f.write(_AUDIO_CACHE[cache_key])
            return output_path

        # 2. Mock TTS provider
        if self.provider == "mock":
            with open(output_path, "wb") as f:
                f.write(b"mock_audio_data")
            return output_path

        # 3. Kokoro TTS (Primary)
        if self.provider in ["kokoro", "local"]:
            res = await self._generate_kokoro(text, output_path, self.model)
            if res and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                with open(output_path, "rb") as f:
                    if len(_AUDIO_CACHE) < 150:
                        _AUDIO_CACHE[cache_key] = f.read()
                return output_path
            # If Kokoro failed, try edge-tts as backup
            logger.info("Falling back to Edge-TTS backup...")

        # 4. Edge TTS
        res = await self._generate_edge_tts(text, output_path, self.model)
        if res and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, "rb") as f:
                if len(_AUDIO_CACHE) < 150:
                    _AUDIO_CACHE[cache_key] = f.read()
            return output_path

        # If audio generation failed or timed out, return None so UI can use instant WebSpeech
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass
        return None
