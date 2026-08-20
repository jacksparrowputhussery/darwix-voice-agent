import os
import pytest
import asyncio
from pathlib import Path
from voice_agent.tools.tts import TTSEngine, load_kokoro_engine, get_kokoro_model_paths

@pytest.mark.asyncio
async def test_kokoro_model_assets_exist():
    model_path, voices_path = get_kokoro_model_paths()
    assert model_path.exists(), f"Model file {model_path} should exist"
    assert voices_path.exists(), f"Voices file {voices_path} should exist"
    assert model_path.stat().st_size > 1000000

@pytest.mark.asyncio
async def test_kokoro_engine_gpu_or_cpu_fallback():
    engine, device = load_kokoro_engine()
    assert engine is not None
    assert "cpu" in device.lower() or "gpu" in device.lower()

@pytest.mark.asyncio
async def test_tts_engine_kokoro_synthesis():
    tts = TTSEngine()
    tts.provider = "kokoro"
    tts.model = "af_bella"
    
    output_path = await tts.generate_audio("Hello, your business loan qualification test passed successfully.")
    assert output_path is not None
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 1000
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)
