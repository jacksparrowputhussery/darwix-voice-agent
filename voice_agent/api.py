import os
import sys
import time
import json
import base64
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path when running script directly
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from voice_agent.flows.conversation import VoiceSessionManager
from voice_agent.flows.rules import BusinessLoanRuleEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Darwix AI Voice Intelligence Platform", version="2.0.0")

# Store active sessions in memory
sessions = {}

def get_or_create_session(session_id: str = "default_session") -> VoiceSessionManager:
    if session_id not in sessions:
        sessions[session_id] = VoiceSessionManager(session_id=session_id)
    return sessions[session_id]

# Path to frontend
FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.on_event("startup")
async def startup_event():
    """Pre-warm STT engine, Kokoro TTS, ChromaDB, and session singletons in background."""
    logger.info("⚡ Pre-warming speech, vector, and neural synthesis engines for ultra-low latency...")
    loop = asyncio.get_event_loop()
    def warmup():
        try:
            from voice_agent.tools.stt import STTEngine
            from voice_agent.tools.tts import TTSEngine
            from knowledge_base.retrieval.vector_store import VectorStore
            STTEngine().warmup()
            TTSEngine().warmup()
            VectorStore().retrieve("business loan qualification", top_k=1)
            logger.info("✓ Background engine warmup complete.")
        except Exception as e:
            logger.warning(f"Engine warmup notice: {e}")
    loop.run_in_executor(None, warmup)

@app.get("/")
async def get_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>Darwix Voice Agent UI</h1><p>Frontend assets not found.</p>")

class TextMessageRequest(BaseModel):
    text: str
    session_id: Optional[str] = "web_session"
    skip_tts: Optional[bool] = False

@app.get("/api/status")
async def get_status():
    """Get system health and provider details."""
    primary_provider = os.getenv("LLM_PRIMARY_PROVIDER", "groq")
    fallback_provider = os.getenv("LLM_FALLBACK_PROVIDER", "gemini")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    
    tts_provider = os.getenv("TTS_PROVIDER", "kokoro")
    tts_voice = os.getenv("TTS_MODEL", "af_bella")
    
    return {
        "status": "online",
        "primary_provider": primary_provider,
        "fallback_provider": fallback_provider,
        "groq_model": groq_model,
        "gemini_model": gemini_model,
        "vector_store": "ChromaDB (Local)",
        "tts_engine": f"Kokoro-82M ({tts_voice})" if tts_provider == "kokoro" else f"{tts_provider} ({tts_voice})",
        "asr_engine": os.getenv("ASR_PROVIDER", "faster-whisper")
    }

@app.get("/api/requirements")
async def get_requirements():
    """Get the active dynamic qualification requirements parsed from business_loan_policy.txt."""
    from voice_agent.flows.rules import BusinessLoanRuleEngine
    return {"requirements": BusinessLoanRuleEngine.load_requirements_from_policy()}

@app.get("/api/session")
async def get_session_state(session_id: str = "web_session"):
    """Get the active session qualification state, requirements, and conversation history."""
    from voice_agent.flows.rules import BusinessLoanRuleEngine
    session = get_or_create_session(session_id)
    is_qualified, message = BusinessLoanRuleEngine.evaluate(session.state)
    return {
        "session_id": session_id,
        "requirements": BusinessLoanRuleEngine.load_requirements_from_policy(),
        "qualification": session.state.get("qualification", {}),
        "conversation_history": session.state.get("conversation_history", []),
        "is_qualified": is_qualified,
        "qualification_message": message,
        "human_required": session.state.get("human_required", False)
    }

@app.post("/api/reset")
async def reset_session_endpoint(session_id: str = "web_session"):
    """Reset the conversation and qualification state."""
    session = get_or_create_session(session_id)
    session.reset_session()
    return {"status": "reset_successful", "session_id": session_id}

@app.post("/api/chat")
async def chat_text(req: TextMessageRequest):
    """Process a text message turn with low latency."""
    from voice_agent.flows.rules import BusinessLoanRuleEngine
    start_time = time.time()
    session = get_or_create_session(req.session_id)
    result = await session.process_text_turn(req.text, skip_tts=req.skip_tts)
    latency_ms = round((time.time() - start_time) * 1000, 1)
    
    audio_base64 = None
    tts_path = result.get("tts_audio_path")
    if tts_path and os.path.exists(tts_path):
        try:
            with open(tts_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(tts_path)
        except Exception as e:
            logger.warning(f"Error reading TTS audio: {e}")

    return {
        "type": "turn_result",
        "user_transcription": result["user_transcription"],
        "response_text": result["response_text"],
        "citations": result.get("citations", []),
        "requirements": BusinessLoanRuleEngine.load_requirements_from_policy(),
        "qualification": result.get("qualification", {}),
        "is_qualified": result.get("is_qualified", False),
        "qualification_message": result.get("qualification_message", ""),
        "requires_escalation": result.get("requires_escalation", False),
        "conversation_history": result.get("conversation_history", []),
        "latency_ms": latency_ms,
        "audio_base64": audio_base64
    }

@app.post("/api/audio")
async def process_audio_upload(file: UploadFile = File(...), session_id: str = Form("web_session")):
    """Process an uploaded audio recording."""
    start_time = time.time()
    session = get_or_create_session(session_id)
    
    suffix = Path(file.filename or "recording.webm").suffix or ".webm"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    os.close(fd)
    
    try:
        result = await session.process_turn(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    latency_ms = round((time.time() - start_time) * 1000, 1)
    
    audio_base64 = None
    tts_path = result.get("tts_audio_path")
    if tts_path and os.path.exists(tts_path):
        try:
            with open(tts_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode("utf-8")
            os.remove(tts_path)
        except Exception as e:
            logger.warning(f"Error reading TTS audio: {e}")

    return {
        "type": "turn_result",
        "user_transcription": result["user_transcription"],
        "response_text": result["response_text"],
        "citations": result.get("citations", []),
        "qualification": result.get("qualification", {}),
        "is_qualified": result.get("is_qualified", False),
        "qualification_message": result.get("qualification_message", ""),
        "requires_escalation": result.get("requires_escalation", False),
        "conversation_history": result.get("conversation_history", []),
        "latency_ms": latency_ms,
        "audio_base64": audio_base64
    }

@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = get_or_create_session("web_session")
    
    try:
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": "Connected to Darwix Voice Engine",
            "state": "ready"
        }))
        
        while True:
            message = await websocket.receive()
            start_time = time.time()
            
            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
                
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "Transcribing speech...",
                    "state": "processing"
                }))
                
                fd, temp_path = tempfile.mkstemp(suffix=".webm")
                with open(temp_path, "wb") as f:
                    f.write(data)
                os.close(fd)
                
                try:
                    result = await session.process_turn(temp_path)
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    payload = {"type": "text", "text": message["text"]}
                    
                msg_type = payload.get("type", "text")
                if msg_type == "reset":
                    session.reset_session()
                    await websocket.send_text(json.dumps({
                        "type": "reset_complete",
                        "message": "Session reset successfully"
                    }))
                    continue
                    
                user_text = payload.get("text", "")
                if not user_text:
                    continue
                    
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "Generating grounded response...",
                    "state": "processing"
                }))
                
                skip_tts = payload.get("skip_tts", False)
                result = await session.process_text_turn(user_text, skip_tts=skip_tts)
            else:
                continue

            latency_ms = round((time.time() - start_time) * 1000, 1)
            
            tts_path = result.get("tts_audio_path")
            audio_base64 = None
            raw_audio_bytes = None
            if tts_path and os.path.exists(tts_path):
                try:
                    with open(tts_path, "rb") as f:
                        raw_audio_bytes = f.read()
                        audio_base64 = base64.b64encode(raw_audio_bytes).decode("utf-8")
                    os.remove(tts_path)
                except Exception as e:
                    logger.warning(f"TTS reading error: {e}")

            payload_out = {
                "type": "turn_result",
                "user_transcription": result["user_transcription"],
                "response_text": result["response_text"],
                "citations": result.get("citations", []),
                "requirements": BusinessLoanRuleEngine.load_requirements_from_policy(),
                "qualification": result.get("qualification", {}),
                "is_qualified": result.get("is_qualified", False),
                "qualification_message": result.get("qualification_message", ""),
                "requires_escalation": result.get("requires_escalation", False),
                "conversation_history": result.get("conversation_history", []),
                "latency_ms": latency_ms,
                "audio_base64": audio_base64
            }
            
            await websocket.send_text(json.dumps(payload_out))
            
            if raw_audio_bytes:
                await websocket.send_bytes(raw_audio_bytes)
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.warning(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
