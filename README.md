# AI Voice Agent & Real-Time Intelligence Platform

## 1. Overview
This project implements an AI Voice Agent and Real-Time Intelligence Platform, featuring a production-ready knowledge base, a knowledge-grounded voice agent, multilingual support for the Philippines and Indonesia, and a real-time call insights extraction pipeline. The primary domain is **Business-loan qualification**.

## 2. Assessment Coverage
- **Q1:** Knowledge-Grounded Voice Agent (Business-loan qualification)
- **Q2:** Production-Ready Knowledge Base (Supports text, web, pdf, deduplication, PII redaction)
- **Q3:** Native-Language Voice Bots (Philippines Taglish, Indonesia Multifinance)
- **Q4:** Live Insights and Nudges (Real-time processing and latency tracking)

## 3. High-Performance Multi-Provider LLM Architecture

To minimize conversational latency and ensure high availability, the platform uses a tiered multi-provider LLM strategy:

```
[Voice Agent / Insights Extractor]
                │
                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  1. PRIMARY: Groq LLM (Ultra-Low Latency Inference)         │
  │     • Models: llama-3.3-70b-versatile / llama-3.1-8b-instant│
  │     • Fast-failover timeout: 4.0s                           │
  │     • Structured JSON response mode                         │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ (On RateLimit, Timeout, Error)
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  2. FALLBACK: Google Gemini API                             │
  │     • Models: gemini-2.5-flash-lite / gemini-1.5-flash      │
  │     • Structured JSON response mode                         │
  └──────────────────────────────┬──────────────────────────────┘
                                 │ (On Quota Exceeded / Network Failure)
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  3. OFFLINE FALLBACK: Rule-Based Heuristic Extraction       │
  │     • Zero-API fallback with smart entity extraction        │
  └─────────────────────────────────────────────────────────────┘
```

### Key Latency & Reliability Benefits:
- **Ultra-Low Latency:** Groq delivers responses at 300–800+ tokens/second, cutting LLM turn latency significantly for smooth voice communication.
- **Resilient Fallback:** Automatically redirects to Gemini upon Groq failure or rate limiting, preventing voice session dropouts.
- **Fail-Safe Offline Mode:** Guarantees that even if all cloud APIs are exhausted or down, basic loan qualification and intent recognition continue uninterrupted.

## 4. Technology Choices
- **Primary LLM:** Groq API (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) for ultra-low latency.
- **Fallback LLM / Embeddings:** Gemini API (`gemini-2.5-flash-lite`, `gemini-embedding-001`).
- **ASR:** `faster-whisper` for free, robust local speech recognition.
- **TTS:** `Kokoro-82M` (Local neural TTS with dynamic GPU detection & CPU fallback; secondary fallback to `edge-tts`).
- **Vector DB:** `chromadb` for lightweight local vector storage.
- **Backend:** `FastAPI` to handle streaming WebSockets and asynchronous tasks.

## 5. Setup & Configuration

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your environment variables in `.env` (refer to `.env.example`):
   ```env
   # API Keys
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here

   # LLM Providers & Models
   LLM_PRIMARY_PROVIDER=groq
   LLM_FALLBACK_PROVIDER=gemini
   GROQ_MODEL=llama-3.3-70b-versatile
   GEMINI_MODEL=gemini-2.5-flash-lite
   GROQ_TIMEOUT=4.0

   # TTS Configuration
   TTS_PROVIDER=kokoro
   TTS_MODEL=af_bella
   KOKORO_DEVICE=auto
   ```
   *Note: If no API key is set, the system automatically falls back to offline rule-based extraction for zero-downtime local testing.*

## 6. How to Run Demos & Benchmarks

### LLM Latency & Fallback Benchmark
- Run the multi-provider benchmark and verify fallback live:
  ```bash
  python scripts/benchmark_llm_latency.py
  ```
- Output: `outputs/llm_benchmark_report.json`

### Unit Tests for Fallback Resilience
- Run automated fallback tests:
  ```bash
  pytest voice_agent/tests/test_llm_fallback.py -v
  ```

### Question 1: Knowledge-Grounded Voice Agent
- Live voice backend (web calling interface): `python voice_agent/api.py` (Open `http://localhost:8000`)
- Automated batch tests: `python scripts/run_voice_demo.py`
- Output: `outputs/q1/`

### Question 2: Knowledge Base Ingestion & Evaluation
- Ingest raw data into ChromaDB: `python scripts/ingest_data.py`
- Run retrieval tests: `python scripts/evaluate_retrieval.py`
- Output: `outputs/retrieval_tests.csv`

### Question 3: Native-Language Voice Bots
- Run Taglish and Bahasa Indonesia tests: `python scripts/run_multilingual_demo.py`
- Output: `outputs/q3/`

### Question 4: Real-Time Insights & Nudges
- Run real-time simulation with low-latency signal extraction: `python scripts/run_realtime_demo.py`
- Output: `outputs/q4/` (live nudges, suppression logs, and latency report)
