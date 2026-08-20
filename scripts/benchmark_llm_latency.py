import os
import sys
import time
import json
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from voice_agent.tools.llm import LLMEngine
from voice_agent.prompts.system_prompt import SYSTEM_PROMPT_BUSINESS_LOAN

def run_benchmark():
    print("\n=======================================================")
    print("   AI Voice Agent: LLM Latency & Fallback Benchmark    ")
    print("=======================================================\n")
    
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    test_context = "Minimum revenue required is $5,000/month. Minimum 6 months in business. Minimum credit score 650."
    test_input = "Hi, we have been operating for 2 years in Texas and our monthly revenue is around $12,000. My credit score is 710."
    test_history = [{"speaker": "agent", "text": "Hello! How can I help your business today?"}]
    
    results = []

    # 1. Test Standard Multi-Provider Engine (Groq -> Gemini fallback)
    print("[1/3] Testing Configured LLMEngine (Groq primary with Gemini fallback)...")
    engine = LLMEngine()
    start_t = time.time()
    response = engine.generate_response(
        system_prompt=SYSTEM_PROMPT_BUSINESS_LOAN,
        context=test_context,
        user_input=test_input,
        conversation_history=test_history
    )
    total_time = (time.time() - start_t) * 1000
    metrics = engine.last_metrics
    
    results.append({
        "mode": "Default Pipeline",
        "provider_used": metrics.get("provider"),
        "model": metrics.get("model", "n/a"),
        "fallback_used": metrics.get("fallback_used"),
        "latency_ms": round(total_time, 2),
        "response_excerpt": response.get("response_text", "")[:80] + "...",
        "extracted_fields": response.get("extracted_fields", {})
    })
    print(f"      -> Provider Used: {metrics.get('provider')} | Latency: {total_time:.2f}ms | Fallback: {metrics.get('fallback_used')}")

    # 2. Test Forced Fallback (Simulating Groq Downtime/Failure)
    print("\n[2/3] Simulating Groq Outage / Fast Failover -> Fallback to Gemini...")
    fallback_engine = LLMEngine()
    if fallback_engine.groq_client:
        # Intentionally disable Groq client to test seamless fallback
        fallback_engine.groq_client = None
    
    start_t = time.time()
    fallback_response = fallback_engine.generate_response(
        system_prompt=SYSTEM_PROMPT_BUSINESS_LOAN,
        context=test_context,
        user_input=test_input,
        conversation_history=test_history
    )
    total_fallback_time = (time.time() - start_t) * 1000
    fallback_metrics = fallback_engine.last_metrics
    
    results.append({
        "mode": "Forced Fallback (Groq Disabled)",
        "provider_used": fallback_metrics.get("provider"),
        "model": fallback_metrics.get("model", "n/a"),
        "fallback_used": fallback_metrics.get("fallback_used"),
        "latency_ms": round(total_fallback_time, 2),
        "response_excerpt": fallback_response.get("response_text", "")[:80] + "...",
        "extracted_fields": fallback_response.get("extracted_fields", {})
    })
    print(f"      -> Provider Used: {fallback_metrics.get('provider')} | Latency: {total_fallback_time:.2f}ms | Fallback: {fallback_metrics.get('fallback_used')}")

    # 3. Test Offline Fallback (Simulating All APIs Down)
    print("\n[3/3] Simulating Total API Outage -> Fallback to Rule-based Offline Extraction...")
    offline_engine = LLMEngine()
    offline_engine.groq_client = None
    offline_engine.gemini_model = None
    
    start_t = time.time()
    offline_response = offline_engine.generate_response(
        system_prompt=SYSTEM_PROMPT_BUSINESS_LOAN,
        context=test_context,
        user_input=test_input,
        conversation_history=test_history
    )
    total_offline_time = (time.time() - start_t) * 1000
    offline_metrics = offline_engine.last_metrics
    
    results.append({
        "mode": "Offline Heuristic Fallback",
        "provider_used": offline_metrics.get("provider"),
        "model": "rule_based",
        "fallback_used": True,
        "latency_ms": round(total_offline_time, 2),
        "response_excerpt": offline_response.get("response_text", "")[:80] + "...",
        "extracted_fields": offline_response.get("extracted_fields", {})
    })
    print(f"      -> Provider Used: {offline_metrics.get('provider')} | Latency: {total_offline_time:.2f}ms | Fallback: True")

    # Print Formatted Results Table
    print("\n" + "="*85)
    print(f"{'Mode':<30} | {'Provider':<16} | {'Latency (ms)':<14} | {'Fallback?':<10}")
    print("="*85)
    for r in results:
        print(f"{r['mode']:<30} | {str(r['provider_used']):<16} | {r['latency_ms']:<14} | {str(r['fallback_used']):<10}")
    print("="*85)
    
    report_file = output_dir / "llm_benchmark_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark report to {report_file}\n")

if __name__ == "__main__":
    run_benchmark()
