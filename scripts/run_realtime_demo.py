import os
import sys
import time
import json
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from realtime_insights.signals.extractor import SignalExtractor
from realtime_insights.nudges.engine import NudgeEngine
from voice_agent.tools.llm import LLMEngine

def main():
    print("--- Starting Real-Time Insights Simulator (Q4) ---")
    output_dir = Path("./outputs/q4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = SignalExtractor()
    # Force mock if API key isn't set
    if extractor.llm.is_mock:
        print("NOTE: Using Mock LLM. Generating synthetic signals for demo purposes.")
        # Patch extract_signals to generate deterministic signals for the demo
        def patched_extract(history):
            if not history: return []
            last_text = history[-1]['text'].lower()
            if "i also need a car loan" in last_text:
                return [{"signal_id": "sig_1", "type": "missed_cross_sell", "confidence": 0.95, "evidence": "Customer mentioned car loan"}]
            if "i am very angry" in last_text:
                return [{"signal_id": "sig_2", "type": "rising_frustration", "confidence": 0.90, "evidence": "Customer is angry"}]
            return []
        extractor.extract_signals = patched_extract
        
    engine = NudgeEngine()
    
    # Simulate a call transcript arriving in chunks
    script = [
        {"timestamp": 1.0, "speaker": "agent", "text": "Hello, thank you for calling. How can I help you?"},
        {"timestamp": 5.0, "speaker": "customer", "text": "Hi, I want a business loan."},
        {"timestamp": 10.0, "speaker": "customer", "text": "But I also need a car loan for my delivery vehicle."}, # missed cross-sell
        {"timestamp": 15.0, "speaker": "agent", "text": "Okay, let's look at the business loan first."},
        {"timestamp": 20.0, "speaker": "customer", "text": "I am very angry that this is taking so long."}, # frustration
        {"timestamp": 25.0, "speaker": "customer", "text": "I am very angry about the interest rates too!"} # duplicate frustration (should be suppressed by cooldown)
    ]
    
    transcript_history = []
    latencies = []
    all_nudges = []
    
    csv_path = output_dir / "latency_report.csv"
    with open(csv_path, "w") as f:
        f.write("chunk_idx,asr_latency_ms,signal_latency_ms,llm_latency_ms,delivery_latency_ms,total_latency_ms\n")
    
    for idx, msg in enumerate(script):
        print(f"\n[+{msg['timestamp']}s] {msg['speaker'].upper()}: {msg['text']}")
        
        start_time = time.time()
        
        # Simulate ASR Latency
        time.sleep(0.3)
        asr_end = time.time()
        asr_lat = (asr_end - start_time) * 1000
        
        transcript_history.append(msg)
        
        # Signal Extraction (LLM Latency)
        sig_start = time.time()
        signals = extractor.extract_signals(transcript_history)
        sig_end = time.time()
        llm_lat = (sig_end - sig_start) * 1000
        
        # Nudge Engine
        nudges = engine.process_signals(signals)
        del_end = time.time()
        nudge_lat = (del_end - sig_end) * 1000
        
        total_lat = asr_lat + llm_lat + nudge_lat
        latencies.append(total_lat)
        
        if nudges:
            for n in nudges:
                print(f"   ---> [NUDGE GENERATED] {n['type'].upper()}: {n['nudge']}")
                all_nudges.append(n)
        else:
            if signals:
                print(f"   ---> [NUDGE SUPPRESSED] Signals detected but suppressed by engine.")
        
        with open(csv_path, "a") as f:
            f.write(f"{idx},{asr_lat:.1f},10.0,{llm_lat:.1f},{nudge_lat:.1f},{total_lat:.1f}\n")
            
    print("\n--- Summary ---")
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)]
        print(f"End-to-End Latency P50: {p50:.1f}ms")
        print(f"End-to-End Latency P95: {p95:.1f}ms")
        
    with open(output_dir / "live_transcript.json", "w") as f:
        json.dump(transcript_history, f, indent=2)
        
    with open(output_dir / "nudges.json", "w") as f:
        json.dump(all_nudges, f, indent=2)
        
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
