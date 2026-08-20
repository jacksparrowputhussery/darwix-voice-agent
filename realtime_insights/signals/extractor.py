import os
import sys
import json
import uuid
import time
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from voice_agent.tools.llm import LLMEngine

class SignalExtractor:
    def __init__(self):
        self.llm = LLMEngine()
        # Ensure it knows to extract signals
        self.system_prompt = """
You are a real-time call insights extractor.
Analyze the provided transcript so far, particularly the newest turns.
Look for the following signals:
- missed_cross_sell: The customer mentions an additional need or product.
- compliance_gap: A required disclosure is missing before a critical step.
- rising_frustration: Customer expresses annoyance or anger.
- payment_difficulty: Customer indicates inability to pay.

Return ONLY valid JSON matching this schema:
{
    "signals": [
        {
            "signal_id": "...",
            "type": "missed_cross_sell",
            "confidence": 0.9,
            "evidence": "Customer mentioned ..."
        }
    ]
}
If no new signals are present, return {"signals": []}.
"""

    def extract_signals(self, transcript_history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        history_text = "\n".join([f"[{msg['timestamp']}s] {msg['speaker'].upper()}: {msg['text']}" for msg in transcript_history])
        
        user_content = f"""
TRANSCRIPT HISTORY:
{history_text}
"""
        try:
            if self.llm.is_mock:
                print("Mock Signal Extractor: returning empty signals")
                return []
                
            data = self.llm.generate_json(self.system_prompt, user_content)
            return data.get("signals", [])
        except Exception as e:
            print(f"Signal Extractor Error: {e}")
            return []
