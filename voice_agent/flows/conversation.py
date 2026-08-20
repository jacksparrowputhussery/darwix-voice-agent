import os
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.append(str(Path(__file__).parent.parent.parent))

from voice_agent.tools.stt import STTEngine
from voice_agent.tools.tts import TTSEngine
from voice_agent.tools.llm import LLMEngine
from voice_agent.flows.rules import BusinessLoanRuleEngine
from voice_agent.prompts.system_prompt import SYSTEM_PROMPT_BUSINESS_LOAN
from knowledge_base.retrieval.vector_store import VectorStore

class VoiceSessionManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.state = {
            "session_id": session_id,
            "qualification": {},
            "conversation_history": [],  # List of dictionaries: [{"speaker": "customer"|"agent", "text": "...", "role": "user"|"assistant"}]
            "objections": [],
            "escalation_requested": False,
            "human_required": False
        }
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.llm = LLMEngine()
        self.kb = VectorStore()
        
    def reset_session(self):
        """Reset the conversation and qualification state."""
        self.state = {
            "session_id": self.session_id,
            "qualification": {},
            "conversation_history": [],
            "objections": [],
            "escalation_requested": False,
            "human_required": False
        }

    async def _process_user_text(self, user_text: str, skip_tts: bool = False) -> Dict[str, Any]:
        """
        Shared pipeline: Retrieval -> Grounded LLM Response with Conversational History -> Rule Engine -> TTS Audio.
        Maintains conversation history as a list of dictionaries.
        """
        # 1. Append user turn to conversation history (list of dictionaries)
        self.state["conversation_history"].append({
            "speaker": "customer",
            "text": user_text,
            "role": "user"
        })
        
        # 2. Compute missing qualification fields
        collected = self.state.get("qualification", {})
        missing = [f for f in BusinessLoanRuleEngine.REQUIRED_FIELDS if collected.get(f) is None]

        # 3. Retrieve Grounding Context
        results = self.kb.retrieve(user_text, top_k=2)
        context = ""
        citations = []
        for r in results:
            context += f"Source ({r.source}): {r.content}\n\n"
            citations.append(r.source)
            
        if not context:
            context = "No specific policy information found."

        # 4. Generate Grounded Response using full conversational history & state
        llm_out = self.llm.generate_response(
            system_prompt=SYSTEM_PROMPT_BUSINESS_LOAN,
            context=context,
            user_input=user_text,
            conversation_history=self.state["conversation_history"],
            qualification_state=self.state["qualification"],
            missing_fields=missing
        )
        
        response_text = llm_out.get("response_text", "I'm having trouble processing that.")
        extracted = llm_out.get("extracted_fields", {})
        intent = llm_out.get("intent", "general")
        escalate = llm_out.get("escalate", False)
        
        # 5. Update Qualification State (only non-None values, never overwrite known data with None)
        # First, apply heuristic extraction from user_text to ensure no obvious entity is missed
        heuristic_extracted = self.llm._offline_fallback(user_text, self.state["qualification"]).get("extracted_fields", {})
        combined_extracted = dict(heuristic_extracted)
        if isinstance(extracted, dict):
            for k, v in extracted.items():
                if v is not None and v != "":
                    combined_extracted[k] = v

        if isinstance(combined_extracted, dict):
            import re
            for k, v in combined_extracted.items():
                if v is not None and v != "":
                    if k in ["time_in_business_months", "monthly_revenue", "credit_score"]:
                        try:
                            if isinstance(v, str):
                                digits = re.sub(r"[^\d]", "", v)
                                if digits:
                                    v = int(digits)
                            elif isinstance(v, (int, float)):
                                v = int(v)
                        except Exception:
                            pass
                    self.state["qualification"][k] = v
        if intent == "objection":
            self.state["objections"].append(user_text)
        if escalate:
            self.state["escalation_requested"] = True
            
        # 6. Evaluate Business Loan Rules
        requires_escalation = BusinessLoanRuleEngine.requires_escalation(self.state)
        if requires_escalation:
            response_text = "It seems we need a human representative to assist you further. I am transferring you now."
            self.state["human_required"] = True
            
        is_qualified, message = BusinessLoanRuleEngine.evaluate(self.state)
        if is_qualified and "congratulations" not in self.state.get("last_status", ""):
            response_text += " Based on your answers, you meet our preliminary requirements! A loan officer will review your application."
            self.state["last_status"] = "congratulations"

        # 7. Append Agent turn to conversation history
        self.state["conversation_history"].append({
            "speaker": "agent",
            "text": response_text,
            "role": "assistant"
        })

        # 8. Generate TTS (only if not skipped)
        tts_audio_path = None
        if not skip_tts:
            tts_audio_path = await self.tts.generate_audio(response_text)
        
        return {
            "response_text": response_text,
            "tts_audio_path": tts_audio_path,
            "user_transcription": user_text,
            "citations": citations,
            "qualification": self.state["qualification"],
            "is_qualified": is_qualified,
            "qualification_message": message,
            "requires_escalation": requires_escalation or self.state.get("human_required", False),
            "conversation_history": self.state["conversation_history"],
            "state": self.state
        }

    async def process_turn(self, audio_path: str, skip_tts: bool = False) -> Dict[str, Any]:
        """
        Process a single conversational turn from an audio file.
        """
        user_text = self.stt.transcribe(audio_path)
        return await self._process_user_text(user_text, skip_tts=skip_tts)

    async def process_text_turn(self, user_text: str, skip_tts: bool = False) -> Dict[str, Any]:
        """
        Process a single conversational turn from text input.
        """
        return await self._process_user_text(user_text, skip_tts=skip_tts)
