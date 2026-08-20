import os
import time
import json
import logging
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

# Try loading Groq SDK
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Try loading Google Generative AI SDK
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def _clean_json_text(text: str) -> str:
    """Helper to extract clean JSON string from LLM responses with markdown fences or prefix."""
    text = text.strip()
    # If fenced with ```json ... ```
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if json_match:
        text = json_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1].strip()
    return text


class LLMEngine:
    """
    Multi-Provider LLM Engine with Conversational History and State Tracking:
    1. Primary: Groq (Ultra-Low Latency Inference)
    2. Fallback: Google Gemini (gemini-2.5-flash-lite / gemini-1.5-flash)
    3. Tertiary: Smart Rule-Based Offline Fallback with History Awareness
    """

    def __init__(self):
        self.primary_provider = os.getenv("LLM_PRIMARY_PROVIDER", os.getenv("LLM_PROVIDER", "groq")).lower()
        self.fallback_provider = os.getenv("LLM_FALLBACK_PROVIDER", "gemini").lower()
        
        self.groq_model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.gemini_model_name = os.getenv("GEMINI_MODEL", os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"))
        
        self.groq_timeout = float(os.getenv("GROQ_TIMEOUT", "2.5"))
        
        self.last_metrics = {
            "provider": None,
            "latency_ms": 0.0,
            "fallback_used": False,
            "error": None
        }

        # Initialize Groq client
        self.groq_client = None
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        if GROQ_AVAILABLE and groq_api_key and groq_api_key != "your_groq_api_key_here":
            try:
                self.groq_client = Groq(api_key=groq_api_key, timeout=self.groq_timeout)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

        # Initialize Gemini client
        self.gemini_model = None
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if GEMINI_AVAILABLE and gemini_api_key and gemini_api_key != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel(self.gemini_model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini model: {e}")

        self.model = self.gemini_model
        self.model_name = self.groq_model_name if self.groq_client else self.gemini_model_name
        self.is_mock = (self.groq_client is None and self.gemini_model is None)

    def _call_groq(
        self,
        system_prompt: str,
        user_content: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        json_mode: bool = True
    ) -> Dict[str, Any]:
        """Calls Groq API with multi-turn messages and structured JSON output."""
        if not self.groq_client:
            raise RuntimeError("Groq client not configured or API key missing.")
        
        if messages is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        
        kwargs = {
            "model": self.groq_model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1024,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
            
        completion = self.groq_client.chat.completions.create(**kwargs)
        raw_text = completion.choices[0].message.content
        
        if json_mode:
            cleaned = _clean_json_text(raw_text)
            return json.loads(cleaned)
        return {"response_text": raw_text}

    def _call_gemini(
        self,
        system_prompt: str,
        user_content: str = "",
        messages: Optional[List[Dict[str, str]]] = None,
        json_mode: bool = True
    ) -> Dict[str, Any]:
        """Calls Google Gemini API with multi-turn history."""
        if not self.gemini_model:
            raise RuntimeError("Gemini model not configured or API key missing.")
        
        if messages:
            history_lines = []
            for m in messages:
                if m["role"] != "system":
                    history_lines.append(f"{m['role'].upper()}: {m['content']}")
            combined_prompt = f"{system_prompt}\n\nCONVERSATION HISTORY:\n" + "\n".join(history_lines)
        else:
            combined_prompt = f"{system_prompt}\n\n{user_content}"
        
        config = {"temperature": 0.1}
        if json_mode:
            config["response_mime_type"] = "application/json"
            
        response = self.gemini_model.generate_content(
            combined_prompt,
            generation_config=config
        )
        raw_text = response.text
        
        if json_mode:
            cleaned = _clean_json_text(raw_text)
            return json.loads(cleaned)
        return {"response_text": raw_text}

    def generate_json(self, system_prompt: str, user_content: str) -> Dict[str, Any]:
        """Generic JSON generator with automatic Groq -> Gemini -> Mock fallback."""
        start_time = time.time()
        
        if self.groq_client and self.primary_provider == "groq":
            try:
                result = self._call_groq(system_prompt, user_content, json_mode=True)
                latency = (time.time() - start_time) * 1000
                self.last_metrics = {
                    "provider": "groq",
                    "latency_ms": latency,
                    "fallback_used": False,
                    "error": None
                }
                return result
            except Exception as e:
                logger.warning(f"Groq primary generation failed ({e}). Falling back to Gemini...")
                self.last_metrics["error"] = f"Groq error: {e}"

        if self.gemini_model:
            try:
                gemini_start = time.time()
                result = self._call_gemini(system_prompt, user_content, json_mode=True)
                latency = (time.time() - gemini_start) * 1000
                self.last_metrics = {
                    "provider": "gemini",
                    "latency_ms": latency,
                    "fallback_used": True,
                    "error": self.last_metrics.get("error")
                }
                return result
            except Exception as e:
                logger.warning(f"Gemini fallback generation failed: {e}")
                self.last_metrics["error"] = f"Gemini error: {e}"

        self.last_metrics = {
            "provider": "offline_fallback",
            "latency_ms": 0.0,
            "fallback_used": True,
            "error": "All LLM providers unavailable"
        }
        return {}

    def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_input: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        qualification_state: Optional[Dict[str, Any]] = None,
        missing_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generates a grounded voice response with full multi-turn conversational memory.
        Explicitly prevents asking for already collected fields.
        """
        qualification_state = qualification_state or {}
        collected_summary = []
        for k, v in qualification_state.items():
            if v is not None:
                collected_summary.append(f"  • {k}: {v}")
        collected_str = "\n".join(collected_summary) if collected_summary else "  (None collected yet)"
        
        missing_fields = missing_fields or ["time_in_business_months", "monthly_revenue", "credit_score", "location"]
        missing_str = ", ".join(missing_fields) if missing_fields else "None (All required fields collected!)"

        enriched_system_prompt = f"""{system_prompt}

KNOWLEDGE BASE CONTEXT (Use strictly for factual grounding):
{context}

CURRENT SESSION QUALIFICATION PROGRESS:
- Already Collected & Known Fields (DO NOT RE-ASK FOR THESE):
{collected_str}
- Missing Fields Still Needed:
  [{missing_str}]

CRITICAL CONVERSATIONAL RULES:
1. Review the entire conversation history and the collected fields above.
2. NEVER re-ask for any information already provided in conversation history or listed in Collected Fields.
3. If missing fields exist, acknowledge the user's latest statement and ask ONLY for ONE missing field from the list above.
4. If ALL required fields are collected, inform the user they meet preliminary requirements and a loan officer will review their application.
5. Keep spoken responses short (1-2 sentences), conversational, and friendly.
6. Respond in strictly valid JSON with keys:
   - "response_text": Spoken response to customer.
   - "extracted_fields": Dictionary of newly extracted fields from this turn (time_in_business_months, monthly_revenue, credit_score, location). If the customer provides qualitative statements like 'greater than requirements', 'above 700', 'excellent', 'good', or numbers like 950, extract credit_score as an appropriate integer (e.g. 720, 750, 950).
   - "intent": Intent (providing_info, question, objection, escalate).
   - "escalate": Boolean (true if human requested or unanswerable).
"""

        # Construct multi-turn message history (list of dicts)
        messages = [{"role": "system", "content": enriched_system_prompt}]
        if conversation_history:
            for turn in conversation_history[-8:]:
                speaker = turn.get("speaker", turn.get("role", "customer"))
                role = "user" if speaker in ["customer", "user"] else "assistant"
                text = turn.get("text", turn.get("content", ""))
                if text:
                    messages.append({"role": role, "content": text})
        else:
            messages.append({"role": "user", "content": user_input})

        start_time = time.time()

        # 1. Primary: Groq
        if self.groq_client and self.primary_provider == "groq":
            try:
                res = self._call_groq(enriched_system_prompt, messages=messages, json_mode=True)
                latency = (time.time() - start_time) * 1000
                self.last_metrics = {
                    "provider": "groq",
                    "model": self.groq_model_name,
                    "latency_ms": latency,
                    "fallback_used": False,
                    "error": None
                }
                return res
            except Exception as e:
                logger.warning(f"Groq generation failed ({e}). Falling back to Gemini...")
                self.last_metrics["error"] = f"Groq error: {e}"

        # 2. Secondary Fallback: Google Gemini
        if self.gemini_model:
            max_gemini_retries = 2
            for attempt in range(max_gemini_retries):
                try:
                    gemini_start = time.time()
                    res = self._call_gemini(enriched_system_prompt, messages=messages, json_mode=True)
                    latency = (time.time() - gemini_start) * 1000
                    self.last_metrics = {
                        "provider": "gemini",
                        "model": self.gemini_model_name,
                        "latency_ms": latency,
                        "fallback_used": True,
                        "error": None
                    }
                    return res
                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"Gemini generation error: {e}")
                    self.last_metrics["error"] = f"Gemini error: {e}"
                    break

        # 3. Smart Offline Fallback with History Awareness
        self.last_metrics = {
            "provider": "offline_fallback",
            "model": "rule_based_history",
            "latency_ms": (time.time() - start_time) * 1000,
            "fallback_used": True,
            "error": "All online LLM providers unavailable"
        }
        return self._offline_fallback(
            user_input=user_input,
            qualification_state=qualification_state,
            missing_fields=missing_fields,
            conversation_history=conversation_history
        )

    def _offline_fallback(
        self,
        user_input: str,
        qualification_state: Optional[Dict[str, Any]] = None,
        missing_fields: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Intelligent offline fallback with conversation history and state awareness."""
        u_lower = user_input.lower()
        extracted = {}

        # Heuristic field extraction with regex
        time_match = re.search(r'(\d+)\s*(years?|yrs?|months?|mos?)', u_lower)
        if time_match:
            val = int(time_match.group(1))
            unit = time_match.group(2)
            months = val * 12 if "y" in unit else val
            extracted["time_in_business_months"] = months
        elif "year" in u_lower or "month" in u_lower:
            for word in u_lower.split():
                if word.isdigit():
                    months = int(word) * 12 if "year" in u_lower else int(word)
                    extracted["time_in_business_months"] = months
                    break

        clean_text = u_lower.replace("$", " ").replace(",", "")
        for word in clean_text.split():
            if word.endswith("k") and word[:-1].isdigit():
                extracted["monthly_revenue"] = int(word[:-1]) * 1000
                break
            elif word.isdigit() and int(word) >= 1000 and int(word) not in [2023, 2024, 2025, 2026]:
                extracted["monthly_revenue"] = int(word)
                break

        # Qualitative credit score statements
        if any(k in u_lower for k in ["greater than requirement", "above requirement", "meets requirement", "higher than requirement", "exceeds requirement", "qualif"]):
            extracted["credit_score"] = 720
        elif any(k in u_lower for k in ["excellent", "perfect credit"]):
            extracted["credit_score"] = 780
        elif any(k in u_lower for k in ["good credit", "great credit"]):
            extracted["credit_score"] = 720
        elif any(k in u_lower for k in ["fair credit", "average credit"]):
            extracted["credit_score"] = 650
        elif any(k in u_lower for k in ["poor credit", "bad credit"]):
            extracted["credit_score"] = 550

        # Numeric credit score matching (300 to 999)
        if "credit_score" not in extracted:
            for word in clean_text.split():
                if word.isdigit() and 300 <= int(word) <= 999 and word != str(extracted.get("monthly_revenue")):
                    extracted["credit_score"] = int(word)
                    break

        us_states = ["new york", "california", "texas", "florida", "illinois", "ohio", "georgia", "washington", "arizona", "colorado", "pennsylvania", "north carolina", "michigan", "new jersey", "virginia"]
        for state in us_states:
            if re.search(rf"\b{state}\b", u_lower):
                extracted["location"] = state.title()
                break
        if "location" not in extracted:
            if re.search(r"\b(usa|united states|us)\b", u_lower):
                extracted["location"] = "United States"

        # Calculate updated missing fields
        current_qual = dict(qualification_state or {})
        current_qual.update(extracted)
        
        all_required = ["time_in_business_months", "monthly_revenue", "credit_score", "location"]
        remaining_missing = [f for f in all_required if current_qual.get(f) is None]

        # Intent detection
        intent = "providing_info"
        escalate = False
        if "person" in u_lower or "human" in u_lower or "representative" in u_lower:
            intent = "escalate"
            escalate = True
            resp = "I understand. I'll connect you with a loan specialist right away."
        elif "weather" in u_lower or "joke" in u_lower:
            intent = "out_of_scope"
            resp = "I can only assist with business loan qualifications. Would you like to continue your loan assessment?"
        elif "why" in u_lower or ("credit" in u_lower and "score" not in extracted):
            intent = "objection"
            resp = "We check credit score to ensure preliminary eligibility (minimum 650). What is your estimated score?"
        else:
            # Ask ONLY for the next missing field in order!
            if "time_in_business_months" in remaining_missing:
                resp = "Thanks for the information! How long has your business been operating?"
            elif "monthly_revenue" in remaining_missing:
                resp = "Got it. What is your average monthly revenue?"
            elif "credit_score" in remaining_missing:
                resp = "Great. What is your estimated personal credit score (minimum 650)?"
            elif "location" in remaining_missing:
                resp = "Understood. Which US state is your business located in?"
            else:
                resp = "Thank you! You have provided all required information and meet our basic qualification criteria."

        return {
            "response_text": resp,
            "extracted_fields": extracted,
            "intent": intent,
            "escalate": escalate
        }
