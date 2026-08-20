import os
import json
import pytest
from unittest.mock import MagicMock, patch
from voice_agent.tools.llm import LLMEngine, _clean_json_text

@pytest.fixture
def clean_env():
    """Ensure environment is controlled for tests."""
    with patch.dict(os.environ, {
        "GROQ_API_KEY": "gsk_test_groq_key",
        "GEMINI_API_KEY": "test_gemini_key",
        "LLM_PRIMARY_PROVIDER": "groq",
        "LLM_FALLBACK_PROVIDER": "gemini",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
        "GEMINI_MODEL": "gemini-2.5-flash-lite",
        "GROQ_TIMEOUT": "2.0"
    }):
        yield

def test_clean_json_text():
    # Test markdown stripping
    raw_with_fences = "```json\n{\"response_text\": \"hello\"}\n```"
    assert _clean_json_text(raw_with_fences) == "{\"response_text\": \"hello\"}"
    
    raw_plain = "{\"response_text\": \"hello\"}"
    assert _clean_json_text(raw_plain) == "{\"response_text\": \"hello\"}"

def test_groq_primary_success(clean_env):
    """Test that Groq is called first and returns valid JSON response."""
    with patch("voice_agent.tools.llm.Groq") as mock_groq_cls, \
         patch("voice_agent.tools.llm.genai") as mock_genai:
        
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        
        # Mock groq chat completion response
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "response_text": "Hello! How can I help you with your loan application?",
            "extracted_fields": {"monthly_revenue": 15000},
            "intent": "providing_info",
            "escalate": False
        })
        mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]
        
        engine = LLMEngine()
        assert engine.groq_client is not None
        
        response = engine.generate_response(
            system_prompt="You are a loan officer assistant.",
            context="Minimum revenue is $5000.",
            user_input="We make 15000 a month.",
            conversation_history=[]
        )
        
        assert response["response_text"] == "Hello! How can I help you with your loan application?"
        assert response["extracted_fields"]["monthly_revenue"] == 15000
        assert engine.last_metrics["provider"] == "groq"
        assert engine.last_metrics["fallback_used"] is False
        assert mock_groq_instance.chat.completions.create.called

def test_groq_failure_fallback_to_gemini(clean_env):
    """Test that when Groq raises an exception, Gemini fallback is triggered automatically."""
    with patch("voice_agent.tools.llm.Groq") as mock_groq_cls, \
         patch("voice_agent.tools.llm.genai") as mock_genai:
        
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        mock_groq_instance.chat.completions.create.side_effect = Exception("Groq rate limit 429: Too Many Requests")
        
        mock_gemini_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_gemini_model
        
        # Mock Gemini response
        mock_gemini_resp = MagicMock()
        mock_gemini_resp.text = json.dumps({
            "response_text": "Gemini fallback response: Got your details!",
            "extracted_fields": {"credit_score": 750},
            "intent": "providing_info",
            "escalate": False
        })
        mock_gemini_model.generate_content.return_value = mock_gemini_resp
        
        engine = LLMEngine()
        
        response = engine.generate_response(
            system_prompt="You are a loan assistant.",
            context="Credit score minimum 650.",
            user_input="My score is 750.",
            conversation_history=[]
        )
        
        assert response["response_text"] == "Gemini fallback response: Got your details!"
        assert response["extracted_fields"]["credit_score"] == 750
        assert engine.last_metrics["provider"] == "gemini"
        assert engine.last_metrics["fallback_used"] is True
        assert mock_gemini_model.generate_content.called

def test_all_providers_fail_offline_fallback(clean_env):
    """Test that when both Groq and Gemini fail, rule-based offline fallback is returned."""
    with patch("voice_agent.tools.llm.Groq") as mock_groq_cls, \
         patch("voice_agent.tools.llm.genai") as mock_genai:
        
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        mock_groq_instance.chat.completions.create.side_effect = Exception("Groq Network Down")
        
        mock_gemini_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_gemini_model
        mock_gemini_model.generate_content.side_effect = Exception("Gemini Quota Exceeded")
        
        engine = LLMEngine()
        
        response = engine.generate_response(
            system_prompt="You are a loan assistant.",
            context="",
            user_input="I want to speak with a human person please.",
            conversation_history=[]
        )
        
        assert response["intent"] == "escalate"
        assert response["escalate"] is True
        assert engine.last_metrics["provider"] == "offline_fallback"
        assert engine.last_metrics["fallback_used"] is True

def test_generate_json_signal_extractor(clean_env):
    """Test that generate_json works for real-time signal extraction."""
    with patch("voice_agent.tools.llm.Groq") as mock_groq_cls:
        mock_groq_instance = MagicMock()
        mock_groq_cls.return_value = mock_groq_instance
        
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "signals": [
                {
                    "signal_id": "sig_101",
                    "type": "missed_cross_sell",
                    "confidence": 0.95,
                    "evidence": "Customer mentioned equipment loan"
                }
            ]
        })
        mock_groq_instance.chat.completions.create.return_value.choices = [mock_choice]
        
        engine = LLMEngine()
        signals_data = engine.generate_json(
            system_prompt="Extract real-time call signals.",
            user_content="Customer: We need an equipment loan too."
        )
        
        assert len(signals_data.get("signals", [])) == 1
        assert signals_data["signals"][0]["type"] == "missed_cross_sell"
        assert engine.last_metrics["provider"] == "groq"
