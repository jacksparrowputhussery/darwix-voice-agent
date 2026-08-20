import pytest
from voice_agent.flows.conversation import VoiceSessionManager

@pytest.mark.asyncio
async def test_session_conversation_history_is_list_of_dicts():
    session = VoiceSessionManager("test_history_session")
    
    # 1. Verify initial history is empty list
    assert isinstance(session.state["conversation_history"], list)
    assert len(session.state["conversation_history"]) == 0
    
    # 2. Turn 1
    res1 = await session.process_text_turn("We have been in business for 3 years.", skip_tts=True)
    hist1 = session.state["conversation_history"]
    assert isinstance(hist1, list)
    assert len(hist1) == 2
    assert all(isinstance(entry, dict) for entry in hist1)
    assert hist1[0]["role"] == "user"
    assert hist1[0]["speaker"] == "customer"
    assert hist1[1]["role"] == "assistant"
    assert hist1[1]["speaker"] == "agent"
    assert session.state["qualification"]["time_in_business_months"] == 36
    
    # Turn 1 should ask for revenue, NOT re-ask time in business
    assert "time" not in res1["response_text"].lower() or "month" in res1["response_text"].lower() or "revenue" in res1["response_text"].lower()

    # 3. Turn 2
    res2 = await session.process_text_turn("Our monthly revenue is $20,000.", skip_tts=True)
    hist2 = session.state["conversation_history"]
    assert len(hist2) == 4
    assert all(isinstance(entry, dict) for entry in hist2)
    assert session.state["qualification"]["monthly_revenue"] == 20000

    # 4. Turn 3
    res3 = await session.process_text_turn("Credit score is 700 and located in Texas.", skip_tts=True)
    hist3 = session.state["conversation_history"]
    assert len(hist3) == 6
    assert all(isinstance(entry, dict) for entry in hist3)
    assert session.state["qualification"]["credit_score"] == 700
    assert "Texas" in session.state["qualification"]["location"]
    assert res3["is_qualified"] is True

    # 5. Verify reset_session
    session.reset_session()
    assert isinstance(session.state["conversation_history"], list)
    assert len(session.state["conversation_history"]) == 0
    assert session.state["qualification"] == {}
