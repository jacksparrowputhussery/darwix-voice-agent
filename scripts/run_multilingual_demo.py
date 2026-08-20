import os
import sys
import asyncio
import json
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from voice_agent.flows.conversation import VoiceSessionManager
from voice_agent.tools.tts import TTSEngine
from multilingual.prompts.philippines_prompt import SYSTEM_PROMPT_PH
from multilingual.prompts.indonesia_prompt import SYSTEM_PROMPT_ID

async def generate_customer_audio(text: str, filename: str, lang: str) -> str:
    tts = TTSEngine()
    if lang == "fil":
        tts.model = "fil-PH-AngeloNeural"
    elif lang == "id":
        tts.model = "id-ID-ArdiNeural"
    return await tts.generate_audio(text, filename)

async def run_market_scenario(market: str, scenario_name: str, turns: list[str], prompt: str, tts_model: str):
    print(f"\n--- Running {market} Scenario: {scenario_name} ---")
    output_dir = Path(f"./outputs/q3/{market}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    session = VoiceSessionManager(session_id=f"{market}_{scenario_name}")
    session.tts.model = tts_model
    transcript = []
    
    # Store original function outside loop
    original_gen = session.llm.generate_response
    
    for i, user_text in enumerate(turns):
        print(f"\nTurn {i+1}")
        
        audio_path = str(output_dir / f"{scenario_name}_turn_{i}_customer.mp3")
        lang = "fil" if market == "philippines" else "id"
        await generate_customer_audio(user_text, audio_path, lang)
        
        session.stt.transcribe = lambda path, text=user_text: text
        
        def patched_gen(system_prompt=None, context="", user_input="", conversation_history=None, **kwargs):
            return original_gen(prompt, context, user_input, conversation_history, **kwargs)
        session.llm.generate_response = patched_gen
        
        result = await session.process_turn(audio_path)
        
        agent_audio = str(output_dir / f"{scenario_name}_turn_{i}_agent.mp3")
        os.replace(result["tts_audio_path"], agent_audio)
        
        agent_text = result["response_text"]
        
        print(f"Customer: {user_text}")
        print(f"Agent: {agent_text}")
        
        transcript.append({"speaker": "customer", "text": user_text, "audio": audio_path})
        transcript.append({"speaker": "agent", "text": agent_text, "audio": agent_audio})
        
        if session.state.get("escalation_requested") or session.state.get("human_required"):
            print("[ESCALATION TRIGGERED]")
            break
            
    transcript_file = output_dir / f"{scenario_name}_transcript.json"
    with open(transcript_file, "w") as f:
        json.dump({
            "scenario": scenario_name,
            "transcript": transcript
        }, f, indent=2)
    print(f"Saved transcript to {transcript_file}")

async def main():
    ph_scenarios = {
        "cooperative": [
            "Hi, magkano po ang premium kung kukuha ako ng life insurance?",
            "Ang beneficiary ko po sana ay ang asawa ko. Ano pong kailangan?"
        ],
        "objection_taglish": [
            "Ang mahal naman ng premium. Pwede bang babaan yung coverage para mas mura?",
            "What if mag-lapse yung policy ko dahil nakalimutan ko magbayad?"
        ]
    }
    
    id_scenarios = {
        "cooperative_colloquial": [
            "Halo, cicilan bulan ini berapa ya?",
            "Kalau mau bayar DP dulu bisa nggak?"
        ],
        "escalation_regional": [
            "Nuwun sewu, kulo mau tanya soal denda jatuh tempo.",
            "I want to talk to an agent please, I don't understand the tenor."
        ]
    }
    
    for name, turns in ph_scenarios.items():
        await run_market_scenario("philippines", name, turns, SYSTEM_PROMPT_PH, "fil-PH-BlessicaNeural")
        
    for name, turns in id_scenarios.items():
        await run_market_scenario("indonesia", name, turns, SYSTEM_PROMPT_ID, "id-ID-GadisNeural")
        
if __name__ == "__main__":
    asyncio.run(main())
