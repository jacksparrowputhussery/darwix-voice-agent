import time
from typing import Dict, Any, List

class NudgeEngine:
    def __init__(self):
        self.cooldown_seconds = 15
        self.min_confidence = 0.80
        self.max_active_nudges = 3
        self.active_nudges = []
        self.last_nudge_time = {}

    def process_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        new_nudges = []
        current_time = time.time()
        
        # Clean expired nudges (example 30s expiry)
        self.active_nudges = [n for n in self.active_nudges if current_time - n.get("created_at", current_time) < 30]
        
        for sig in signals:
            if sig.get("confidence", 0) < self.min_confidence:
                continue
                
            sig_type = sig.get("type")
            
            # Cooldown check
            if sig_type in self.last_nudge_time:
                if current_time - self.last_nudge_time[sig_type] < self.cooldown_seconds:
                    continue # Suppressed
                    
            nudge_text = self._generate_nudge_text(sig)
            if not nudge_text:
                continue
                
            nudge = {
                "nudge_id": sig.get("signal_id", "nudge_id"),
                "type": sig_type,
                "nudge": nudge_text,
                "reason": sig.get("evidence", ""),
                "priority": "high" if sig_type == "rising_frustration" else "medium",
                "created_at": current_time
            }
            
            self.active_nudges.append(nudge)
            new_nudges.append(nudge)
            self.last_nudge_time[sig_type] = current_time
            
        # Ensure we don't exceed max active
        if len(self.active_nudges) > self.max_active_nudges:
            self.active_nudges = self.active_nudges[-self.max_active_nudges:]
            
        return new_nudges

    def _generate_nudge_text(self, signal: Dict[str, Any]) -> str:
        sig_type = signal.get("type")
        if sig_type == "missed_cross_sell":
            return "Customer mentioned an additional need. Suggest a relevant cross-sell offer."
        elif sig_type == "compliance_gap":
            return "Required disclosure is missing. Remind the agent before proceeding."
        elif sig_type == "rising_frustration":
            return "Acknowledge the customer's concern before continuing."
        elif sig_type == "payment_difficulty":
            return "Offer an approved payment-support or callback path."
        return ""
