import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

# Default policy path
DEFAULT_POLICY_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "business_loan_policy.txt"


def parse_requirement_bullet(bullet: str) -> Dict[str, Any]:
    """
    Parses a single requirement bullet line from the policy .txt file
    into a structured requirement schema used by the rule engine and frontend scorecard.
    """
    b_lower = bullet.lower().strip()
    
    # 1. Operation / Time in Business
    if any(k in b_lower for k in ["operation", "operating", "time in business", "in business"]):
        months_match = re.search(r"(\d+)\s*(month|mo|year|yr)", b_lower)
        min_months = 12
        display = "≥ 12 Months"
        if months_match:
            val = int(months_match.group(1))
            unit = months_match.group(2)
            min_months = val * 12 if "y" in unit else val
            display = f"≥ {min_months} Months"
        return {
            "key": "time_in_business_months",
            "title": "Time in Business",
            "req": display,
            "icon": "⏱️",
            "type": "min_number",
            "min_value": min_months,
            "raw_text": bullet.strip("- ")
        }
        
    # 2. Monthly Revenue
    elif any(k in b_lower for k in ["revenue", "monthly revenue", "sales", "$"]):
        rev_match = re.search(r"\$?\s*([\d,]+)", bullet)
        min_rev = 5000.0
        display = "≥ $5,000 / mo"
        if rev_match:
            val = float(rev_match.group(1).replace(",", ""))
            if val > 0:
                min_rev = val
                display = f"≥ ${min_rev:,.0f} / mo"
        return {
            "key": "monthly_revenue",
            "title": "Monthly Revenue",
            "req": display,
            "icon": "💵",
            "type": "min_number",
            "min_value": min_rev,
            "raw_text": bullet.strip("- ")
        }
        
    # 3. Credit Score
    elif any(k in b_lower for k in ["credit score", "credit", "fico"]):
        score_match = re.search(r"(\d{3})", bullet)
        min_score = 650
        display = "≥ 650 FICO"
        if score_match:
            min_score = int(score_match.group(1))
            display = f"≥ {min_score} FICO"
        return {
            "key": "credit_score",
            "title": "Credit Score",
            "req": display,
            "icon": "📈",
            "type": "min_number",
            "min_value": min_score,
            "raw_text": bullet.strip("- ")
        }
        
    # 4. Location
    elif any(k in b_lower for k in ["located", "location", "united states", "country", "state"]):
        return {
            "key": "location",
            "title": "Business Location",
            "req": "United States",
            "icon": "📍",
            "type": "location",
            "raw_text": bullet.strip("- ")
        }
        
    # 5. Generic Custom Requirement (e.g. employees, bank accounts, licenses)
    else:
        cleaned = re.sub(r"^(the business must|applicants must|must have|must be|the business|must)\s*", "", bullet, flags=re.IGNORECASE).strip()
        title_words = cleaned.split()[:4]
        title = " ".join(title_words).title() if title_words else "Requirement"
        
        num_match = re.search(r"\b(\d+)\b", bullet)
        min_val = int(num_match.group(1)) if num_match else None
        key = re.sub(r"[^\w]+", "_", title.lower()).strip("_") or f"custom_req_{abs(hash(bullet)) % 1000}"
        
        icon = "📋"
        if "employee" in b_lower:
            icon = "👥"
            key = "employees"
            title = "Full-Time Employees"
            if min_val:
                req_display = f"≥ {min_val} Employees"
            else:
                req_display = bullet.strip("- ")
        elif "bank" in b_lower or "account" in b_lower:
            icon = "📄"
            key = "business_bank_account"
            title = "Business Bank Account"
            req_display = bullet.strip("- ")
        elif "tax" in b_lower or "return" in b_lower or "statement" in b_lower:
            icon = "📄"
            key = "tax_returns"
            title = "Tax Returns"
            req_display = bullet.strip("- ")
        elif "bankruptcy" in b_lower or "debt" in b_lower or "legal" in b_lower:
            icon = "⚖️"
            key = "no_bankruptcies"
            title = "Bankruptcy History"
            req_display = bullet.strip("- ")
        else:
            req_display = bullet.strip("- ")

        return {
            "key": key,
            "title": title,
            "req": req_display,
            "icon": icon,
            "type": "min_number" if min_val is not None else "string",
            "min_value": min_val,
            "raw_text": bullet.strip("- ")
        }


class BusinessLoanRuleEngine:
    """
    Dynamic Underwriting Rule Engine:
    Parses requirements directly from business_loan_policy.txt and evaluates qualification state dynamically.
    """
    
    @classmethod
    def load_requirements_from_policy(cls, policy_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Reads the policy .txt file and dynamically extracts all eligibility requirements."""
        path = policy_path or DEFAULT_POLICY_PATH
        if not path.exists():
            logger.warning(f"Policy file {path} not found. Falling back to standard 4 requirements.")
            return cls._default_requirements()
            
        try:
            content = path.read_text(encoding="utf-8")
            
            # Extract Eligibility section
            match = re.search(r"(?:1\.\s*)?Eligibility Requirements([\s\S]*?)(?=\n\s*\d+\.|\Z)", content, re.IGNORECASE)
            target_text = match.group(1) if match else content
            
            # Find bullet lines
            bullets = re.findall(r"^[ \t]*-[ \t]*(.+)$", target_text, re.MULTILINE)
            if not bullets:
                bullets = re.findall(r"^[ \t]*\*[ \t]*(.+)$", target_text, re.MULTILINE)
                
            if not bullets:
                return cls._default_requirements()
                
            requirements = []
            seen_keys = set()
            for b in bullets:
                b_clean = b.strip()
                if not b_clean:
                    continue
                req = parse_requirement_bullet(b_clean)
                # Avoid duplicate keys
                base_key = req["key"]
                counter = 2
                while req["key"] in seen_keys:
                    req["key"] = f"{base_key}_{counter}"
                    counter += 1
                seen_keys.add(req["key"])
                requirements.append(req)
                
            return requirements if requirements else cls._default_requirements()
            
        except Exception as e:
            logger.error(f"Error parsing policy file {path}: {e}")
            return cls._default_requirements()

    @classmethod
    def _default_requirements(cls) -> List[Dict[str, Any]]:
        """Fallback requirement definitions."""
        return [
            {"key": "time_in_business_months", "title": "Time in Business", "req": "≥ 12 Months", "icon": "⏱️", "type": "min_number", "min_value": 12},
            {"key": "monthly_revenue", "title": "Monthly Revenue", "req": "≥ $5,000 / mo", "icon": "💵", "type": "min_number", "min_value": 5000.0},
            {"key": "credit_score", "title": "Credit Score", "req": "≥ 650 FICO", "icon": "📈", "type": "min_number", "min_value": 650},
            {"key": "location", "title": "Business Location", "req": "United States", "icon": "📍", "type": "location"}
        ]

    @classmethod
    @property
    def REQUIRED_FIELDS(cls) -> List[str]:
        """Dynamically returns list of required field keys parsed from the policy file."""
        reqs = cls.load_requirements_from_policy()
        return [r["key"] for r in reqs]

    @classmethod
    def evaluate_field(cls, req: Dict[str, Any], val: Any) -> Tuple[str, str]:
        """
        Evaluates a single value against a requirement specification.
        Returns: (status, display_value)
        status in ["passed", "failed", "pending"]
        """
        if val is None or val == "" or str(val).strip() in ["—", "-", "none", "null"]:
            return "pending", "—"
            
        req_type = req.get("type", "string")
        
        # 1. Minimum numerical threshold
        if req_type == "min_number":
            min_val = req.get("min_value")
            val_str = str(val).strip().lower()
            
            # Check for qualitative statements that indicate meeting or exceeding requirements
            if any(k in val_str for k in ["greater", "above", "higher", "more than", "meet", "exceed", "excellent", "perfect", "good", "pass"]):
                num_match = re.search(r"(\d+(?:\.\d+)?)", val_str)
                if num_match:
                    num_val = float(num_match.group(1))
                    if min_val is not None and num_val >= min_val:
                        return "passed", f"≥ {int(num_val) if num_val.is_integer() else num_val}"
                return "passed", f"≥ {int(min_val) if min_val is not None else 650}"
                
            if any(k in val_str for k in ["below", "lower", "less than", "under", "poor", "bad", "fail"]):
                return "failed", "Below Criteria"

            try:
                # Sanitize number from string like "$35,000", "24 months", "720", "950"
                cleaned = re.sub(r"[^\d.]", "", str(val))
                if not cleaned:
                    return "pending", str(val)
                num_val = float(cleaned)
                
                # Format display
                if req["key"] == "monthly_revenue":
                    display_str = f"${num_val:,.0f}"
                elif req["key"] == "time_in_business_months":
                    display_str = f"{int(num_val)} mos"
                elif req["key"] == "credit_score":
                    display_str = f"{int(num_val)}"
                else:
                    display_str = f"{int(num_val) if num_val.is_integer() else num_val}"
                    
                if min_val is not None:
                    if num_val >= min_val:
                        return "passed", display_str
                    else:
                        return "failed", display_str
                return "passed", display_str
            except Exception:
                return "pending", str(val)
                
        # 2. Location requirement
        elif req_type == "location":
            val_str = str(val).strip()
            val_lower = val_str.lower()
            us_indicators = ["us", "united states", "usa", "california", "texas", "new york", "florida", "illinois", "ohio", "georgia", "nc", "sc", "wa", "or", "az", "co", "pa", "mi", "nj", "va", "ca", "tx", "ny", "fl"]
            if any(ind in val_lower for ind in us_indicators):
                return "passed", val_str.title()
            else:
                return "failed", val_str.title()
                
        # 3. String / Boolean custom requirement
        else:
            val_str = str(val).strip()
            if any(negative in val_str.lower() for negative in ["no", "false", "none", "0", "denied"]):
                return "failed", val_str
            return "passed", val_str

    @classmethod
    def evaluate(cls, state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate if the conversation state qualifies according to active dynamic requirements.
        Returns: (is_qualified, message)
        """
        qualification = state.get("qualification", {})
        requirements = cls.load_requirements_from_policy()
        
        missing = []
        failed = []
        
        for req in requirements:
            key = req["key"]
            val = qualification.get(key)
            status, _ = cls.evaluate_field(req, val)
            
            if status == "pending":
                missing.append(req.get("title", key))
            elif status == "failed":
                failed.append(f"{req.get('title', key)} (Requirement: {req.get('req', '')})")
                
        if missing:
            return False, f"Missing required information: {', '.join(missing)}"
            
        if failed:
            return False, f"Does not meet minimum requirements: {'; '.join(failed)}"
            
        return True, "Congratulations, the applicant meets all preliminary requirements for a business loan."

    @staticmethod
    def requires_escalation(state: Dict[str, Any]) -> bool:
        """Check if the conversation state requires human escalation."""
        if state.get("escalation_requested"):
            return True
        if len(state.get("objections", [])) >= 3:
            return True
        if state.get("human_required"):
            return True
        return False
