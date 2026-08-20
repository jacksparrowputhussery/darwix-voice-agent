import pytest
import tempfile
from pathlib import Path
from voice_agent.flows.rules import BusinessLoanRuleEngine

def test_dynamic_policy_bullet_parsing():
    custom_policy = """
BUSINESS LOAN POLICY
1. Eligibility Requirements
- The business must have been in operation for a minimum of 36 months.
- The business must generate a minimum of $25,000 in monthly revenue.
- The business owner must have a personal credit score of 720 or higher.
- The business must be located within the United States.
- The business must employ at least 5 full-time employees.
"""
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as f:
        f.write(custom_policy)
        temp_path = Path(f.name)

    try:
        reqs = BusinessLoanRuleEngine.load_requirements_from_policy(temp_path)
        assert len(reqs) == 5
        
        keys = [r["key"] for r in reqs]
        assert "time_in_business_months" in keys
        assert "monthly_revenue" in keys
        assert "credit_score" in keys
        assert "location" in keys
        assert any("employee" in k for k in keys)

        # Verify parsed thresholds
        tib = next(r for r in reqs if r["key"] == "time_in_business_months")
        assert tib["min_value"] == 36
        assert tib["req"] == "≥ 36 Months"

        rev = next(r for r in reqs if r["key"] == "monthly_revenue")
        assert rev["min_value"] == 25000.0
        assert "$25,000" in rev["req"]

        cs = next(r for r in reqs if r["key"] == "credit_score")
        assert cs["min_value"] == 720
        assert "720" in cs["req"]

    finally:
        if temp_path.exists():
            temp_path.unlink()
