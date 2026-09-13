import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.routing import should_escalate


def test_low_confidence_escalates():
    yes, reason = should_escalate("help", 0.2, [{"score": 0.8}], 0.62, [])
    assert yes and "confidence" in reason.lower()


def test_strong_safe_case_auto_handles():
    yes, _ = should_escalate("wifi is not working", 0.9, [{"score": 0.8}], 0.62, [])
    assert not yes


def test_high_risk_escalates():
    yes, reason = should_escalate("my account was hacked", 0.99, [{"score": 0.9}], 0.62, [r"account\s+(was\s+)?hacked"])
    assert yes and "high-risk" in reason.lower()
