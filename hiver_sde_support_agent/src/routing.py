import re


def should_escalate(text, confidence, evidence, threshold, patterns):
    t = text.lower()
    for p in patterns:
        if re.search(p.lower(), t):
            return True, f"Matched high-risk escalation trigger: '{p}'."
    if confidence < threshold:
        return True, f"Intent confidence {confidence:.2f} is below {threshold:.2f}."
    if not evidence or evidence[0]["score"] < 0.15:
        return True, "No sufficiently similar historical resolution was retrieved."
    if re.search(r"\b(refund|chargeback|cancel.*payment|stolen|hacked|fraud)\b", t):
        return True, "Case-specific financial/account action should be reviewed by a human."
    return False, "Confidence and historical evidence are sufficient and no escalation trigger fired."
