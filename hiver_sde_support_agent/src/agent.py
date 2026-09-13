import json
import os

SYSTEM = """You are a customer-support drafting assistant for AppleSupport.

Use ONLY the historical evidence supplied by the application.
Never invent policies, refunds, eligibility, prices, links, timelines, account access,
or technical guarantees. Do not claim to have accessed the customer's account.
If evidence is insufficient, recommend human review.
Keep the answer concise, polite and practical.
Return JSON: {\"reply\":\"...\",\"grounding_note\":\"...\"}"""


def draft_reply(customer_text, intent, evidence):
    """Generate a grounded draft. Falls back safely when no API key is configured."""
    key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        return {
            "reply": evidence[0]["support_reply"] if evidence else "I’m sorry you’re having trouble. A support specialist should review this case.",
            "grounding_note": "No LLM key configured; strongest historical resolution returned.",
        }

    # Lazy import keeps the deterministic pipeline runnable without the SDK/API key.
    from openai import OpenAI

    if os.getenv("GROQ_API_KEY"):
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    else:
        client = OpenAI(api_key=key)

    evidence_text = "\n\n".join(
        f"[Example {i+1} | similarity={e['score']:.3f}]\nCustomer: {e['customer_text']}\nAgent: {e['support_reply']}"
        for i, e in enumerate(evidence)
    )
    prompt = (
        f"Customer message:\n{customer_text}\n\nPredicted intent:\n{intent}\n\n"
        f"Historical AppleSupport examples:\n{evidence_text}"
    )
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    )
    return json.loads(response.choices[0].message.content)
