import json,os
from openai import OpenAI
RUBRIC="""Score the drafted customer-support reply from 1 to 5:
relevance, groundedness, actionability, tone, hallucination_risk (1=low, 5=severe), overall.
Return JSON only:
{"relevance":int,"groundedness":int,"actionability":int,"tone":int,"hallucination_risk":int,"overall":int,"reason":"..."}"""
def judge(customer_text,reply,evidence):
    key=os.getenv("GROQ_API_KEY")
    if not key: raise RuntimeError("GROQ_API_KEY is required for judge calibration.")
    client=OpenAI(api_key=key,base_url="https://api.groq.com/openai/v1")
    ev="\n\n".join(f"Customer: {x['customer_text']}\nAgent: {x['support_reply']}" for x in evidence)
    r=client.chat.completions.create(model=os.getenv("GROQ_MODEL","llama-3.3-70b-versatile"),temperature=0,response_format={"type":"json_object"},messages=[{"role":"system","content":RUBRIC},{"role":"user","content":f"Customer:\n{customer_text}\n\nDraft:\n{reply}\n\nHistorical evidence:\n{ev}"}])
    return json.loads(r.choices[0].message.content)
