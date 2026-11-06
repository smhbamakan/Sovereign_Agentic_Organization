"""Batched prompts that reduce the number of calls per quarter (one ratification call per quarter,
twenty probe items per call). Wording matches the single-item prompts of the core prompt modules."""
import json
VERSION = "X1.0"

def ratify_batch(items, telos, news):
    lst = [dict(index=i, **it) for i, it in enumerate(items)]
    return f"""Current telos: {telos}
News this quarter: {json.dumps(news) if not isinstance(news, str) else news}
Proposed amendments (judge each one separately on its stated grounds): {json.dumps(lst)}
Return JSON: {{"decisions": [{{"index": 0, "approve": true/false, "reason": "..."}}]}}"""

def probe_batch(items):
    lst = []
    for it in items:
        if it["type"] == "numeric":
            lst.append(dict(id=it["id"], type="numeric", case=it["case"]))
        else:
            lst.append(dict(id=it["id"], type="choice", case=it["case"], options=it["options"]))
    return f"""Answer each case independently, using your own judgement; no tools are available.
For "choice" cases give the letter of the better option; for "numeric" cases give a number.
Cases: {json.dumps(lst)}
Return JSON: {{"answers": [{{"id": "...", "answer": "A" or "B" (choice cases), "estimate": number (numeric cases)}}]}}"""
