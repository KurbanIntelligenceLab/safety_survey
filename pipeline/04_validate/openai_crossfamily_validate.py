#!/usr/bin/env python3
"""Cross-family validation with a peer-capability non-Claude model (OpenAI GPT).

Re-runs the is_safety GATE (300-paper accept/reject sample) and the CATEGORY
assignment (140-paper sample) with an OpenAI model, mirroring the prompts used
for the Qwen run so the agreement numbers are directly comparable. Only paper
titles + abstracts (published metadata) are sent to the API.

  export OPENAI_API_KEY=$(cat .openai_key)
  python pipeline/04_validate/openai_crossfamily_validate.py [model]
"""
import io, json, os, sys, time
import requests

PS = "paper_survey/_census"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
KEY = os.environ.get("OPENAI_API_KEY") or open(".openai_key").read().strip()
URL = "https://api.openai.com/v1/chat/completions"
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

CATS = ["Adversarial Robustness","Backdoor & Data Poisoning","Fairness & Bias",
 "Interpretability & Explainability","Privacy & Data Protection","Out-of-Distribution & Uncertainty",
 "Safe Reinforcement Learning","Alignment & RLHF","Deepfakes & Media Integrity",
 "Hallucination & Factuality","Toxicity & Content Safety","Watermarking & IP Protection",
 "AI Governance & Policy","Agentic & Frontier-Model Safety"]

GATE_SYS = ("You decide whether a machine-learning paper's PRIMARY contribution is AI safety "
 "research (adversarial robustness, alignment, fairness, interpretability, privacy, OOD/uncertainty, "
 "safe RL, hallucination, toxicity, backdoors, deepfakes, watermarking, governance, agentic safety). "
 "A paper that only uses or mentions these in passing is NOT safety. Answer with exactly one word: YES or NO.")
CAT_SYS = ("Assign the machine-learning paper to the single best-fit AI-safety category from this list:\n"
 + "\n".join(f"- {c}" for c in CATS) + "\nReply with exactly one category name from the list.")

def chat(sys_p, user, n=8):
    body = {"model": MODEL, "temperature": 0, "max_tokens": n,
            "messages": [{"role":"system","content":sys_p},{"role":"user","content":user}]}
    for attempt in range(6):
        r = requests.post(URL, headers=HDR, json=body, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 * (attempt + 1)); continue
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")
    raise RuntimeError("retries exhausted")

def match(r):
    r = r.lower()
    for c in CATS:
        if c.lower() in r or c.split(" & ")[0].lower() in r:
            return c
    return "UNMATCHED"

def main():
    print(f"model={MODEL}", flush=True)
    gate = json.load(io.open(f"{PS}/_gate_val_sample.json", encoding="utf-8"))
    gp = {}
    for i, p in enumerate(gate):
        u = f"TITLE: {p['title']}\nABSTRACT: {p['abstract'] or '(none)'}"
        r = chat(GATE_SYS, u, 4).upper()
        gp[p["id"]] = ("YES" in r and "NO" not in r[:4])
        if (i+1) % 50 == 0: print(f"  gate {i+1}/{len(gate)}", flush=True)
    json.dump(gp, io.open(f"{PS}/_gate_val_gpt.json", "w"), indent=0)
    cat = json.load(io.open(f"{PS}/_cat_val_sample.json", encoding="utf-8"))
    cp = {}
    for i, p in enumerate(cat):
        u = f"TITLE: {p['title']}\nABSTRACT: {p['abstract'] or '(none)'}"
        cp[p["id"]] = match(chat(CAT_SYS, u, 16))
        if (i+1) % 50 == 0: print(f"  cat {i+1}/{len(cat)}", flush=True)
    json.dump(cp, io.open(f"{PS}/_cat_val_gpt.json", "w"), indent=0)
    print(f"wrote _gate_val_gpt.json ({len(gp)}) and _cat_val_gpt.json ({len(cp)})")

if __name__ == "__main__":
    main()
