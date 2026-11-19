#!/usr/bin/env python3
"""Cross-family validation: re-run the is_safety gate and the category assignment
with a different-family open model (Qwen2.5-7B-Instruct) and write its decisions,
so we can report agreement with the Claude Sonnet 4.6 pipeline.

  module load python/gpu/3.11.5 ; python pipeline/04_validate/qwen_crossfamily_validate.py
"""
import io, json, os, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PS = "paper_survey/_census"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
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

def gen(model, tok, sys, user, n=8):
    msgs=[{"role":"system","content":sys},{"role":"user","content":user}]
    text=tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp=tok(text, return_tensors="pt").to(model.device)
    out=model.generate(**inp, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    # --- gate ---
    gate=json.load(io.open(f"{PS}/_gate_val_sample.json",encoding="utf-8"))
    gp={}
    for i,p in enumerate(gate):
        u=f"TITLE: {p['title']}\nABSTRACT: {p['abstract'] or '(none)'}"
        r=gen(model,tok,GATE_SYS,u,4).upper()
        gp[p["id"]]= ("YES" in r and "NO" not in r[:4])
        if (i+1)%50==0: print(f"  gate {i+1}/{len(gate)}", flush=True)
    json.dump(gp, io.open(f"{PS}/_gate_val_qwen.json","w"), indent=0)
    # --- category ---
    cat=json.load(io.open(f"{PS}/_cat_val_sample.json",encoding="utf-8"))
    def match(r):
        r=r.lower()
        best=None
        for c in CATS:
            if c.lower() in r or c.split(" & ")[0].lower() in r: best=c; break
        return best or "UNMATCHED"
    cp={}
    for i,p in enumerate(cat):
        u=f"TITLE: {p['title']}\nABSTRACT: {p['abstract'] or '(none)'}"
        cp[p["id"]]=match(gen(model,tok,CAT_SYS,u,16))
        if (i+1)%50==0: print(f"  cat {i+1}/{len(cat)}", flush=True)
    json.dump(cp, io.open(f"{PS}/_cat_val_qwen.json","w"), indent=0)
    print("wrote _gate_val_qwen.json and _cat_val_qwen.json")

if __name__=="__main__":
    main()
