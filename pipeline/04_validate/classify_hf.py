#!/usr/bin/env python3
"""
Route C: cross-FAMILY blind re-classification with a local open-weights model.

This re-annotates the IDENTICAL 250-paper validation sample that the Claude-Opus
check used (paper_survey/_val_opus/batch_*.txt are the canonical blind prompts
built by build_val_opus_input.py), but with a non-Claude model (default
Qwen2.5-14B-Instruct). The point is a genuinely independent annotator from a
different model family, so the reliability claim becomes "accurate-ish across
families", not just "self-consistent within Claude".

Output mirrors the Opus route exactly: paper_survey/_val_opus/out_<tag>/batch_XXX.json
each holding {"results": [{id,is_safety,primary,secondary,confidence}, ...]},
so validate_crossfamily.py / validate_opus.py can score it against _gold.json.

Usage (on a GPU node):
  HF_HOME=<hf cache dir> \
  python classify_hf.py --model Qwen/Qwen2.5-14B-Instruct --tag qwen
"""
import argparse, glob, io, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [str(p) for p in __import__("pathlib").Path(__file__).resolve().parents[1].iterdir() if p.is_dir()]  # every pipeline stage folder
from classify_llm import CATEGORIES  # canonical 16-category list

PS = os.path.normpath(os.path.join(HERE, "..", "..", "paper_survey"))
VAL = os.path.join(PS, "_val_opus")
CATSET = set(CATEGORIES)

JSON_INSTRUCTION = (
    "\n\nReturn ONLY a single JSON object, no prose and no markdown fences, of the form:\n"
    '{"results": [{"id": <int>, "is_safety": <true|false>, '
    '"primary": "<one category name>", "secondary": <"category name" or null>, '
    '"confidence": <0..1>}]}\n'
    "Include exactly one object per paper, echoing each paper's id. "
    "Use category names verbatim from the list above."
)


def extract_json(text):
    """Pull the {\"results\": [...]} object out of a model completion, tolerantly."""
    # strip code fences
    text = re.sub(r"```(?:json)?", "", text)
    # find the first '{' that starts a balanced object containing "results"
    start = text.find('{')
    while start != -1:
        depth, instr, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if instr:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    instr = False
            else:
                if c == '"':
                    instr = True
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        chunk = text[start:i + 1]
                        try:
                            obj = json.loads(chunk)
                            if isinstance(obj, dict) and "results" in obj:
                                return obj
                        except Exception:
                            pass
                        break
        start = text.find('{', start + 1)
    return None


def coerce(results):
    """Clean each result: map to canonical fields, drop out-of-taxonomy labels."""
    out = []
    for r in results or []:
        if not isinstance(r, dict) or "id" not in r:
            continue
        prim = r.get("primary")
        if prim not in CATSET:
            # tolerate minor variants by case-insensitive match
            m = {c.lower(): c for c in CATEGORIES}.get(str(prim).strip().lower())
            prim = m or prim
        sec = r.get("secondary")
        if sec is not None and sec not in CATSET:
            m = {c.lower(): c for c in CATEGORIES}.get(str(sec).strip().lower())
            sec = m if m else None
        out.append({
            "id": int(r["id"]),
            "is_safety": bool(r.get("is_safety", True)),
            "primary": prim,
            "secondary": sec,
            "confidence": r.get("confidence"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--tag", default="qwen")
    ap.add_argument("--val-dir", default=VAL)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outdir = os.path.join(args.val_dir, f"out_{args.tag}")
    os.makedirs(outdir, exist_ok=True)
    index = json.load(io.open(os.path.join(args.val_dir, "batch_index.json"), encoding="utf-8"))
    batch_files = sorted(index.keys())
    print(f"[{args.tag}] {len(batch_files)} batches, model={args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    t0 = time.time()
    n_papers = 0
    for fn in batch_files:
        prompt = io.open(os.path.join(args.val_dir, fn), encoding="utf-8").read()
        msgs = [
            {"role": "system", "content": "You are a precise scientific-paper classifier. You output only valid JSON."},
            {"role": "user", "content": prompt + JSON_INSTRUCTION},
        ]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        completion = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        obj = extract_json(completion)
        results = coerce(obj.get("results") if obj else None)
        want = set(index[fn])
        got = {r["id"] for r in results}
        status = "ok" if got == want else f"MISMATCH miss={sorted(want-got)} extra={sorted(got-want)}"
        outpath = os.path.join(outdir, fn.replace(".txt", ".json"))
        json.dump({"results": results}, io.open(outpath, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        n_papers += len(results)
        print(f"  {fn}: {len(results)}/{len(want)} {status}", flush=True)

    print(f"[{args.tag}] done: {n_papers} labels in {time.time()-t0:.0f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
