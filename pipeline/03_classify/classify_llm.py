#!/usr/bin/env python3
"""
LLM classifier over (title, abstract). Two interchangeable execution routes,
ONE canonical prompt + schema so labels are identical either way:

  (A) External API  - set ANTHROPIC_API_KEY (or OPENAI_API_KEY); run this script
                      directly. Costs $ but is fully automated.
  (B) In-agent      - the workflow harness imports PROMPT_TEMPLATE / SCHEMA /
                      build_batch_prompt / parse_result from this module and
                      runs the same classification via subagents (no $ cost).

The classifier does two jobs per paper: (1) is_safety gate - exclude papers the
keyword search falsely matched; (2) assign 1-2 taxonomy categories (multi-label,
fixing the single-label distortion) with a primary label.

Usage (route A):
  ANTHROPIC_API_KEY=... python3 classify_llm.py \
      --in paper_survey/papers_enriched.json --out paper_survey/papers_llm.json
"""
import argparse, io, json, os, sys, time

CATEGORIES = [
    "Adversarial Robustness", "Backdoor & Data Poisoning", "Fairness & Bias",
    "Interpretability & Explainability", "Privacy & Data Protection",
    "Out-of-Distribution & Uncertainty", "Safe Reinforcement Learning",
    "Alignment & RLHF", "Deepfakes & Media Integrity", "Hallucination & Factuality",
    "Toxicity & Content Safety", "Verification & Formal Methods",
    "Watermarking & IP Protection", "AI Governance & Policy",
    "Agentic & Frontier-Model Safety", "Miscellaneous Safety",
]

DEFINITIONS = """\
- Adversarial Robustness: attacks/defenses on inputs (perturbations, patches), certified/empirical robustness.
- Backdoor & Data Poisoning: training-time attacks via poisoned data or hidden triggers.
- Fairness & Bias: group/individual fairness, bias measurement and mitigation, discrimination.
- Interpretability & Explainability: attribution, mechanistic interp, probing, concept-based explanations.
- Privacy & Data Protection: differential privacy, membership inference, unlearning, federated privacy, memorization.
- Out-of-Distribution & Uncertainty: OOD detection, calibration, selective prediction, distribution shift, uncertainty.
- Safe Reinforcement Learning: safety constraints/exploration in RL, reward hacking in RL control.
- Alignment & RLHF: aligning model goals/values, RLHF/DPO, honesty, sycophancy, oversight, refusal, safety fine-tuning.
- Deepfakes & Media Integrity: detection of synthetic/manipulated media, provenance.
- Hallucination & Factuality: factual errors/faithfulness in generation, grounding, fact verification.
- Toxicity & Content Safety: harmful content detection, jailbreaks, prompt injection, guardrails, moderation.
- Verification & Formal Methods: formal proofs/certificates of network properties (distinct from empirical adversarial defenses).
- Watermarking & IP Protection: model/content watermarking, fingerprinting, model stealing/extraction.
- AI Governance & Policy: governance, regulation, auditing frameworks, accountability, sociotechnical/policy.
- Agentic & Frontier-Model Safety: safety of LLM agents/tool use, dangerous-capability/autonomy evaluation, dual-use.
- Miscellaneous Safety: safety-relevant but not fitting the above.
"""

PROMPT_TEMPLATE = """\
You are classifying machine-learning papers for a survey of AI SAFETY research \
(2018-2026). For each paper decide:
1. is_safety: true only if the paper's PRIMARY contribution is about AI safety \
in the broad sense (robustness, alignment, fairness, interpretability, privacy, \
OOD/uncertainty, safe RL, hallucination, toxicity, backdoors, deepfakes, \
verification, watermarking, governance, agentic/frontier safety). A paper that \
merely uses or mentions these in passing is false.
2. primary: the single best-fit category.
3. secondary: a second category if the paper genuinely spans two (else null). \
Use this for cross-cutting work (e.g. "robust RLHF" = Alignment + Adversarial).

Categories and definitions:
{definitions}

Return ONLY the categories listed. Papers here were retrieved by keyword search, \
so some are false positives - be willing to set is_safety=false.

Classify these papers:
{papers}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "is_safety": {"type": "boolean"},
                    "primary": {"type": "string", "enum": CATEGORIES},
                    "secondary": {"type": ["string", "null"], "enum": CATEGORIES + [None]},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "is_safety", "primary", "secondary"],
            },
        }
    },
    "required": ["results"],
}

BATCH = 20


def build_batch_prompt(batch):
    """batch: list of dicts with id,title,abstract. Returns the prompt string."""
    lines = []
    for p in batch:
        abs = (p.get("abstract") or "")[:1200]
        lines.append(f"[{p['id']}] TITLE: {p['title']}\nABSTRACT: {abs or '(none)'}")
    return PROMPT_TEMPLATE.format(definitions=DEFINITIONS,
                                  papers="\n\n".join(lines))


def parse_result(obj, batch_by_id):
    """Apply a {'results':[...]} object onto the paper dicts. Returns updated list."""
    out = []
    for r in obj.get("results", []):
        p = batch_by_id.get(r["id"])
        if not p:
            continue
        p = dict(p)
        p["is_safety"] = r["is_safety"]
        p["category"] = r["primary"]
        p["category_secondary"] = r.get("secondary")
        p["llm_confidence"] = r.get("confidence")
        out.append(p)
    return out


# ---- Route A: external API -------------------------------------------------
def call_anthropic(prompt):
    import anthropic
    client = anthropic.Anthropic()
    tool = {"name": "classify", "description": "Return classifications",
            "input_schema": SCHEMA}
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=4096,
        tools=[tool], tool_choice={"type": "tool", "name": "classify"},
        messages=[{"role": "user", "content": prompt}])
    for block in msg.content:
        if block.type == "tool_use":
            return block.input
    return {"results": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Route A needs ANTHROPIC_API_KEY. For the no-cost in-agent route, "
                 "drive classification through the workflow harness using "
                 "build_batch_prompt()/SCHEMA/parse_result from this module.")

    papers = json.load(io.open(args.inp, encoding="utf-8"))
    if args.limit:
        papers = papers[: args.limit]
    for i, p in enumerate(papers):
        p["id"] = i
    labeled = []
    for i in range(0, len(papers), BATCH):
        batch = papers[i:i + BATCH]
        obj = call_anthropic(build_batch_prompt(batch))
        labeled += parse_result(obj, {p["id"]: p for p in batch})
        sys.stdout.write(f"\r  {min(i+BATCH,len(papers))}/{len(papers)}")
        sys.stdout.flush()
        time.sleep(0.5)
    print()
    keep = [p for p in labeled if p.get("is_safety")]
    json.dump(labeled, io.open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"classified={len(labeled)}  kept(is_safety)={len(keep)}  "
          f"dropped={len(labeled)-len(keep)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
