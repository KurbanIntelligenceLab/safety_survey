export const meta = {
  name: 'layer1-deepcode',
  description: 'Layer-1 ground-truth deep coding: carefully code each sampled paper against its subfield open-problem labels + free-text novel gaps',
  phases: [{ title: 'DeepCode' }],
}

// Self-contained (no fs / no args round-trip): the batch list is regenerated from
// BATCHCOUNTS (filled in after build_layer1_sample.py prints the per-subfield batch
// counts). Tags match build_layer1_sample's slug + index scheme. Each agent reads
// one small batch prompt file (full closing text, ~5 papers) and WRITES its result
// to layer1/<tag>.result.json; merge happens afterward via merge_layer1.py.
const ROOT = 'paper_survey/_fulltext12987'
const OUTDIR = `${ROOT}/layer1`

// __BATCHCOUNTS__  (subfield -> number of batch files); populated by the launcher.
const BATCHCOUNTS = {
  'AI Governance & Policy': 10,
  'Adversarial Robustness': 10,
  'Agentic & Frontier-Model Safety': 10,
  'Alignment & RLHF': 10,
  'Backdoor & Data Poisoning': 10,
  'Deepfakes & Media Integrity': 10,
  'Fairness & Bias': 10,
  'Hallucination & Factuality': 10,
  'Interpretability & Explainability': 10,
  'Out-of-Distribution & Uncertainty': 10,
  'Privacy & Data Protection': 10,
  'Safe Reinforcement Learning': 10,
  'Toxicity & Content Safety': 10,
  'Watermarking & IP Protection': 10,
}

const slug = (s) => s.replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
const pad3 = (n) => String(n).padStart(3, '0')

const batches = []
for (const sub of Object.keys(BATCHCOUNTS)) {
  for (let i = 0; i < BATCHCOUNTS[sub]; i++) batches.push([`${slug(sub)}_${pad3(i)}`, sub])
}

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['results'],
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'labels', 'novel_gaps', 'off_topic'],
        properties: {
          id: { type: 'integer' },
          labels: { type: 'array', items: { type: 'string' } },
          novel_gaps: { type: 'array', items: { type: 'string' } },
          off_topic: { type: 'boolean' },
        },
      },
    },
  },
}

log(`deep-code: ${batches.length} batches`)

const out = await parallel(batches.map(([tag, subfield]) => async () => {
  const file = `${OUTDIR}/${tag}.txt`
  const resultFile = `${OUTDIR}/${tag}.result.json`
  const r = await agent(
    `Read the file ${file}. It is a careful AI-safety open-problem DEEP-CODING ` +
    `task (this is ground-truth annotation, so be precise and thorough). It lists ` +
    `one subfield's allowed open-problem labels (id: meaning) and a few papers, ` +
    `each with an [id], TITLE, and FULL CLOSING TEXT. For EACH paper, read the ` +
    `closing text closely and assign every label whose open problem the authors ` +
    `THEMSELVES state as open / unresolved / future work — code only what they ` +
    `genuinely raise as open, not what they solve or merely cite. Also list, in ` +
    `novel_gaps, any substantive open problem the authors raise that none of the ` +
    `labels capture (short phrase each; empty if none). Use only label ids that ` +
    `appear in the file.\n\n` +
    `Then use the Write tool to save your answer to the file ${resultFile}. ` +
    `Its contents must be EXACTLY a JSON object of the form ` +
    `{"results":[{"id":<int>,"labels":[<label-id strings>],` +
    `"novel_gaps":[<short strings>],"off_topic":<bool>}, ...]} with one entry per ` +
    `paper. Finally, return that same JSON object as your output.`,
    { label: tag, phase: 'DeepCode', schema: SCHEMA }
  )
  return { tag, subfield, n: (r && r.results || []).length }
}))

const ok = out.filter(Boolean)
const npapers = ok.reduce((a, b) => a + b.n, 0)
log(`deep-coded ${npapers} papers across ${ok.length}/${batches.length} batches`)
return { batches: ok.length, papers: npapers }
