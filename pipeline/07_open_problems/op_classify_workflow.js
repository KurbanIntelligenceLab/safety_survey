export const meta = {
  name: 'op-census-classify',
  description: 'Layer-2 open-problem census: classify each closing-section batch against its subfield label set',
  phases: [{ title: 'Classify' }],
}

// Workflow scripts have NO filesystem access and args did not round-trip, so the
// batch list is regenerated here from the subfield -> usable-paper counts printed
// by build_op_batches.py (BATCH=25). Tags match build_op_batches' slug + index
// scheme exactly: slug = non-alnum -> "_", trimmed; tag = `${slug}_${NNN}`.
// Each agent reads its batch prompt file (Read) and WRITES its result to
// op_batches/<tag>.result.json (Write); merge_op_results.py stitches them after.
const ROOT = 'paper_survey/_fulltext12987'
const OUTDIR = `${ROOT}/op_batches`
const BATCH = 25

const COUNTS = {
  'AI Governance & Policy': 248,
  'Adversarial Robustness': 1596,
  'Agentic & Frontier-Model Safety': 108,
  'Alignment & RLHF': 362,
  'Backdoor & Data Poisoning': 141,
  'Deepfakes & Media Integrity': 340,
  'Fairness & Bias': 961,
  'Hallucination & Factuality': 865,
  'Interpretability & Explainability': 678,
  'Out-of-Distribution & Uncertainty': 818,
  'Privacy & Data Protection': 840,
  'Safe Reinforcement Learning': 333,
  'Toxicity & Content Safety': 491,
  'Watermarking & IP Protection': 235,
}

const slug = (s) => s.replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '')
const pad3 = (n) => String(n).padStart(3, '0')

const batches = []
for (const sub of Object.keys(COUNTS)) {
  const nb = Math.ceil(COUNTS[sub] / BATCH)
  for (let i = 0; i < nb; i++) batches.push([`${slug(sub)}_${pad3(i)}`, sub])
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
        required: ['id', 'labels', 'off_topic', 'other'],
        properties: {
          id: { type: 'integer' },
          labels: { type: 'array', items: { type: 'string' } },
          off_topic: { type: 'boolean' },
          other: { type: 'boolean' },
        },
      },
    },
  },
}

log(`classify: ${batches.length} batches`)

const out = await parallel(batches.map(([tag, subfield]) => async () => {
  const file = `${OUTDIR}/${tag}.txt`
  const resultFile = `${OUTDIR}/${tag}.result.json`
  const r = await agent(
    `Read the file ${file}. It is a self-contained AI-safety open-problem ` +
    `classification task: it lists one subfield's allowed open-problem labels ` +
    `(id: meaning) and a set of papers, each with an [id], TITLE, and CLOSING ` +
    `TEXT. Follow its instructions exactly. For EVERY paper, assign the label ids ` +
    `whose open problem the paper itself states as open/unresolved/future work ` +
    `(multi-label; may be empty), and set off_topic and other as instructed. Use ` +
    `only label ids that appear in the file.\n\n` +
    `Then use the Write tool to save your answer to the file ${resultFile}. ` +
    `Its contents must be EXACTLY a JSON object of the form ` +
    `{"results":[{"id":<int>,"labels":[<label-id strings>],"off_topic":<bool>,` +
    `"other":<bool>}, ...]} with one entry per paper. Finally, return that same ` +
    `JSON object as your output.`,
    { label: tag, phase: 'Classify', schema: SCHEMA }
  )
  return { tag, subfield, n: (r && r.results || []).length }
}))

const ok = out.filter(Boolean)
const npapers = ok.reduce((a, b) => a + b.n, 0)
log(`classified ${npapers} papers across ${ok.length}/${batches.length} batches`)
return { batches: ok.length, papers: npapers }
