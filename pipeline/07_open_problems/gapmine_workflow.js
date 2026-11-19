export const meta = {
  name: 'gapmine',
  description: 'Cluster the deep-read free-text novel gaps per subfield; flag recurring open problems the ruler missed',
  phases: [{ title: 'Cluster' }],
}

const ROOT = 'paper_survey/_fulltext12987'
const DIR = `${ROOT}/gapmine`

// [file-slug, subfield] — slugs match the extractor (space->_, &->and, -->_)
const SUBS = [
  ['AI_Governance_and_Policy', 'AI Governance & Policy'],
  ['Adversarial_Robustness', 'Adversarial Robustness'],
  ['Agentic_and_Frontier_Model_Safety', 'Agentic & Frontier-Model Safety'],
  ['Alignment_and_RLHF', 'Alignment & RLHF'],
  ['Backdoor_and_Data_Poisoning', 'Backdoor & Data Poisoning'],
  ['Deepfakes_and_Media_Integrity', 'Deepfakes & Media Integrity'],
  ['Fairness_and_Bias', 'Fairness & Bias'],
  ['Hallucination_and_Factuality', 'Hallucination & Factuality'],
  ['Interpretability_and_Explainability', 'Interpretability & Explainability'],
  ['Out_of_Distribution_and_Uncertainty', 'Out-of-Distribution & Uncertainty'],
  ['Privacy_and_Data_Protection', 'Privacy & Data Protection'],
  ['Safe_Reinforcement_Learning', 'Safe Reinforcement Learning'],
  ['Toxicity_and_Content_Safety', 'Toxicity & Content Safety'],
  ['Watermarking_and_IP_Protection', 'Watermarking & IP Protection'],
]

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['candidates', 'verdict'],
  properties: {
    verdict: { type: 'string' },  // one-line: does a recurring missing theme exist?
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['name', 'approx_count', 'recurring', 'covered_by_existing', 'example'],
        properties: {
          name: { type: 'string' },
          approx_count: { type: 'integer' },        // how many of the listed gaps map here
          recurring: { type: 'boolean' },           // raised by >=3 distinct papers
          covered_by_existing: { type: 'boolean' }, // really already in the ruler?
          example: { type: 'string' },
        },
      },
    },
  },
}

log(`gapmine: ${SUBS.length} subfields`)

const out = await parallel(SUBS.map(([slug, subfield]) => async () => {
  const file = `${DIR}/${slug}.txt`
  const resultFile = `${DIR}/${slug}.result.json`
  const r = await agent(
    `Read the file ${file}. It lists, for one AI-safety subfield, the EXISTING ` +
    `ruler labels (open problems already covered) and a long list of FREE-TEXT ` +
    `open problems that papers raised but that fell OUTSIDE the ruler. Your job: ` +
    `cluster the free-text list into coherent themes, and decide for each theme ` +
    `whether it is (a) RECURRING — genuinely raised by 3 or more distinct papers, ` +
    `not a one-off niche problem — and (b) whether it is actually already covered ` +
    `by one of the existing ruler labels (set covered_by_existing accordingly). ` +
    `Only themes that are recurring AND not covered are real candidate additions ` +
    `to the ruler. Be strict: most of these gaps are hyper-specific one-offs that ` +
    `should NOT become a category. Give approx_count = how many listed gaps map to ` +
    `the theme, and one short example. In 'verdict', state in one line whether any ` +
    `recurring uncovered theme exists for this subfield and name it if so.\n\n` +
    `Then use the Write tool to save your answer as JSON to ${resultFile} in the ` +
    `form {"verdict":"...","candidates":[{"name":...,"approx_count":<int>,` +
    `"recurring":<bool>,"covered_by_existing":<bool>,"example":"..."}]}, and return ` +
    `the same object.`,
    { label: slug, phase: 'Cluster', schema: SCHEMA }
  )
  return { subfield, verdict: (r && r.verdict) || '', candidates: (r && r.candidates) || [] }
}))

const ok = out.filter(Boolean)
log(`clustered ${ok.length}/${SUBS.length} subfields`)
return { subfields: ok.length }
