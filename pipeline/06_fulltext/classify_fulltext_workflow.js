export const meta = {
  name: 'census-fulltext-classify',
  description: 'Full-text re-classification of census safety papers; each agent reads a reclass batch_NNN.txt (TITLE/ABSTRACT/METHOD/CONCLUSION) and writes batch_NNN.json with canonical 16-cat labels',
  phases: [{ title: 'Classify' }],
}

// Config hardcoded (args does not thread through the scriptPath launch path).
const A = (typeof args === 'object' && args) ? args : {}
const promptDir = A.promptDir || 'paper_survey/_census/_fulltext/reclass_full_batches'
const outDir = A.outDir || 'paper_survey/_census/_fulltext/reclass_full_out'
const N = A.n || 179
// RETRY: still-missing batch indices for a re-run (rewritten each round from disk). Empty => full [0,N).
const RETRY = [177, 178]
const batches = A.batches || (RETRY.length ? RETRY : Array.from({ length: N }, (_, i) => i))

log(`classify: ${batches.length} batches -> ${outDir}`)

const pad = (n) => String(n).padStart(3, '0')

const results = await parallel(batches.map((b) => async () => {
  const tag = pad(b)
  const promptFile = `${promptDir}/batch_${tag}.txt`
  const outFile = `${outDir}/batch_${tag}.json`
  const status = await agent(
    `Read the file ${promptFile}. It contains a complete AI-safety paper classification task: ` +
    `instructions, the list of allowed category names with definitions, and a list of papers each ` +
    `with an [id], TITLE, ABSTRACT, and (when available) METHOD and CONCLUSION excerpts.\n\n` +
    `Follow those instructions exactly. For EVERY paper decide: is_safety (boolean), primary (one ` +
    `category string copied EXACTLY from the allowed list), secondary (an allowed category string or ` +
    `null), confidence (0-1 number). Use the METHOD/CONCLUSION text when present to classify more ` +
    `accurately than title+abstract alone.\n\n` +
    `Then use the Write tool to write to EXACTLY this path: ${outFile}\n` +
    `The file content must be ONLY a JSON object of this shape, one entry per paper, ids matching the ` +
    `[id] in the prompt:\n` +
    `{"results":[{"id":0,"is_safety":true,"primary":"Fairness & Bias","secondary":null,"confidence":0.9}]}\n\n` +
    `Use only category names that appear in the prompt file. After writing the file, reply with just ` +
    `the word OK and the count of results written. Do not print the JSON in your reply.`,
    { label: `classify:${tag}`, phase: 'Classify' }
  )
  return { batch: b, status: status ? String(status).slice(0, 40) : null }
}))

const ok = results.filter((r) => r && r.status).length
log(`classify done: ${ok}/${batches.length} agents reported`)
return { requested: batches.length, reported: ok }
