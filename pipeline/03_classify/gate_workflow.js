export const meta = {
  name: 'census-is-safety-gate',
  description: 'In-agent is_safety gate over census batch prompts; each agent reads a batch_NNN.txt, classifies, writes batch_NNN.json',
  phases: [{ title: 'Gate' }],
}

// Config is hardcoded below because `args` does not thread reliably through the
// scriptPath launch path. Override via args.{promptDir,outDir,batches,n} if set.
const A = (typeof args === 'object' && args) ? args : {}
const promptDir = A.promptDir || 'paper_survey/_census/prompts'
const outDir = A.outDir || 'paper_survey/_census/gate_out'
const N = A.n || 675
// RETRY holds the still-missing batch indices for a re-run (rewritten each round
// from disk). Empty => process the full [0,N) range.
const RETRY = [619, 624, 627, 629, 646, 649, 650, 651, 652, 653, 654, 655, 657, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674]
const batches = A.batches || (RETRY.length ? RETRY : Array.from({ length: N }, (_, i) => i))

log(`gate: ${batches.length} batches -> ${outDir}`)

const pad = (n) => String(n).padStart(3, '0')

const results = await parallel(batches.map((b) => async () => {
  const tag = pad(b)
  const promptFile = `${promptDir}/batch_${tag}.txt`
  const outFile = `${outDir}/batch_${tag}.json`
  const status = await agent(
    `Read the file ${promptFile}. It contains a complete AI-safety paper classification task: ` +
    `instructions, the list of allowed category names with definitions, and a list of papers each ` +
    `with an [id], TITLE and ABSTRACT.\n\n` +
    `Follow those instructions exactly. For EVERY paper decide: is_safety (boolean), primary (one ` +
    `category string copied EXACTLY from the allowed list), secondary (an allowed category string or ` +
    `null), confidence (0-1 number).\n\n` +
    `Then use the Write tool to write to EXACTLY this path: ${outFile}\n` +
    `The file content must be ONLY a JSON object of this shape, one entry per paper, ids matching the ` +
    `[id] in the prompt:\n` +
    `{"results":[{"id":0,"is_safety":true,"primary":"Fairness & Bias","secondary":null,"confidence":0.9}]}\n\n` +
    `Use only category names that appear in the prompt file. After writing the file, reply with just ` +
    `the word OK and the count of results written. Do not print the JSON in your reply.`,
    { label: `gate:${tag}`, phase: 'Gate' }
  )
  return { batch: b, status: status ? String(status).slice(0, 40) : null }
}))

const ok = results.filter((r) => r && r.status).length
log(`gate done: ${ok}/${batches.length} agents reported`)
return { requested: batches.length, reported: ok }
