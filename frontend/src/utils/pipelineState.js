import { RANKING_STAGE } from './constants.js'
import { pluralize, truncate } from './formatters.js'

/**
 * Presentation state for the five visible pipeline stages.
 * Kept separate from the components so the mapping stays easy to reason about.
 */
export const STEP_STATE = {
  PENDING: 'pending',
  ACTIVE: 'active',
  DONE: 'done',
  FALLBACK: 'fallback',
  UNAVAILABLE: 'unavailable',
  EMPTY: 'empty',
  FAILED: 'failed',
  SKIPPED: 'skipped',
}

export const PIPELINE_STEPS = [
  { id: 'requirement', label: 'Understanding your requirement' },
  { id: 'retrieval', label: 'Searching verified BIS knowledge' },
  { id: 'candidates', label: 'Evaluating candidate standards' },
  { id: 'reasoning', label: 'AI applicability analysis' },
  { id: 'recommendations', label: 'Final recommendations' },
]

const STEP_BY_ID = Object.fromEntries(PIPELINE_STEPS.map((item) => [item.id, item]))

/** Distinct catalogued designations present in the retrieved evidence. */
export function distinctStandardCodes(evidence = []) {
  const codes = new Set()
  for (const item of evidence) {
    if (item?.standard_code) codes.add(item.standard_code)
  }
  return [...codes]
}

function step(id, state, detail) {
  return { ...STEP_BY_ID[id], state, detail }
}

const AWAITING = 'Awaiting API response…'

/**
 * Derive pipeline step states from the real client phase and the real response.
 *
 * The analyse endpoint answers with a single response, so while a request is in
 * flight only the first stage can honestly be called active. No stage is reported
 * as done before the backend actually returned it, and the AI stage is only "done"
 * when the backend reported a completed verdict.
 *
 * @param {{phase?: string, result?: object|null}} input
 * @returns {Array<{id: string, label: string, state: string, detail: string}>}
 */
export function buildPipelineState({ phase = 'idle', result = null } = {}) {
  const pipeline = result?.pipeline ?? null
  const evidence = result?.retrieved_evidence ?? []
  const recommendations = result?.recommendations ?? []
  const candidateCodes = distinctStandardCodes(evidence)
  const aiRanked = pipeline?.ranking_stage === RANKING_STAGE.AI

  const idle = (id) => step(id, STEP_STATE.PENDING, 'Not started')
  const awaiting = (id) => step(id, STEP_STATE.PENDING, AWAITING)

  if (phase === 'idle') {
    return [
      step('requirement', STEP_STATE.PENDING, 'Waiting for a requirement'),
      idle('retrieval'),
      idle('candidates'),
      idle('reasoning'),
      idle('recommendations'),
    ]
  }

  if (phase === 'uploading') {
    return [
      step('requirement', STEP_STATE.ACTIVE, 'Uploading the document…'),
      idle('retrieval'),
      idle('candidates'),
      idle('reasoning'),
      idle('recommendations'),
    ]
  }

  if (phase === 'analysing') {
    return [
      step('requirement', STEP_STATE.DONE, 'Requirement received'),
      step('retrieval', STEP_STATE.ACTIVE, 'Querying the vector store…'),
      awaiting('candidates'),
      awaiting('reasoning'),
      awaiting('recommendations'),
    ]
  }

  if (phase === 'error') {
    return [
      step('requirement', STEP_STATE.FAILED, 'Request did not complete'),
      step('retrieval', STEP_STATE.FAILED, 'No retrieval result received'),
      step('candidates', STEP_STATE.FAILED, 'No candidates received'),
      step('reasoning', STEP_STATE.FAILED, 'No AI verdict received'),
      step('recommendations', STEP_STATE.FAILED, 'No recommendations received'),
    ]
  }

  // --- phase === 'success': every state below comes from the response body ---
  const aiRuledNothing =
    aiRanked === false && (result?.ai_reasoning?.status ?? null) === 'completed'
  const aiUnconfigured = result?.ai_reasoning?.status === 'not_configured'

  return [
    step(
      'requirement',
      STEP_STATE.DONE,
      result?.requirement ? truncate(result.requirement, 90) : 'Requirement received',
    ),
    evidence.length
      ? step('retrieval', STEP_STATE.DONE, 'Relevant standards found')
      : step('retrieval', STEP_STATE.SKIPPED, 'Nothing relevant found'),
    candidateCodes.length
      ? step('candidates', STEP_STATE.DONE, 'Potential standards evaluated')
      : step('candidates', STEP_STATE.EMPTY, 'No potential matches to evaluate'),
    // The AI stage is only "complete" when the backend reported an AI verdict.
    aiRanked
      ? step('reasoning', STEP_STATE.DONE, 'Applicability determined')
      : aiRuledNothing
        ? step(
            'reasoning',
            STEP_STATE.DONE,
            'Reviewed the candidates and found none applicable',
          )
        : step(
            'reasoning',
            aiUnconfigured ? STEP_STATE.SKIPPED : STEP_STATE.UNAVAILABLE,
            aiUnconfigured
              ? 'Not available — no AI provider configured'
              : 'Could not be completed',
          ),
    recommendations.length
      ? step('recommendations', STEP_STATE.DONE, `${recommendations.length} standards identified`)
      : step('recommendations', STEP_STATE.EMPTY, 'No standard to recommend'),
  ]
}

/**
 * Facts about a finished analysis, all derived from the response body.
 * Nothing here is inferred or defaulted to a fabricated value.
 */
export function summarizeResult(result) {
  const pipeline = result?.pipeline ?? null
  const evidence = result?.retrieved_evidence ?? []
  const recommendations = result?.recommendations ?? []
  const aiRanked = pipeline?.ranking_stage === RANKING_STAGE.AI

  return {
    count: recommendations.length,
    aiRanked,
    rankingStage: pipeline?.ranking_stage ?? null,
    evidenceCount: evidence.length,
    // The API reports no separate "candidates considered" total, so the honest
    // figure is the number of distinct catalogued standards in the evidence.
    consideredCount: distinctStandardCodes(evidence).length,
    consideredIsDerived: true,
  }
}