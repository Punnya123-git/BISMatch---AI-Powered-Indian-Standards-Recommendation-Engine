/** UI-level constants that mirror the backend contract. */

export const ACCEPTED_DOCUMENT_TYPES = '.pdf,.txt,.md,text/plain,application/pdf'

export const MAX_UPLOAD_SIZE_MB = 25

export const MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

export const MIN_REQUIREMENT_LENGTH = 3

export const MAX_REQUIREMENT_LENGTH = 4000

/** Recommendation statuses returned by POST /api/recommendations/analyze. */
export const RECOMMENDATION_STATUS = {
  OK: 'ok',
  AI_UNAVAILABLE: 'ai_unavailable',
  PLACEHOLDER: 'placeholder',
  NOT_CONFIGURED: 'not_configured',
  DATASET_UNAVAILABLE: 'dataset_unavailable',
}

export const STATUS_LABELS = {
  [RECOMMENDATION_STATUS.OK]: 'Applicable standards identified',
  [RECOMMENDATION_STATUS.AI_UNAVAILABLE]: 'AI analysis unavailable',
  [RECOMMENDATION_STATUS.PLACEHOLDER]: 'No applicable standard identified',
  [RECOMMENDATION_STATUS.NOT_CONFIGURED]: 'Analysis unavailable',
  [RECOMMENDATION_STATUS.DATASET_UNAVAILABLE]: 'Standards data unavailable',
}

export const STATUS_TONES = {
  [RECOMMENDATION_STATUS.OK]: 'success',
  [RECOMMENDATION_STATUS.AI_UNAVAILABLE]: 'warning',
  [RECOMMENDATION_STATUS.PLACEHOLDER]: 'neutral',
  [RECOMMENDATION_STATUS.NOT_CONFIGURED]: 'warning',
  [RECOMMENDATION_STATUS.DATASET_UNAVAILABLE]: 'error',
}

/**
 * `pipeline.ranking_stage` values returned by the backend. The stage is the only
 * authority the UI has on *how* a result was produced, so it drives every
 * label. Only `AI` may ever accompany a non-empty recommendations list.
 */
export const RANKING_STAGE = {
  AI: 'ai_reasoning_ranking',
  DETERMINISTIC: 'deterministic_retrieval_ranking',
  UNAVAILABLE: 'ai_reasoning_unavailable',
}

export const RANKING_STAGE_LABELS = {
  [RANKING_STAGE.AI]: 'AI applicability analysis complete',
  [RANKING_STAGE.DETERMINISTIC]: 'Search complete',
  [RANKING_STAGE.UNAVAILABLE]: 'AI analysis unavailable',
}

export const RANKING_STAGE_HINTS = {
  [RANKING_STAGE.AI]:
    'The language model reviewed the standards found in the verified catalogue and determined which ones apply to your requirement.',
  [RANKING_STAGE.DETERMINISTIC]:
    'Relevant standards were found, but no AI applicability analysis was recorded for this response.',
  [RANKING_STAGE.UNAVAILABLE]:
    'The AI applicability analysis could not be completed, so no standard is presented as applicable. The potential matches below are search results only.',
}

/** `applicability_type` values on a recommendation. */
export const APPLICABILITY_LABELS = {
  direct: 'Directly applicable',
  related: 'Related / supporting',
  needs_verification: 'Verification required',
}

export const APPLICABILITY_TONES = {
  direct: 'success',
  related: 'info',
  needs_verification: 'warning',
}

export const APPLICABILITY_DESCRIPTIONS = {
  direct: 'This standard directly governs the product or work you described.',
  related: 'A related or supporting standard — not the primary governing standard.',
  needs_verification:
    'Not enough verified evidence to confirm full applicability. Confirm manually before use.',
}

/**
 * Example requirements, one per procurement domain.
 *
 * These are *user-facing seeds* for a click-to-fill interaction, not test
 * fixtures. Each carries a realistic clause a procurement officer would write,
 * and each is honest about coverage: domains outside the local verified index
 * (rotating electrical machines) demonstrate the coverage-gap state rather than
 * pretending a recommendation exists.
 */
export const EXAMPLE_DOMAINS = [
  {
    id: 'motor',
    label: 'Electrical Motor',
    icon: '⚙',
    requirement:
      'Supply of 15 kW IE3 three-phase squirrel cage induction motor, 415 V, 50 Hz, IP55 enclosure, foot mounted, class F insulation',
  },
  {
    id: 'cement',
    label: 'Cement',
    icon: '🧱',
    requirement:
      'Ordinary Portland Cement of 43 Grade for structural concrete work, supplied in 50 kg laminated bags',
  },
  {
    id: 'concrete',
    label: 'Concrete',
    icon: '🏗',
    requirement:
      'M40 grade ready-mixed concrete for reinforced columns, with compressive strength testing at 7 and 28 days',
  },
  {
    id: 'transformer',
    label: 'Transformer',
    icon: '🔌',
    requirement:
      'Distribution transformer 500 kVA 11/0.415 kV as per IS 1180 with energy efficiency class conforming to BEE regulations',
  },
  {
    id: 'safety',
    label: 'Safety Equipment',
    icon: '🦺',
    requirement:
      'Industrial machine requiring operator safety protection: emergency stop, guard interlocks and safety relays to IEC 60204-1',
  },
  {
    id: 'water',
    label: 'Packaged Water',
    icon: '💧',
    requirement:
      'BIS standards applicable to packaged drinking water manufacturing and bottling plant',
  },
]
