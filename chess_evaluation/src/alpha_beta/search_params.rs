use serde::{Deserialize, Serialize};

/// All tunable search-pruning parameters.
///
/// `Default` reproduces the hardcoded baseline.
/// `Deserialize`/`Serialize` lets Optuna write a `params.json` and
/// `puzzle_bench --params params.json` read it for automated tuning.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SearchParams {
    // ── Reverse Futility Pruning ──────────────────────────────────────────────
    /// Margin per depth unit when the position is improving (cp). Baseline: 65.
    pub rfp_improving_margin: i32,
    /// Margin per depth unit when not improving (cp). Baseline: 85.
    pub rfp_not_improving_margin: i32,
    /// Maximum depth at which RFP fires. Baseline: 7.
    pub rfp_max_depth: i32,

    // ── Futility pruning ──────────────────────────────────────────────────────
    /// Margin per depth unit (cp). Baseline: 200.
    pub futility_margin: i32,
    /// Maximum depth at which futility pruning fires. Baseline: 3.
    pub futility_max_depth: i32,

    // ── Delta pruning (quiescence) ────────────────────────────────────────────
    /// Safety margin added to SEE before comparing to alpha/beta (cp). Baseline: 250.
    pub delta_margin: i32,

    // ── ProbCut ───────────────────────────────────────────────────────────────
    /// Offset above beta for the ProbCut threshold (cp). Baseline: 200.
    pub probcut_margin: i32,
    /// Minimum depth for ProbCut. Baseline: 5.
    pub probcut_min_depth: i32,

    // ── Null Move Pruning ─────────────────────────────────────────────────────
    /// Divisor for the excess-eval component of the NMP reduction.
    /// `excess = (eval − beta) / nmp_excess_divisor`, clamped to [0, 3].
    /// Baseline: 200.
    pub nmp_excess_divisor: i32,

    // ── Late Move Pruning ─────────────────────────────────────────────────────
    /// Quiet-move threshold table indexed by depth (depth 0 unused).
    /// When the position is improving, the effective threshold is 1.5×.
    /// For depths > 4 the depth-4 entry is reused.
    /// Baseline: [0, 4, 8, 13, 20].
    pub lmp_base: [usize; 5],

    // ── Aspiration windows ────────────────────────────────────────────────────
    /// Initial half-width of the aspiration window (cp). Baseline: 50.
    pub aspiration_delta: i32,
    /// Minimum depth at which aspiration windows are enabled. Baseline: 3.
    pub aspiration_min_depth: i32,

    // ── Singular Extension ────────────────────────────────────────────────────
    /// Margin below the TT score at which a move is considered singular (cp). Baseline: 50.
    pub se_margin: i32,

    // ── History pruning ───────────────────────────────────────────────────────
    /// History score threshold multiplier (per depth): prune if `hist < -threshold * depth`.
    /// Baseline: 256.
    pub history_pruning_threshold: i32,
    /// Maximum depth at which history pruning fires. Baseline: 3.
    pub history_pruning_max_depth: i32,
}

impl Default for SearchParams {
    /// Tuned via Optuna (200 trials, depth 7, 2000 Lichess puzzles ≥1500,
    /// HalfKAv2 1024×64 epoch-300 model). Verified at +4.5pp over baseline.
    fn default() -> Self {
        Self {
            rfp_improving_margin:      46,
            rfp_not_improving_margin:  147,
            rfp_max_depth:             4,
            futility_margin:           141,
            futility_max_depth:        1,
            delta_margin:              350,
            probcut_margin:            339,
            probcut_min_depth:         3,
            nmp_excess_divisor:        283,
            lmp_base:                  [0, 4, 15, 22, 39],
            aspiration_delta:          139,
            aspiration_min_depth:      4,
            se_margin:                 81,
            history_pruning_threshold: 722,
            history_pruning_max_depth: 4,
        }
    }
}

impl SearchParams {
    /// Original hardcoded baseline — preserved for reference and A/B testing.
    /// These were the values before Optuna tuning on the HalfKAv2 1024×64 model.
    pub fn baseline() -> Self {
        Self {
            rfp_improving_margin:      65,
            rfp_not_improving_margin:  85,
            rfp_max_depth:             7,
            futility_margin:           200,
            futility_max_depth:        3,
            delta_margin:              250,
            probcut_margin:            200,
            probcut_min_depth:         5,
            nmp_excess_divisor:        200,
            lmp_base:                  [0, 4, 8, 13, 20],
            aspiration_delta:          50,
            aspiration_min_depth:      3,
            se_margin:                 50,
            history_pruning_threshold: 256,
            history_pruning_max_depth: 3,
        }
    }
}
