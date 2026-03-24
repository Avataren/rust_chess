//! Pawn hash table and pawn structure evaluation.

use std::cell::UnsafeCell;

const PAWN_TABLE_SIZE: usize = 1 << 14; // 16k entries (~1 MB per thread)

/// A single pawn hash entry.  Stores all pawn-only MG/EG contributions
/// (PSTs, structure penalties, passed pawn base bonuses) plus the passed-pawn
/// bitboards needed to compute king-proximity bonuses at lookup time.
#[derive(Clone, Copy)]
pub(super) struct PawnHashEntry {
    pub(super) key:          u64,
    pub(super) pawn_mg:      i32,
    pub(super) pawn_eg:      i32,
    pub(super) struct_score: i32,
    pub(super) pass_w_mg:    i32,
    pub(super) pass_w_eg:    i32,
    pub(super) pass_b_mg:    i32,
    pub(super) pass_b_eg:    i32,
    pub(super) white_passers: u64,
    pub(super) black_passers: u64,
}

impl PawnHashEntry {
    pub(super) const EMPTY: Self = Self {
        key: 0, pawn_mg: 0, pawn_eg: 0, struct_score: 0,
        pass_w_mg: 0, pass_w_eg: 0, pass_b_mg: 0, pass_b_eg: 0,
        white_passers: 0, black_passers: 0,
    };
}

struct PawnHashTable {
    entries: Box<[PawnHashEntry; PAWN_TABLE_SIZE]>,
}

impl PawnHashTable {
    fn new() -> Self {
        Self { entries: Box::new([PawnHashEntry::EMPTY; PAWN_TABLE_SIZE]) }
    }

    #[inline(always)]
    fn probe(&self, key: u64) -> Option<&PawnHashEntry> {
        let e = &self.entries[key as usize & (PAWN_TABLE_SIZE - 1)];
        if e.key == key { Some(e) } else { None }
    }

    #[inline(always)]
    fn store(&mut self, entry: PawnHashEntry) {
        self.entries[entry.key as usize & (PAWN_TABLE_SIZE - 1)] = entry;
    }
}

// UnsafeCell allows mutation via a shared reference inside thread_local!
// This is sound because thread_local storage is never shared across threads.
struct UnsafePawnTable(UnsafeCell<PawnHashTable>);
unsafe impl Sync for UnsafePawnTable {}

thread_local! {
    pub(super) static PAWN_TABLE: UnsafePawnTable =
        UnsafePawnTable(UnsafeCell::new(PawnHashTable::new()));
}

/// Compute a pawn-only Zobrist hash from the two pawn bitboards.
#[inline(always)]
pub(super) fn pawn_key(white_pawns: u64, black_pawns: u64) -> u64 {
    white_pawns.wrapping_mul(0x9E3779B97F4A7C15)
        ^ black_pawns.wrapping_mul(0x517CC1B727220A95)
}

/// Pawn structure penalty (doubled + isolated) for one side.
/// Returns a positive value (penalty amount).
pub(super) fn pawn_structure_penalty(pawns_bb: u64) -> i32 {
    let mut penalty = 0i32;
    for file in 0..8usize {
        let on_file = (pawns_bb & super::FILE_MASKS[file]).count_ones() as i32;
        if on_file == 0 {
            continue;
        }
        if on_file > 1 {
            penalty += (on_file - 1) * super::DOUBLED_PAWN_PENALTY;
        }
        let mut adjacent = 0u64;
        if file > 0 { adjacent |= super::FILE_MASKS[file - 1]; }
        if file < 7 { adjacent |= super::FILE_MASKS[file + 1]; }
        if pawns_bb & adjacent == 0 {
            penalty += on_file * super::ISOLATED_PAWN_PENALTY;
        }
    }
    penalty
}

/// Probe (or populate) the per-thread pawn hash table.
///
/// Returns the cached `PawnHashEntry`, which the caller uses to read:
/// * `pawn_mg` / `pawn_eg`   — pawn PST contributions
/// * `struct_score`           — doubled + isolated penalty delta
/// * `pass_w_*` / `pass_b_*` — passed-pawn base bonuses
/// * `white_passers` / `black_passers` — bitboards for king-proximity calc
pub(super) fn probe_or_fill(
    white_pawns_bb: u64,
    black_pawns_bb: u64,
    // closures supplied by evaluate() to avoid circular imports
    compute_pst:       &impl Fn() -> (i32, i32),
    compute_passers_w: &impl Fn() -> (i32, i32, u64),
    compute_passers_b: &impl Fn() -> (i32, i32, u64),
) -> PawnHashEntry {
    let pkey = pawn_key(white_pawns_bb, black_pawns_bb);
    PAWN_TABLE.with(|t| {
        // SAFETY: thread_local, never aliased.
        let table = unsafe { &mut *t.0.get() };
        if let Some(e) = table.probe(pkey) {
            return *e;
        }
        let (pmg, peg)              = compute_pst();
        let (pw_mg, pw_eg, wpass)   = compute_passers_w();
        let (pb_mg, pb_eg, bpass)   = compute_passers_b();
        let w_struct = pawn_structure_penalty(white_pawns_bb);
        let b_struct = pawn_structure_penalty(black_pawns_bb);
        let entry = PawnHashEntry {
            key:          pkey,
            pawn_mg:      pmg,
            pawn_eg:      peg,
            struct_score: b_struct as i32 - w_struct as i32,
            pass_w_mg:    pw_mg,
            pass_w_eg:    pw_eg,
            pass_b_mg:    pb_mg,
            pass_b_eg:    pb_eg,
            white_passers: wpass,
            black_passers: bpass,
        };
        table.store(entry);
        entry
    })
}
