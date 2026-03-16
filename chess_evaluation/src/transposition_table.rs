use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU8, Ordering};

use chess_foundation::ChessMove;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TtFlag {
    Exact,
    LowerBound,
    UpperBound,
}

/// How much effective depth to subtract per generation gap.
/// A 3-gen-old depth-12 entry becomes depth-0, replaceable by any new entry.
const AGE_COST: i32 = 4;


/// A transposition table entry — 16 bytes.
///
/// The best move is stored as a raw `u16` (the `move_value` field of
/// `ChessMove`) rather than `Option<ChessMove>`.  `Option<ChessMove>` costs
/// 8 bytes because `ChessMove` has no niche; a bare `u16` costs 2 bytes and
/// the sentinel value 0 (a1→a1, never a legal move) represents "no move".
#[derive(Clone, Copy)]
pub struct TtEntry {
    pub hash:       u64,
    pub depth:      i32,
    pub score:      i32,
    pub flag:       TtFlag,
    /// Search generation that wrote this entry.
    pub generation: u8,
    /// Packed move: bits[5:0]=start, bits[11:6]=target, bits[15:12]=flag.
    /// 0 means no best move recorded.
    best_move_raw:  u16,
}

impl TtEntry {
    /// Decode the stored move.  Returns `None` when no best move was recorded.
    ///
    /// The reconstructed `ChessMove` contains only start/target/flag — the
    /// `chess_piece` and `capture` fields are `None`.  Callers that need the
    /// full move (e.g. for make_move) should match it against the legal-move
    /// list, which already populates those fields.
    #[inline]
    pub fn best_move(&self) -> Option<ChessMove> {
        if self.best_move_raw == 0 {
            None
        } else {
            Some(ChessMove::new_with_flag(
                self.best_move_raw & 0x003F,         // bits 5:0  — start square
                (self.best_move_raw >> 6) & 0x003F,  // bits 11:6 — target square
                self.best_move_raw >> 12,             // bits 15:12 — move flag
            ))
        }
    }
}

impl Default for TtEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            depth: 0,
            score: 0,
            flag: TtFlag::Exact,
            generation: 0,
            best_move_raw: 0,
        }
    }
}

pub struct TranspositionTable {
    table: UnsafeCell<Vec<TtEntry>>,
    size: usize,
    /// Current search generation.  Incremented once per move via `new_search()`.
    generation: AtomicU8,
}

// Safety: concurrent probe/store calls accept benign data races (hash field
// acts as a validity check).  new_search() is always called before parallel
// access begins, so the generation counter is never raced.
unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    pub fn new(size: usize) -> Self {
        Self {
            table: UnsafeCell::new(vec![TtEntry::default(); size]),
            size,
            generation: AtomicU8::new(0),
        }
    }

    /// Advance the generation counter.  Call once before each new root search
    /// (i.e. each move).  Old entries remain in the table for move-ordering
    /// but are replaced more eagerly than fresh ones.
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// How many generations old is this entry?
    #[inline]
    fn age_of(&self, entry: &TtEntry) -> i32 {
        self.generation.load(Ordering::Relaxed).wrapping_sub(entry.generation) as i32
    }

    pub fn probe(&self, hash: u64) -> Option<TtEntry> {
        let idx = (hash as usize) % self.size;
        // Safety: benign data race — hash field validates the copy.
        let entry = unsafe { (&(*self.table.get()))[idx] };
        if entry.hash == hash { Some(entry) } else { None }
    }


    /// Store an entry using a generation-aware replacement policy.
    ///
    /// Replacement priority (lower effective depth = replaced first):
    ///   effective_depth = stored_depth - age * AGE_COST
    ///
    /// This means old entries (from previous moves) are evicted even when
    /// they are deep, preventing stale analysis from poisoning new searches.
    ///
    /// Store an entry using a generation-aware replacement policy.
    ///
    /// Replacement priority (lower effective depth = replaced first):
    ///   effective_depth = stored_depth - age * AGE_COST
    ///
    /// This means old entries (from previous moves) are evicted even when
    /// they are deep, preventing stale analysis from poisoning new searches.
    pub fn store(
        &self,
        hash: u64,
        depth: i32,
        score: i32,
        flag: TtFlag,
        best_move: Option<ChessMove>,
    ) {
        let idx = (hash as usize) % self.size;
        // Safety: concurrent reads/writes accepted (benign data race).
        let table = unsafe { &mut *self.table.get() };
        let existing = table[idx];
        let age = self.age_of(&existing);
        // Effective depth of the existing entry, penalised by age.
        let existing_eff = existing.depth - age * AGE_COST;
        // Replace if: slot is empty, hash collision (different position), or
        // the new entry's depth beats the age-adjusted existing depth.
        if existing.hash == 0 || existing.hash != hash || depth >= existing_eff {
            table[idx] = TtEntry {
                hash, depth, score, flag,
                generation: self.generation.load(Ordering::Relaxed),
                best_move_raw: best_move.map_or(0, |m| m.value()),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_empty_table_returns_none() {
        let tt = TranspositionTable::new(1024);
        assert!(tt.probe(0xDEADBEEF).is_none());
    }

    #[test]
    fn store_and_probe_exact_entry() {
        let tt = TranspositionTable::new(1024);
        tt.store(42, 3, 100, TtFlag::Exact, None);
        let entry = tt.probe(42).expect("should find stored entry");
        assert_eq!(entry.depth, 3);
        assert_eq!(entry.score, 100);
        assert_eq!(entry.flag, TtFlag::Exact);
        assert!(entry.best_move().is_none());
    }

    #[test]
    fn store_and_probe_lower_bound() {
        let tt = TranspositionTable::new(1024);
        tt.store(99, 5, 200, TtFlag::LowerBound, None);
        let entry = tt.probe(99).expect("should find stored entry");
        assert_eq!(entry.flag, TtFlag::LowerBound);
        assert_eq!(entry.score, 200);
    }

    #[test]
    fn store_and_probe_upper_bound() {
        let tt = TranspositionTable::new(1024);
        tt.store(77, 2, -50, TtFlag::UpperBound, None);
        let entry = tt.probe(77).expect("should find stored entry");
        assert_eq!(entry.flag, TtFlag::UpperBound);
        assert_eq!(entry.score, -50);
    }

    #[test]
    fn hash_collision_returns_none() {
        // With size=8, hashes 0 and 8 map to the same slot but are different keys.
        let tt = TranspositionTable::new(8);
        tt.store(0, 3, 100, TtFlag::Exact, None);
        assert!(tt.probe(8).is_none(), "different hash in same slot must return None");
    }

    #[test]
    fn deeper_entry_replaces_shallower() {
        let tt = TranspositionTable::new(1024);
        tt.store(42, 2, 100, TtFlag::Exact, None);
        tt.store(42, 5, 200, TtFlag::Exact, None); // deeper
        let entry = tt.probe(42).expect("should find entry");
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.score, 200);
    }

    #[test]
    fn shallower_entry_does_not_replace_deeper() {
        let tt = TranspositionTable::new(1024);
        tt.store(42, 5, 200, TtFlag::Exact, None);
        tt.store(42, 2, 100, TtFlag::Exact, None); // shallower
        let entry = tt.probe(42).expect("should find entry");
        assert_eq!(entry.depth, 5, "deeper entry should survive");
        assert_eq!(entry.score, 200);
    }

    #[test]
    fn stored_best_move_is_retrieved() {
        let tt = TranspositionTable::new(1024);
        let mv = ChessMove::new(0, 16); // a1 → a3
        tt.store(55, 4, 150, TtFlag::Exact, Some(mv));
        let entry = tt.probe(55).expect("should find entry");
        let stored = entry.best_move().expect("best_move should be stored");
        assert_eq!(stored.start_square(), 0);
        assert_eq!(stored.target_square(), 16);
    }

    #[test]
    fn entry_replaced_for_empty_slot() {
        // An initially empty slot (hash==0) should be filled by any store.
        let tt = TranspositionTable::new(1024);
        assert!(tt.probe(1).is_none());
        tt.store(1, 1, 0, TtFlag::Exact, None);
        assert!(tt.probe(1).is_some());
    }

    // ── Generation / aging ───────────────────────────────────────────────────

    #[test]
    fn new_search_increments_generation() {
        let tt = TranspositionTable::new(1024);
        // Store an entry at generation 0.
        tt.store(1, 5, 100, TtFlag::Exact, None);
        assert_eq!(tt.probe(1).unwrap().generation, 0);

        tt.new_search();
        // Store another entry at generation 1.
        tt.store(2, 5, 200, TtFlag::Exact, None);
        assert_eq!(tt.probe(2).unwrap().generation, 1);

        // Original entry still visible (different hash).
        assert_eq!(tt.probe(1).unwrap().generation, 0);
    }

    #[test]
    fn fresh_shallow_entry_evicts_stale_deep_entry() {
        // AGE_COST = 4, so after 3 new_search() calls the old entry at depth 10
        // has effective_depth = 10 - 3*4 = -2, which any new entry (depth >= -2) beats.
        let tt = TranspositionTable::new(1024);
        tt.store(42, 10, 999, TtFlag::Exact, None); // deep, generation 0

        tt.new_search(); // generation 1
        tt.new_search(); // generation 2
        tt.new_search(); // generation 3  →  effective_depth = 10 - 12 = -2

        // A shallow depth-1 entry (>= -2) should now evict the old one.
        tt.store(42, 1, 42, TtFlag::Exact, None);
        let entry = tt.probe(42).expect("should find entry");
        assert_eq!(entry.depth, 1, "fresh shallow entry should have evicted stale deep entry");
        assert_eq!(entry.score, 42);
        assert_eq!(entry.generation, 3);
    }

    #[test]
    fn fresh_shallow_does_not_evict_recent_deep_entry() {
        // After only 1 new_search(), a depth-10 entry has effective_depth = 10 - 4 = 6.
        // A depth-5 entry (< 6) must NOT evict it.
        let tt = TranspositionTable::new(1024);
        tt.store(42, 10, 999, TtFlag::Exact, None); // deep, generation 0

        tt.new_search(); // generation 1  →  effective_depth = 10 - 4 = 6

        tt.store(42, 5, 42, TtFlag::Exact, None); // depth 5 < 6, should not replace
        let entry = tt.probe(42).expect("should find entry");
        assert_eq!(entry.depth, 10, "recent deep entry should survive a shallower store");
        assert_eq!(entry.score, 999);
    }

    #[test]
    fn probe_returns_stale_entry_for_move_ordering() {
        // Old entries should still be retrievable via probe (used for move ordering)
        // even after several new_search() calls — only store() applies the eviction logic.
        let tt = TranspositionTable::new(1024);
        let mv = ChessMove::new(4, 20); // e1 → e3 (arbitrary)
        tt.store(77, 8, 300, TtFlag::Exact, Some(mv));

        tt.new_search();
        tt.new_search();
        tt.new_search();

        // Entry must still be probed (not auto-evicted — eviction only happens on store).
        let entry = tt.probe(77).expect("stale entry should still be visible via probe");
        assert_eq!(entry.best_move().unwrap().start_square(), 4,
            "stale entry's best_move should still be accessible for move ordering");
    }

    #[test]
    fn generation_wraps_without_panic() {
        // Advance through all 256 generations and verify the table still works.
        let tt = TranspositionTable::new(1024);
        for _ in 0..255 {
            tt.new_search();
        }
        // Now at generation 255; one more wraps to 0.
        tt.new_search();
        tt.store(1, 3, 50, TtFlag::Exact, None);
        let entry = tt.probe(1).expect("should find entry after wrap-around");
        assert_eq!(entry.generation, 0);
        assert_eq!(entry.score, 50);
    }

    #[test]
    fn same_generation_depth_rule_still_applies() {
        // Within the same generation, the old depth-beats-shallower rule holds.
        let tt = TranspositionTable::new(1024);
        tt.new_search(); // generation 1
        tt.store(42, 8, 100, TtFlag::Exact, None);
        tt.store(42, 3, 200, TtFlag::Exact, None); // shallower, same gen — must not replace
        let entry = tt.probe(42).expect("should find entry");
        assert_eq!(entry.depth, 8, "deeper same-gen entry must not be overwritten by shallower");
        assert_eq!(entry.score, 100);
    }
}
