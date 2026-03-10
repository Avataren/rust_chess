use chess_foundation::ChessMove;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TtFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Clone, Copy)]
pub struct TtEntry {
    pub hash: u64,
    pub depth: i32,
    pub score: i32,
    pub flag: TtFlag,
    pub best_move: Option<ChessMove>,
}

impl Default for TtEntry {
    fn default() -> Self {
        Self {
            hash: 0,
            depth: 0,
            score: 0,
            flag: TtFlag::Exact,
            best_move: None,
        }
    }
}

pub struct TranspositionTable {
    table: Vec<TtEntry>,
    size: usize,
}

impl TranspositionTable {
    pub fn new(size: usize) -> Self {
        Self {
            table: vec![TtEntry::default(); size],
            size,
        }
    }

    pub fn probe(&self, hash: u64) -> Option<&TtEntry> {
        let idx = (hash as usize) % self.size;
        let entry = &self.table[idx];
        if entry.hash == hash {
            Some(entry)
        } else {
            None
        }
    }

    /// Store an entry. Replaces if empty, same position, or new depth is deeper.
    pub fn store(
        &mut self,
        hash: u64,
        depth: i32,
        score: i32,
        flag: TtFlag,
        best_move: Option<ChessMove>,
    ) {
        let idx = (hash as usize) % self.size;
        let existing = &self.table[idx];
        // Replace if: slot is empty, hash collision (different position), or same position
        // with a deeper (or equal) search.  Never overwrite a deeper entry for the same position.
        if existing.hash == 0 || existing.hash != hash || depth >= existing.depth {
            self.table[idx] = TtEntry { hash, depth, score, flag, best_move };
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
        let mut tt = TranspositionTable::new(1024);
        tt.store(42, 3, 100, TtFlag::Exact, None);
        let entry = tt.probe(42).expect("should find stored entry");
        assert_eq!(entry.depth, 3);
        assert_eq!(entry.score, 100);
        assert_eq!(entry.flag, TtFlag::Exact);
        assert!(entry.best_move.is_none());
    }

    #[test]
    fn store_and_probe_lower_bound() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(99, 5, 200, TtFlag::LowerBound, None);
        let entry = tt.probe(99).expect("should find stored entry");
        assert_eq!(entry.flag, TtFlag::LowerBound);
        assert_eq!(entry.score, 200);
    }

    #[test]
    fn store_and_probe_upper_bound() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(77, 2, -50, TtFlag::UpperBound, None);
        let entry = tt.probe(77).expect("should find stored entry");
        assert_eq!(entry.flag, TtFlag::UpperBound);
        assert_eq!(entry.score, -50);
    }

    #[test]
    fn hash_collision_returns_none() {
        // With size=8, hashes 0 and 8 map to the same slot but are different keys.
        let mut tt = TranspositionTable::new(8);
        tt.store(0, 3, 100, TtFlag::Exact, None);
        assert!(tt.probe(8).is_none(), "different hash in same slot must return None");
    }

    #[test]
    fn deeper_entry_replaces_shallower() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(42, 2, 100, TtFlag::Exact, None);
        tt.store(42, 5, 200, TtFlag::Exact, None); // deeper
        let entry = tt.probe(42).expect("should find entry");
        assert_eq!(entry.depth, 5);
        assert_eq!(entry.score, 200);
    }

    #[test]
    fn shallower_entry_does_not_replace_deeper() {
        let mut tt = TranspositionTable::new(1024);
        tt.store(42, 5, 200, TtFlag::Exact, None);
        tt.store(42, 2, 100, TtFlag::Exact, None); // shallower
        let entry = tt.probe(42).expect("should find entry");
        assert_eq!(entry.depth, 5, "deeper entry should survive");
        assert_eq!(entry.score, 200);
    }

    #[test]
    fn stored_best_move_is_retrieved() {
        let mut tt = TranspositionTable::new(1024);
        let mv = ChessMove::new(0, 16); // a1 → a3
        tt.store(55, 4, 150, TtFlag::Exact, Some(mv));
        let entry = tt.probe(55).expect("should find entry");
        let stored = entry.best_move.expect("best_move should be stored");
        assert_eq!(stored.start_square(), 0);
        assert_eq!(stored.target_square(), 16);
    }

    #[test]
    fn entry_replaced_for_empty_slot() {
        // An initially empty slot (hash==0) should be filled by any store.
        let mut tt = TranspositionTable::new(1024);
        assert!(tt.probe(1).is_none());
        tt.store(1, 1, 0, TtFlag::Exact, None);
        assert!(tt.probe(1).is_some());
    }
}
