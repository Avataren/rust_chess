use crate::{
    get_bishop_move_patterns, get_king_move_patterns, get_knight_move_patterns,
    get_pawn_move_patterns, get_rook_move_patterns,
};
use chess_foundation::Bitboard;

pub struct PawnMovePatterns {
    pub moves: (Bitboard, Bitboard),    // (White moves, Black moves)
    pub captures: (Bitboard, Bitboard), // (White captures, Black captures)
}

pub struct MovePatterns {
    pub rook_move_patterns: Vec<Bitboard>,
    pub bishop_move_patterns: Vec<Bitboard>,
    pub knight_move_patterns: Vec<Bitboard>,
    pub king_move_patterns: Vec<Bitboard>,
    pub pawn_move_patterns: Vec<PawnMovePatterns>, // ((White moves, Black moves), (White captures, Black captures))
}

impl MovePatterns {
    pub fn new() -> MovePatterns {
        MovePatterns {
            rook_move_patterns: get_rook_move_patterns(),
            bishop_move_patterns: get_bishop_move_patterns(),
            knight_move_patterns: get_knight_move_patterns(),
            king_move_patterns: get_king_move_patterns(),
            pawn_move_patterns: get_pawn_move_patterns()
                .iter()
                .map(|(moves, captures)| PawnMovePatterns {
                    moves: *moves,
                    captures: *captures,
                })
                .collect(),
        }
    }
}
