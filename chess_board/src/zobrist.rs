use std::sync::OnceLock;

use chess_foundation::piece::PieceType;

static TABLE: OnceLock<ZobristTable> = OnceLock::new();

pub struct ZobristTable {
    /// [piece_index][square]  — piece_index 0-5 = white, 6-11 = black
    pub pieces: [[u64; 64]; 12],
    /// One entry per castling-rights nibble (0..=15)
    pub castling: [u64; 16],
    /// XORed in when it is white's turn
    pub side_to_move: u64,
}

fn xorshift64(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

impl ZobristTable {
    fn build() -> Self {
        // Same golden-ratio seed used in the opening book for consistency.
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || xorshift64(&mut s);

        let mut pieces = [[0u64; 64]; 12];
        for p in pieces.iter_mut() {
            for sq in p.iter_mut() {
                *sq = next();
            }
        }

        let mut castling = [0u64; 16];
        for c in castling.iter_mut() {
            *c = next();
        }

        ZobristTable {
            pieces,
            castling,
            side_to_move: next(),
        }
    }

    pub fn get() -> &'static Self {
        TABLE.get_or_init(Self::build)
    }

    pub fn piece_idx(piece_type: PieceType, is_white: bool) -> usize {
        let base = match piece_type {
            PieceType::Pawn   => 0,
            PieceType::Knight => 1,
            PieceType::Bishop => 2,
            PieceType::Rook   => 3,
            PieceType::Queen  => 4,
            PieceType::King   => 5,
            _                 => 0,
        };
        if is_white { base } else { base + 6 }
    }
}
