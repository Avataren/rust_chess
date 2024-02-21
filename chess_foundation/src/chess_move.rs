#[derive(Debug, Clone, Copy)]
pub enum Piece {
    None,
    Rook,
    Knight,
    Bishop,
    Queen,
}

#[derive(Debug, Clone, Copy)]
pub struct ChessMove {
    move_value: u16, // Compact representation of the move
}

impl ChessMove {
    // Constants for flags
    const NO_FLAG: u16 = 0b0000;
    const EN_PASSANT_CAPTURE_FLAG: u16 = 0b0001;
    const CASTLE_FLAG: u16 = 0b0010;
    const PAWN_TWO_UP_FLAG: u16 = 0b0011;

    const PROMOTE_TO_QUEEN_FLAG: u16 = 0b0100;
    const PROMOTE_TO_KNIGHT_FLAG: u16 = 0b0101;
    const PROMOTE_TO_ROOK_FLAG: u16 = 0b0110;
    const PROMOTE_TO_BISHOP_FLAG: u16 = 0b0111;

    // Masks
    const START_SQUARE_MASK: u16 = 0b0000000000111111;
    const TARGET_SQUARE_MASK: u16 = 0b0000111111000000;
    const FLAG_MASK: u16 = 0b1111000000000000;

    // Constructors
    pub fn new(start_square: u16, target_square: u16) -> Self {
        Self {
            move_value: start_square | (target_square << 6),
        }
    }

    pub fn new_with_flag(start_square: u16, target_square: u16, flag: u16) -> Self {
        Self {
            move_value: start_square | (target_square << 6) | (flag << 12),
        }
    }

    // Accessor methods
    pub fn value(&self) -> u16 {
        self.move_value
    }

    pub fn start_square(&self) -> u16 {
        self.move_value & Self::START_SQUARE_MASK
    }

    pub fn target_square(&self) -> u16 {
        (self.move_value & Self::TARGET_SQUARE_MASK) >> 6
    }

    pub fn flag(&self) -> u16 {
        (self.move_value & Self::FLAG_MASK) >> 12
    }

    pub fn is_promotion(&self) -> bool {
        let flag = self.flag();
        flag >= Self::PROMOTE_TO_QUEEN_FLAG && flag <= Self::PROMOTE_TO_BISHOP_FLAG
    }

    // Example method to determine the promotion piece type
    pub fn promotion_piece_type(&self) -> Option<Piece> {
        if !self.is_promotion() {
            return None;
        }

        match self.flag() {
            Self::PROMOTE_TO_ROOK_FLAG => Some(Piece::Rook),
            Self::PROMOTE_TO_KNIGHT_FLAG => Some(Piece::Knight),
            Self::PROMOTE_TO_BISHOP_FLAG => Some(Piece::Bishop),
            Self::PROMOTE_TO_QUEEN_FLAG => Some(Piece::Queen),
            _ => None,
        }
    }
}
