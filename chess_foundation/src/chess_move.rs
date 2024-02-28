use std::default;

use crate::{piece::PieceType, ChessPiece};

#[derive(Debug, Clone, Copy)]
pub struct ChessMove {
    move_value: u16, // Compact representation of the move
    pub chess_piece: Option<ChessPiece>,
    pub capture: Option<ChessPiece>,
}

impl default::Default for ChessMove {
    fn default() -> Self {
        Self {
            move_value: 0,
            chess_piece: None,
            capture: None,
        }
    }
}

impl ChessMove {
    // Constants for flags
    pub const NO_FLAG: u16 = 0b0000;
    pub const EN_PASSANT_CAPTURE_FLAG: u16 = 0b0001;
    pub const CASTLE_FLAG: u16 = 0b0010;
    pub const PAWN_TWO_UP_FLAG: u16 = 0b0011;

    pub const PROMOTE_TO_QUEEN_FLAG: u16 = 0b0100;
    pub const PROMOTE_TO_KNIGHT_FLAG: u16 = 0b0101;
    pub const PROMOTE_TO_ROOK_FLAG: u16 = 0b0110;
    pub const PROMOTE_TO_BISHOP_FLAG: u16 = 0b0111;

    // Masks
    const START_SQUARE_MASK: u16 = 0b0000000000111111;
    const TARGET_SQUARE_MASK: u16 = 0b0000111111000000;
    const FLAG_MASK: u16 = 0b1111000000000000;

    // Constructors
    pub fn new(start_square: u16, target_square: u16) -> Self {
        Self {
            //use 6 bits pr square position
            move_value: start_square | (target_square << 6),
            capture: None,
            chess_piece: None,
        }
    }

    pub fn new_with_flag(start_square: u16, target_square: u16, flag: u16) -> Self {
        // println!("ChessMove new_with_flag --- start_square: {}, target_square: {}, flag: {}", start_square, target_square, flag);
        Self {
            move_value: start_square | (target_square << 6) | (flag << 12),
            capture: None,
            chess_piece: None,
        }
    }

    pub fn new_capture_with_flag(
        start_square: u16,
        capture: Option<ChessPiece>,
        target_square: u16,
        flag: u16,
    ) -> Self {
        Self {
            move_value: start_square | (target_square << 6) | (flag << 12),
            capture: capture,
            chess_piece: None,
        }
    }

    pub fn set_piece(&mut self, piece: ChessPiece) {
        self.chess_piece = Some(piece);
    }

    pub fn set_capture(&mut self, piece: ChessPiece) {
        self.capture = Some(piece);
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

    pub fn has_flag(&self, flag: u16) -> bool {
        self.flag() == flag
    }

    pub fn set_flag(&mut self, flag: u16) {
        self.move_value |= flag << 12;
    }

    pub fn clear_flag(&mut self, flag: u16) {
        self.move_value &= !(flag << 12);
    }

    pub fn is_promotion(&self) -> bool {
        let flag = self.flag();
        flag >= Self::PROMOTE_TO_QUEEN_FLAG && flag <= Self::PROMOTE_TO_BISHOP_FLAG
    }

    pub fn promotion_piece_type(&self) -> Option<PieceType> {
        if !self.is_promotion() {
            return None;
        }

        match self.flag() {
            Self::PROMOTE_TO_ROOK_FLAG => Some(PieceType::Rook), // Assuming `is_black()` method exists
            Self::PROMOTE_TO_KNIGHT_FLAG => Some(PieceType::Knight),
            Self::PROMOTE_TO_BISHOP_FLAG => Some(PieceType::Bishop),
            Self::PROMOTE_TO_QUEEN_FLAG => Some(PieceType::Queen),
            _ => None, // Return None instead of ChessPiece::NONE
        }
    }

    pub fn to_basic_algebraic_notation(&self) -> String {
        let start_square = self.start_square() as u8;
        let target_square = self.target_square() as u8;
        let piece = self
            .chess_piece
            .unwrap_or_else(|| ChessPiece::new(PieceType::None, true)); // Default to a dummy piece if None

        // Convert square indices to board coordinates
        let start_file = (97 + (start_square % 8)) as char;
        //let start_rank = 1 + start_square / 8; //ignore disambiguation for now
        let target_pos = format!(
            "{}{}",
            (97 + (target_square % 8)) as char,
            1 + target_square / 8
        );

        let piece_type = piece.piece_type();
        match self.capture {
            Some(_) => {
                // For pawn captures, include the starting file in the notation
                if piece_type == PieceType::Pawn {
                    format!("{}x{}", start_file, target_pos)
                } else {
                    format!("{}x{}", piece_type, target_pos)
                }
            }
            None => {
                // For non-captures, just use the piece type and target position
                // Include start position if necessary for disambiguation
                format!("{}{}", piece_type, target_pos)
            }
        }
        // Additional logic for special moves (castling, en passant, promotion) can be added here
    }
}
