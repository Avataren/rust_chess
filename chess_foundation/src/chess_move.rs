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

    pub fn to_san_simple(&self) -> String {
        let mut san = String::new();

        // Add the destination square
        san.push(char::from_u32('a' as u32 + (self.start_square() % 8) as u32).unwrap()); // file
        san.push(char::from_u32('1' as u32 + (self.start_square() / 8) as u32).unwrap()); // rank

        san.push(char::from_u32('a' as u32 + (self.target_square() % 8) as u32).unwrap()); // file
        san.push(char::from_u32('1' as u32 + (self.target_square() / 8) as u32).unwrap()); // rank

        // Add promotion notation
        if let Some(promotion_piece_type) = self.promotion_piece_type() {
            //san.push('=');
            san.push(ChessPiece::piecetype_to_char(promotion_piece_type).to_ascii_uppercase()); // Assuming `to_char()` method exists for PieceType
        }
        san
    }        

    pub fn to_san(&self) -> String {
        let mut san = String::new();

        // Identify the piece
        if let Some(piece) = self.chess_piece {
            match piece.piece_type() {
                PieceType::Pawn => {
                    if self.capture.is_some() {
                        // Pawn captures include the file of the departing pawn
                        san.push(char::from_u32('a' as u32 + (self.start_square() % 8) as u32).unwrap());
                    }
                },
                _ => san.push(piece.to_char()), // Non-pawn pieces use their single uppercase letter
            }
        }

        // Add capture notation
        if self.capture.is_some() {
            san.push('x');
        }

        // Add the destination square
        san.push(char::from_u32('a' as u32 + (self.target_square() % 8) as u32).unwrap()); // file
        san.push(char::from_u32('1' as u32 + (self.target_square() / 8) as u32).unwrap()); // rank

        // Add promotion notation
        if let Some(promotion_piece_type) = self.promotion_piece_type() {
            san.push('=');
            san.push(ChessPiece::piecetype_to_char(promotion_piece_type)); // Assuming `to_char()` method exists for PieceType
        }

        // Optional: Add check or checkmate notation
        // if self.is_check() {
        //     san.push('+');
        // } else if self.is_checkmate() {
        //     san.push('#');
        // }

        san
    }    
    
}
