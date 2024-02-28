#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    None = 0,
    Pawn = 1,
    Knight = 2,
    Bishop = 3,
    Rook = 4,
    Queen = 5,
    King = 6,
}

#[derive(Debug, Clone, Copy)]
pub struct ChessPiece(u8);

impl ChessPiece {
    const MASK_COLOR: u8 = 0b1000; // 4th bit for color, 1 for black, 0 for white
    const MASK_TYPE: u8 = 0b0111; // Lower 3 bits for piece type

    // Constructor
    pub fn new(piece_type: PieceType, is_white: bool) -> Self {
        let color_bit = if is_white { 0 } else { Self::MASK_COLOR };
        ChessPiece((piece_type as u8 & Self::MASK_TYPE) | color_bit)
    }

    // Check if the piece is black
    pub fn is_black(&self) -> bool {
        (self.0 & Self::MASK_COLOR) != 0
    }

    pub fn is_white(&self) -> bool {
        !self.is_black()
    }

    // Get the type of the piece
    pub fn piece_type(&self) -> PieceType {
        match self.0 & Self::MASK_TYPE {
            0 => PieceType::None,
            1 => PieceType::Pawn,
            2 => PieceType::Knight,
            3 => PieceType::Bishop,
            4 => PieceType::Rook,
            5 => PieceType::Queen,
            6 => PieceType::King,
            _ => unreachable!(), // This should never happen since we mask with 0b0111
        }
    }
}

use std::fmt;

impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let piece = match self {
            PieceType::None => "???",
            PieceType::Pawn => "",
            PieceType::Knight => "N",
            PieceType::Bishop => "B",
            PieceType::Rook => "R",
            PieceType::Queen => "Q",
            PieceType::King => "K",
        };
        write!(f, "{}", piece)
    }
}
