// src/fen.rs

use crate::ChessBoard;
use chess_foundation::piece::PieceType; // Ensure this path matches your project structure

pub struct FENParser;

impl FENParser {
    pub fn set_board_from_fen(board: &mut ChessBoard, fen: &str) {
        board.clear();

        let mut parts: Vec<&str> = fen.split_whitespace().collect();

        // Ensure there are at least 6 parts, filling in missing parts with default values
        while parts.len() < 6 {
            match parts.len() {
                1 => parts.push("w"), // Default active color to white
                2 => parts.push("-"), // Default castling availability to none
                3 => parts.push("-"), // Default en passant target to none
                4 => parts.push("0"), // Default halfmove clock to 0
                5 => parts.push("1"), // Default fullmove number to 1
                _ => break, // This should not happen
            }
        }

        let board_layout = parts[0];
        let active_color = parts[1];
        let castling_rights = parts[2];
        let en_passant = parts[3];
        let halfmove_clock = parts[4].parse::<u32>().unwrap_or(0);
        let fullmove_number = parts[5].parse::<u32>().unwrap_or(1);

        // Set pieces on the board
        Self::set_pieces_from_fen(board, board_layout);

        // // Set active color
        // board.set_active_color(active_color == "w");
        board.set_active_color(active_color == "w");
        // // Set castling rights
        board.set_castling_rights_from_fen(castling_rights);

        // // Set en passant target square
        // // Assuming ChessBoard has a method to set en passant target
        // board.set_en_passant_target(en_passant);

        // // Set halfmove clock and fullmove number
        // // Assuming ChessBoard has methods to set these values
        // board.set_halfmove_clock(halfmove_clock);
        // board.set_fullmove_number(fullmove_number);
    }

    fn set_pieces_from_fen(board: &mut ChessBoard, layout: &str) {
        let ranks: Vec<&str> = layout.split('/').collect();
        if ranks.len() != 8 {
            panic!(
                "Invalid FEN board layout: expected 8 ranks, found {}",
                ranks.len()
            );
        }

        for (rank_idx, rank) in ranks.iter().enumerate() {
            let mut file_idx = 0;
            for c in rank.chars() {
                if let Some(digit) = c.to_digit(10) {
                    file_idx += digit as usize; // Skip empty squares
                } else {
                    let square = 8 * (7 - rank_idx) + file_idx; // Calculate square index from rank and file
                    let piece_type = Self::piece_type_from_fen_char(c);
                    let is_white = c.is_uppercase();

                    board.set_piece_at_square(square as u16, piece_type, is_white);
                    file_idx += 1;
                }
            }
        }
    }

    pub fn board_to_fen(board: &ChessBoard) -> String {
        let mut fen = String::new();

        // 1. Piece placement
        for rank in 0..8 {
            let mut empty_squares = 0;
            for file in 0..8 {
                let square = (7-rank) * 8 + file;
                if let Some(piece) = board.get_piece_at_square(square as u16) {
                    if empty_squares > 0 {
                        fen.push_str(&empty_squares.to_string());
                        empty_squares = 0;
                    }
                    let piece_char = Self::fen_char_from_piece_type(piece.piece_type(), piece.is_white());
                    fen.push(piece_char);
                } else {
                    empty_squares += 1;
                }
            }
            if empty_squares > 0 {
                fen.push_str(&empty_squares.to_string());
            }
            if rank < 7 {
                fen.push('/');
            }
        }

        // 2. Active color
        fen.push(' ');
        fen.push(if board.is_white_active() { 'w' } else { 'b' });

        // 3. Castling availability
        fen.push(' ');
        fen.push_str(&board.get_castling_rights());

        // 4. En passant target square
        fen.push(' ');
        //fen.push_str(&board.get_en_passant_target().unwrap_or("-".to_string()));
        fen.push('-');

        // 5. Halfmove clock
        fen.push(' ');
        fen.push('0');
        //fen.push_str(&board.get_halfmove_clock().to_string());

        // 6. Fullmove number
        fen.push(' ');
        fen.push('1');
        //fen.push_str(&board.get_fullmove_number().to_string());

        fen
    }

    fn fen_char_from_piece_type(piece_type: PieceType, is_white: bool) -> char {
        let piece_char = match piece_type {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
            PieceType::None => panic!("Invalid piece type: None"),
        };
        if is_white {
            piece_char.to_ascii_uppercase()
        } else {
            piece_char
        }
    }    

    fn piece_type_from_fen_char(c: char) -> PieceType {
        match c.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => panic!("Invalid FEN piece character: '{}'", c),
        }
    }
}
