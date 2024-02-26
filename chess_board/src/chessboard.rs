use crate::FENParser;
use chess_foundation::{bitboard::Bitboard, piece::PieceType, ChessMove, ChessPiece};
pub struct ChessBoard {
    white: Bitboard,
    black: Bitboard,
    pawns: Bitboard,
    knights: Bitboard,
    bishops: Bitboard,
    rooks: Bitboard,
    queens: Bitboard,
    kings: Bitboard,
    pub castling_rights: u8,
    move_history: Vec<(ChessMove, u8)>,
}

#[repr(u8)]
pub enum CastlingRights {
    WhiteKingSide = 0b1000,
    WhiteQueenSide = 0b0100,
    BlackKingSide = 0b0010,
    BlackQueenSide = 0b0001,
    AllCastlingRights = 0b1111,
}

impl ChessBoard {
    pub fn new() -> Self {
        ChessBoard {
            white: Bitboard(0xFFFF),                  // First two ranks
            black: Bitboard(0xFFFF_0000_0000_0000),   // Last two ranks
            pawns: Bitboard(0x00FF_0000_0000_FF00),   // Second and seventh ranks
            knights: Bitboard(0x4200_0000_0000_0042), // b1, g1, b8, g8
            bishops: Bitboard(0x2400_0000_0000_0024), // c1, f1, c8, f8
            rooks: Bitboard(0x8100_0000_0000_0081),   // a1, h1, a8, h8
            queens: Bitboard(0x0800_0000_0000_0008),  // d1, d8
            kings: Bitboard(0x1000_0000_0000_0010),   // e1, e8
            castling_rights: CastlingRights::AllCastlingRights as u8,
            move_history: Vec::with_capacity(100),
        }
    }

    pub fn clear(&mut self) {
        self.white = Bitboard(0);
        self.black = Bitboard(0);
        self.pawns = Bitboard(0);
        self.knights = Bitboard(0);
        self.bishops = Bitboard(0);
        self.rooks = Bitboard(0);
        self.queens = Bitboard(0);
        self.kings = Bitboard(0);
        self.castling_rights = CastlingRights::AllCastlingRights as u8;
        self.move_history.clear();
    }

    pub fn get_fen_castling_rights(&self) -> String {
        let mut result = String::new();
        if self.castling_rights & CastlingRights::WhiteKingSide as u8 != 0 {
            result.push('K');
        }
        if self.castling_rights & CastlingRights::WhiteQueenSide as u8 != 0 {
            result.push('Q');
        }
        if self.castling_rights & CastlingRights::BlackKingSide as u8 != 0 {
            result.push('k');
        }
        if self.castling_rights & CastlingRights::BlackQueenSide as u8 != 0 {
            result.push('q');
        }
        result
    }

    pub fn set_castling_rights_from_fen(&mut self, fen_castling: &str) {
        let mut castling_rights: u8 = 0;

        if fen_castling.contains('K') {
            castling_rights |= CastlingRights::WhiteKingSide as u8;
        }
        if fen_castling.contains('Q') {
            castling_rights |= CastlingRights::WhiteQueenSide as u8;
        }
        if fen_castling.contains('k') {
            castling_rights |= CastlingRights::BlackKingSide as u8;
        }
        if fen_castling.contains('q') {
            castling_rights |= CastlingRights::BlackQueenSide as u8;
        }

        self.castling_rights = castling_rights;
    }

    pub fn is_white_active(&self) -> bool {
        true //todo
    }

    pub fn is_path_clear(&self, start_square: u16, end_square: u16) -> bool {
        let all_pieces = self.get_all_pieces();

        let (start, end) = if start_square < end_square {
            (start_square + 1, end_square)
        } else {
            (end_square + 1, start_square)
        };

        for square in start..end {
            if all_pieces.is_set(square as usize) {
                return false;
            }
        }
        true
    }

    pub fn get_opponent_pieces(&self, is_white: bool) -> Bitboard {
        if is_white {
            self.black
        } else {
            self.white
        }
    }

    pub fn set_from_fen(&mut self, fen: &str) {
        FENParser::set_board_from_fen(self, fen);
    }

    pub fn set_piece_at_square(&mut self, square: u16, piece_type: PieceType, is_white: bool) {
        let square_bb = Bitboard::from_square_index(square);
        self.set_piece_bitboard(piece_type, square_bb, is_white);
    }

    pub fn get_all_pieces(&self) -> Bitboard {
        self.get_white() | (self.get_black())
    }

    pub fn get_all_free_squares(&self) -> Bitboard {
        !(self.get_all_pieces())
    }

    pub fn get_last_move(&self) -> Option<ChessMove> {
        self.move_history
            .last()
            .map(|(chess_move, _)| chess_move.clone())
    }

    pub fn undo_move(&mut self) {
        if let Some((chess_move, prev_castling_rights)) = self.move_history.pop() {
            let target_square = chess_move.target_square();
            let start_square = chess_move.start_square();
            let target_square_bb = Bitboard::from_square_index(target_square);
            let start_square_bb = Bitboard::from_square_index(start_square);
            self.castling_rights = prev_castling_rights;

            // Undo the move for the king or other pieces
            if let Some(piece) = chess_move.chess_piece {
                self.update_piece_bitboard(piece.piece_type(), target_square_bb, start_square_bb);
                self.update_color_bitboard(piece.is_white(), target_square_bb, start_square_bb);

                // Check if the move was a castling move and undo the rook's move
                if chess_move.has_flag(ChessMove::CASTLE_FLAG) {
                    // Determine rook's original and castled squares based on the castling type
                    let (rook_start_square, rook_target_square) =
                        match (start_square, target_square) {
                            (4, 6) => (5, 7),     // White kingside castling
                            (4, 2) => (3, 0),     // White queenside castling
                            (60, 62) => (61, 63), // Black kingside castling
                            (60, 58) => (59, 56), // Black queenside castling
                            _ => panic!("Invalid castling move"),
                        };

                    // Move the rook back
                    let rook_start_square_bb =
                        Bitboard::from_square_index(rook_start_square as u16);
                    let rook_target_square_bb =
                        Bitboard::from_square_index(rook_target_square as u16);
                    self.update_piece_bitboard(
                        PieceType::Rook,
                        rook_target_square_bb,
                        rook_start_square_bb,
                    );
                    self.update_color_bitboard(
                        piece.is_white(),
                        rook_target_square_bb,
                        rook_start_square_bb,
                    );
                }
            }
            // Restore the captured piece, if there was one
            if let Some(captured_piece) = chess_move.capture {
                // If the move was an en passant capture, the captured pawn's location differs from the target square
                if chess_move.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG) {
                    // Calculate the original position of the captured pawn
                    let captured_pawn_square = if captured_piece.is_white() {
                        target_square + 8
                    } else {
                        target_square - 8
                    };
                    let captured_pawn_bb = Bitboard::from_square_index(captured_pawn_square);

                    self.set_piece_bitboard(
                        captured_piece.piece_type(),
                        captured_pawn_bb,
                        captured_piece.is_white(),
                    );
                } else {
                    // For regular captures, just place the piece back on the target square
                    self.set_piece_bitboard(
                        captured_piece.piece_type(),
                        target_square_bb,
                        captured_piece.is_white(),
                    );
                }
            }
        } else {
            println!("No move to undo");
        }
    }

    pub fn get_king(&self, is_white: bool) -> Bitboard {
        if is_white {
            self.kings & self.white
        } else {
            self.kings & self.black
        }
    }

    pub fn make_move(&mut self, chess_move: &mut ChessMove) -> bool {
        let start_square = chess_move.start_square();
        let target_square = chess_move.target_square();
        let start_square_bb = Bitboard::from_square_index(start_square);
        let target_square_bb = Bitboard::from_square_index(target_square);

        let is_white = self.white.is_set(start_square as usize);

        if let Some(piece_type) = self.get_piece_type(start_square) {
            chess_move.set_piece(ChessPiece::new(piece_type, is_white));

            // Check for en passant
            if piece_type == PieceType::Pawn
                && chess_move.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG)
            {
                // Calculate the position of the captured pawn
                let captured_pawn_square = if is_white {
                    target_square - 8
                } else {
                    target_square + 8
                };

                if let Some(captured_pawn) = self.get_piece_at_square(captured_pawn_square) {
                    chess_move.set_capture(captured_pawn);
                    // Remove the captured pawn
                    let captured_pawn_bb = Bitboard::from_square_index(captured_pawn_square);
                    self.clear_piece_bitboard(PieceType::Pawn, captured_pawn_bb, !is_white);
                } else {
                    chess_move.clear_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG);
                }
            } else if let Some(captured_piece) = self.get_piece_at_square(target_square) {
                if captured_piece.is_white() == is_white {
                    println!(
                        "Invalid move: target square is occupied by a piece of the same color"
                    );
                    return false;
                }
                chess_move.set_capture(captured_piece);
                self.clear_piece_bitboard(captured_piece.piece_type(), target_square_bb, !is_white);
            } else {
                // println!("No piece captured");
            }

            // store history before altering castling rights!
            self.move_history
                .push((chess_move.clone(), self.castling_rights));

            if chess_move.has_flag(ChessMove::CASTLE_FLAG) {
                // Determine rook's initial and final positions based on the castling type
                let (rook_start_square, rook_target_square) = match target_square {
                    6 => (7, 5),    // White kingside castling
                    2 => (0, 3),    // White queenside castling
                    62 => (63, 61), // Black kingside castling
                    58 => (56, 59), // Black queenside castling
                    _ => panic!("Invalid castling move"),
                };

                // Move the rook
                let rook_start_square_bb = Bitboard::from_square_index(rook_start_square);
                let rook_target_square_bb = Bitboard::from_square_index(rook_target_square);
                self.update_piece_bitboard(
                    PieceType::Rook,
                    rook_start_square_bb,
                    rook_target_square_bb,
                );
                self.update_color_bitboard(is_white, rook_start_square_bb, rook_target_square_bb);

                // Update castling rights
                self.castling_rights &= if is_white {
                    !(CastlingRights::WhiteKingSide as u8 | CastlingRights::WhiteQueenSide as u8)
                } else {
                    !(CastlingRights::BlackKingSide as u8 | CastlingRights::BlackQueenSide as u8)
                };
            }

            // Update the piece's bitboard
            self.update_piece_bitboard(piece_type, start_square_bb, target_square_bb);

            // Update the color bitboards
            self.update_color_bitboard(is_white, start_square_bb, target_square_bb);
        }
        true
    }

    fn update_piece_bitboard(
        &mut self,
        piece_type: PieceType,
        start_square_bb: Bitboard,
        target_square_bb: Bitboard,
    ) {
        match piece_type {
            PieceType::Pawn => self.pawns = self.pawns ^ (start_square_bb | target_square_bb),
            PieceType::Knight => self.knights = self.knights ^ (start_square_bb | target_square_bb),
            PieceType::Bishop => self.bishops = self.bishops ^ (start_square_bb | target_square_bb),
            PieceType::Rook => self.rooks = self.rooks ^ (start_square_bb | target_square_bb),
            PieceType::Queen => self.queens = self.queens ^ (start_square_bb | target_square_bb),
            PieceType::King => self.kings = self.kings ^ (start_square_bb | target_square_bb),
            _ => {}
        }
    }

    fn set_piece_bitboard(&mut self, piece_type: PieceType, square_bb: Bitboard, is_white: bool) {
        match piece_type {
            PieceType::Pawn => self.pawns |= square_bb,
            PieceType::Knight => self.knights |= square_bb,
            PieceType::Bishop => self.bishops |= square_bb,
            PieceType::Rook => self.rooks |= square_bb,
            PieceType::Queen => self.queens |= square_bb,
            PieceType::King => self.kings |= square_bb,
            _ => {
                println!("No piece to set {}", piece_type as u8);
            }
        }

        if is_white {
            self.white |= square_bb;
        } else {
            self.black |= square_bb;
        }
    }

    fn clear_piece_bitboard(&mut self, piece_type: PieceType, square_bb: Bitboard, is_white: bool) {
        match piece_type {
            PieceType::Pawn => self.pawns &= !square_bb,
            PieceType::Knight => self.knights &= !square_bb,
            PieceType::Bishop => self.bishops &= !square_bb,
            PieceType::Rook => self.rooks &= !square_bb,
            PieceType::Queen => self.queens &= !square_bb,
            PieceType::King => self.kings &= !square_bb,
            _ => {
                println!("No piece to clear {}", piece_type as u8);
            }
        }

        if is_white {
            self.white &= !square_bb;
        } else {
            self.black &= !square_bb;
        }
    }

    fn update_color_bitboard(
        &mut self,
        is_white: bool,
        start_square_bb: Bitboard,
        target_square_bb: Bitboard,
    ) {
        if is_white {
            self.white = self.white ^ (start_square_bb | target_square_bb);
        } else {
            self.black = self.black ^ (start_square_bb | target_square_bb);
        }
    }

    pub fn is_square_white(&self, index: u16) -> bool {
        if self.white.is_set(index as usize) {
            true
        } else {
            false
        }
    }

    pub fn get_piece_type(&self, index: u16) -> Option<PieceType> {
        if self.pawns.is_set(index as usize) {
            Some(PieceType::Pawn)
        } else if self.knights.is_set(index as usize) {
            Some(PieceType::Knight)
        } else if self.bishops.is_set(index as usize) {
            Some(PieceType::Bishop)
        } else if self.rooks.is_set(index as usize) {
            Some(PieceType::Rook)
        } else if self.queens.is_set(index as usize) {
            Some(PieceType::Queen)
        } else if self.kings.is_set(index as usize) {
            Some(PieceType::King)
        } else {
            None
        }
    }

    pub fn get_piece_at_square(&self, index: u16) -> Option<ChessPiece> {
        let is_white = if self.white.is_set(index as usize) {
            true
        } else {
            false
        };
        if (self.pawns | self.knights | self.bishops | self.rooks | self.queens | self.kings)
            .is_set(index as usize)
        {
            Some(ChessPiece::new(
                self.get_piece_type(index).unwrap(),
                is_white,
            ))
        } else {
            None
        }
    }

    pub fn get_pawns(&self) -> Bitboard {
        self.pawns
    }

    pub fn get_rooks(&self) -> Bitboard {
        self.rooks
    }

    pub fn get_bishops(&self) -> Bitboard {
        self.bishops
    }

    pub fn get_knights(&self) -> Bitboard {
        self.knights
    }

    pub fn get_queens(&self) -> Bitboard {
        self.queens
    }

    pub fn get_kings(&self) -> Bitboard {
        self.kings
    }

    pub fn get_white(&self) -> Bitboard {
        self.white
    }

    pub fn get_black(&self) -> Bitboard {
        self.black
    }

    pub fn get_castling_rights(&self) -> String {
        let mut result = String::new();
        if self.castling_rights & 0b1000 != 0 {
            result.push('K');
        }
        if self.castling_rights & 0b0100 != 0 {
            result.push('Q');
        }
        if self.castling_rights & 0b0010 != 0 {
            result.push('k');
        }
        if self.castling_rights & 0b0001 != 0 {
            result.push('q');
        }
        result
    }

    // Determine the character for the piece on a given square
    pub fn get_piece_character(&self, index: usize) -> char {
        let piece_types = [
            (self.pawns, 'p'),
            (self.knights, 'n'),
            (self.bishops, 'b'),
            (self.rooks, 'r'),
            (self.queens, 'q'),
            (self.kings, 'k'),
        ];

        for &(piece_bitboard, piece_char) in &piece_types {
            if piece_bitboard.is_set(index) {
                let is_white = self.white.is_set(index);
                let is_black = self.black.is_set(index);

                // Check if the piece is white or black, then return the appropriate character
                if is_white {
                    return piece_char.to_ascii_uppercase();
                } else if is_black {
                    return piece_char.to_ascii_lowercase();
                }
            }
        }

        '.' // Empty square
    }

    // Generate a 2D array representation of the board
    pub fn generate_board_representation(&self) -> [[char; 8]; 8] {
        let mut board_repr = [['.'; 8]; 8];

        for i in 0..64 {
            let row = 7 - i / 8; // Flip row index for printing
            let col = i % 8;
            board_repr[row][col] = self.get_piece_character(i);
        }

        board_repr
    }

    pub fn print_board(&self) {
        let board_repr = self.generate_board_representation();
        println!("     a   b   c   d   e   f   g   h");
        // Print top border
        println!("   +---+---+---+---+---+---+---+---+");

        for (i, row) in board_repr.iter().enumerate() {
            // Start the row with a border
            print!("{}  |", 8 - i);

            // Print each square in the row
            for &square in row {
                // Use a space for empty squares ('.') for clarity
                let square_repr = if square == '.' { ' ' } else { square };
                print!(" {} |", square_repr);
            }

            // Print row number (8 to 1, descending)
            println!("  {}", 8 - i);

            // Print row border
            println!("   +---+---+---+---+---+---+---+---+");
        }

        // Print column identifiers
        println!("     a   b   c   d   e   f   g   h");
        println!("Castling rights: {}", self.get_castling_rights());
    }

    // Print the board to the console
    pub fn print_board_simple(&self) {
        let board_repr = self.generate_board_representation();

        for row in board_repr.iter() {
            for &square in row.iter() {
                print!("{} ", square);
            }
            println!(); // Move to the next line after printing a row
        }
    }
}
