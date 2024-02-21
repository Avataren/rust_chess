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
    all_pieces: Bitboard,
    castling_rights: u8,
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
            all_pieces: Bitboard(0xFFFF_0000_0000_FFFF),
            castling_rights: 0b00001111,
        }
    }

    pub fn make_move(&mut self, chess_move: ChessMove) {
        let start_square_bb = Bitboard::from_square_index(chess_move.start_square());
        let target_square_bb = Bitboard::from_square_index(chess_move.target_square());

        let is_white = self.white.is_set(chess_move.start_square() as usize);
        let piece_type = self.get_piece_type(chess_move.start_square());
        //let piece = ChessPiece::new(piece_type, is_white);
        
        // Update the piece's bitboard
        self.update_piece_bitboard(piece_type, start_square_bb, target_square_bb);

        // If there's a capture, clear the target square bit in the captured piece's bitboard
        if let Some(captured_piece_type) = self.get_captured_piece_type(chess_move.target_square(), is_white) {
            self.clear_piece_bitboard(captured_piece_type, target_square_bb, !is_white);
        }

        // Update the color bitboards
        self.update_color_bitboard(is_white, start_square_bb, target_square_bb);        

    }

    fn get_captured_piece_type(&self, target_square: u16, is_white: bool) -> Option<PieceType> {
        let piece_type = self.get_piece_type(target_square);
        if piece_type != PieceType::None {
            if (is_white && self.white.is_set(target_square as usize)) || (!is_white && self.black.is_set(target_square as usize)) {
                return Some(piece_type);
            }
        }
        None
    }

    fn update_piece_bitboard(&mut self, piece_type: PieceType, start_square_bb: Bitboard, target_square_bb: Bitboard) {
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

    fn clear_piece_bitboard(&mut self, piece_type: PieceType, square_bb: Bitboard, is_white: bool) {
        match piece_type {
            PieceType::Pawn => self.pawns &= !square_bb,
            PieceType::Knight => self.knights &= !square_bb,
            PieceType::Bishop => self.bishops &= !square_bb,
            PieceType::Rook => self.rooks &= !square_bb,
            PieceType::Queen => self.queens &= !square_bb,
            PieceType::King => self.kings &= !square_bb,
            _ => {}
        }

        if is_white {
            self.white &= !square_bb;
        } else {
            self.black &= !square_bb;
        }
    }

    fn update_color_bitboard(&mut self, is_white: bool, start_square_bb: Bitboard, target_square_bb: Bitboard) {
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

    pub fn get_piece_type(&self, index: u16) -> PieceType {
        let mut piece_type = PieceType::None;
        if self.pawns.is_set(index as usize) {
            piece_type = PieceType::Pawn;
        } else if self.knights.is_set(index as usize) {
            piece_type = PieceType::Knight;
        } else if self.bishops.is_set(index as usize) {
            piece_type = PieceType::Bishop;
        } else if self.rooks.is_set(index as usize) {
            piece_type = PieceType::Rook;
        } else if self.queens.is_set(index as usize) {
            piece_type = PieceType::Queen;
        } else if self.kings.is_set(index as usize) {
            piece_type = PieceType::King;
        }
        piece_type
    }

    // pub fn  MovePiece(int piece, int startSquare, int targetSquare)
    // {
    //     BitBoardUtility.ToggleSquares(ref PieceBitboards[piece], startSquare, targetSquare);
    //     BitBoardUtility.ToggleSquares(ref ColourBitboards[MoveColourIndex], startSquare, targetSquare);

    //     allPieceLists[piece].MovePiece(startSquare, targetSquare);
    //     Square[startSquare] = Piece.None;
    //     Square[targetSquare] = piece;
    // }

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
