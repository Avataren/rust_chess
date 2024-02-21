use chess_foundation::{bitboard::Bitboard, ChessMove};

pub struct ChessBoard {
    white: Bitboard,
    black: Bitboard,
    pawns: Bitboard,
    knights: Bitboard,
    bishops: Bitboard,
    rooks: Bitboard,
    queens: Bitboard,
    kings: Bitboard,

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
            castling_rights: 0b00001111,
        }
    }

    pub fn make_move(&self, chess_move: ChessMove) {
        let start_square = chess_move.start_square();
        let target_square = chess_move.target_square();

        //do I need to look up pieces here, or when generating move?
        //when generating move!
        //I can at least get it when making move on the board as player
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
