
fn san_to_target_square(&self, san: &str) -> Option<u16> {
    // Assuming the last two characters of the SAN string represent the target square
    // e.g., "e4", "h8", etc.
    let len = san.len();
    if len < 2 {
        return None;
    }

    let file = san.chars().nth(len - 2)?;
    let rank = san.chars().nth(len - 1)?.to_digit(10)?;

    if file < 'a' || file > 'h' || rank < 1 || rank > 8 {
        return None;
    }

    // Convert file and rank to 0-based indices
    let file_index = file as u16 - 'a' as u16; // 0 to 7
    let rank_index = rank as u16 - 1; // 0 to 7

    // Convert to square index, assuming a8 is 0, b8 is 1, ..., a1 is 56, h1 is 63
    Some((7 - rank_index) * 8 + file_index)
}

fn find_start_square(
    &self,
    piece_type: PieceType,
    target_square: u16,
    is_capture: bool,
) -> Option<u16> {
    // Get all pieces of the specified type for the current player
    let pieces_bitboard = match piece_type {
        PieceType::Pawn => {
            self.pawns
                & if self.white_is_active {
                    self.white
                } else {
                    self.black
                }
        }
        PieceType::Knight => {
            self.knights
                & if self.white_is_active {
                    self.white
                } else {
                    self.black
                }
        }
        PieceType::Bishop => {
            self.bishops
                & if self.white_is_active {
                    self.white
                } else {
                    self.black
                }
        }
        PieceType::Rook => {
            self.rooks
                & if self.white_is_active {
                    self.white
                } else {
                    self.black
                }
        }
        PieceType::Queen => {
            self.queens
                & if self.white_is_active {
                    self.white
                } else {
                    self.black
                }
        }
        PieceType::King => {
            self.kings
                & if self.white_is_active {
                    self.white
                } else {
                    self.black
                }
        }
        _ => Bitboard(0),
    };

    let conductor = PieceConductor::new();
    // Iterate over each piece of the given type
    for start_square in 0..64 {
        if pieces_bitboard.contains_square(start_square as i32) {
            // Generate pseudo-legal moves for this piece
            let pseudo_legal_moves = get_pseudo_legal_move_list_from_square(
                start_square as u16,
                self,
                &conductor, 
                self.white_is_active,
            );

            // Filter moves to find one that matches the target square and capture status
            for chess_move in pseudo_legal_moves {
                if chess_move.target_square() == target_square {
                    // Check if the move is a capture if required
                    if !is_capture || chess_move.capture.is_some() {
                        return Some(start_square as u16);
                    }
                }
            }
        }
    }

    None // No matching start square found
}

pub fn handle_san_castling(&self, san: &str) -> Option<ChessMove> {
    let is_white = self.is_white_active();
    let (king_start_square, king_target_square, rook_start_square, rook_target_square) =
        match (san, is_white) {
            ("O-O", true) => (4, 6, 7, 5),        // White king-side castling
            ("O-O-O", true) => (4, 2, 0, 3),      // White queen-side castling
            ("O-O", false) => (60, 62, 63, 61),   // Black king-side castling
            ("O-O-O", false) => (60, 58, 56, 59), // Black queen-side castling
            _ => return None,                     // Invalid castling notation
        };

    // Create a ChessMove for the king's move in castling
    let mut castling_move = ChessMove::new_with_flag(
        king_start_square,
        king_target_square,
        ChessMove::CASTLE_FLAG,
    );

    Some(castling_move)
}

fn char_to_piece_type(&self, c: char) -> PieceType {
    match c {
        'P' => PieceType::Pawn,
        'N' => PieceType::Knight,
        'B' => PieceType::Bishop,
        'R' => PieceType::Rook,
        'Q' => PieceType::Queen,
        'K' => PieceType::King,
        _ => panic!("Invalid piece type character: {}", c),
    }
}

pub fn get_move_from_san(&self, san: &str) -> Option<ChessMove> {
    // Parse SAN to identify piece, capture, target square, and special moves
    let mut piece_type = PieceType::Pawn; // Assume Pawn by default
    let mut target_square = 0;
    let mut flag = ChessMove::NO_FLAG;
    let mut promotion = None;

    // Handle castling
    if san == "O-O" || san == "O-O-O" {
        return self.handle_san_castling(san);
    }

    // Handle other moves
    let mut chars = san.chars();
    let first_char = chars.next().unwrap();

    // Check if the first character specifies a piece
    if first_char.is_uppercase() {
        piece_type = self.char_to_piece_type(first_char);
    } else {
        // If not, it's a pawn move, so we rewind the iterator
        chars = san.chars();
    }

    // Collect remaining characters for further processing
    let rest: String = chars.collect();

    // Handle captures, promotions, and target square
    if let Some(capture_index) = rest.find('x') {
        // Logic to handle captures
        //todo!()
    }

    // Handle promotion
    if let Some(promotion_index) = rest.find('=') {
        // Logic to handle promotion
        //todo!()
    }

    // Determine target square from SAN
    if let Some(ts) = self.san_to_target_square(&rest) {
        target_square = ts;
    } else {
        return None;
    }

    // Find the start square based on the piece type, target square, and the current board state
    if let Some(start_square) =
        Self::find_start_square(piece_type, target_square, san.contains('x'))
    {
        let mut chess_move = ChessMove::new(start_square, target_square);

        // Set special flags if necessary (e.g., en passant, promotion)
        chess_move.set_flag(flag);

        // Handle promotion
        if let Some(promo) = promotion {
            chess_move =
                ChessMove::new_capture_with_flag(start_square, None, target_square, flag);
            // Logic to set the promotion piece type on `chess_move`
        }

        Some(chess_move)
    } else {
        None
    }
}
