use chess_foundation::Bitboard;

pub fn get_rook_move_patterns() -> Vec<Bitboard> {
    let mut move_patterns = Vec::<Bitboard>::new();

    // Precompute row and column masks
    let mut row_masks = vec![Bitboard(0); 8];
    let mut col_masks = vec![Bitboard(0); 8];
    for row in 0..8 {
        for col in 0..8 {
            row_masks[row].set_bit(row * 8 + col);
            col_masks[col].set_bit(col + row * 8);
        }
    }

    for square in 0..64 {
        let row = square / 8;
        let col = square % 8;
        // Start with the union of the row and column masks
        let mut move_pattern = row_masks[row].or(col_masks[col]);
        // Clear the bit for the square itself, as the rook can't move to its current position
        move_pattern.clear_bit(square); // is this desired?

        let special_case_mask = !Bitboard::edge_mask(Bitboard::from_square_index(square as u16));
        let mask = !(Bitboard::from_edges() & special_case_mask);

        move_patterns.push(move_pattern & mask);
    }

    move_patterns
}

pub fn get_bishop_move_patterns() -> Vec<Bitboard> {
    let mut move_patterns = Vec::<Bitboard>::new();

    // Precompute diagonal and anti-diagonal masks
    let mut diag_masks = vec![Bitboard(0); 15]; // 15 diagonals on a chessboard
    let mut anti_diag_masks = vec![Bitboard(0); 15]; // 15 anti-diagonals

    for row in 0..8 {
        for col in 0..8 {
            let diag = row + col;
            let anti_diag = row + 7 - col;
            diag_masks[diag].set_bit(row * 8 + col);
            anti_diag_masks[anti_diag].set_bit(row * 8 + col);
        }
    }

    for square in 0..64 {
        let row = square / 8;
        let col = square % 8;
        let diag = row + col;
        let anti_diag = row + 7 - col;
        // Combine the masks for the main diagonal and anti-diagonal
        let mut move_pattern = diag_masks[diag].or(anti_diag_masks[anti_diag]);
        // Clear the bit for the square itself
        move_pattern.clear_bit(square); // is this desired?

        let special_case_mask = !Bitboard::edge_mask(Bitboard::from_square_index(square as u16));
        let mask = !(Bitboard::from_edges() & special_case_mask);

        move_patterns.push(move_pattern & mask);
    }

    move_patterns
}

pub fn get_knight_move_patterns() -> Vec<Bitboard> {
    let knight_moves = [-17, -15, -10, -6, 6, 10, 15, 17];

    let mut move_patterns = Vec::<Bitboard>::new();

    for square in 0..64 {
        let mut move_pattern = Bitboard(0);
        for &move_offset in &knight_moves {
            let to_square = square as i32 + move_offset;
            if to_square >= 0 && to_square < 64 {
                let to_row = to_square / 8;
                let to_col = to_square % 8;
                let row_diff = (to_row - (square / 8) as i32).abs();
                let col_diff = (to_col - (square % 8) as i32).abs();

                // Ensure the move stays within an L-shape
                if row_diff + col_diff == 3 && row_diff != 0 && col_diff != 0 {
                    move_pattern.set_bit(to_square as usize);
                }
            }
        }
        move_patterns.push(move_pattern);
    }

    move_patterns
}

pub fn get_king_move_patterns() -> Vec<Bitboard> {
    let king_moves = [-9, -8, -7, -1, 1, 7, 8, 9];

    let mut move_patterns = Vec::<Bitboard>::new();

    for square in 0..64 {
        let mut move_pattern = Bitboard(0);
        for &move_offset in &king_moves {
            let to_square = square as i32 + move_offset;
            if to_square >= 0 && to_square < 64 {
                let to_row = to_square / 8;
                let to_col = to_square % 8;
                let row_diff = (to_row - (square / 8) as i32).abs();
                let col_diff = (to_col - (square % 8) as i32).abs();

                // Ensure the move is within one square of the original position
                if row_diff <= 1 && col_diff <= 1 {
                    move_pattern.set_bit(to_square as usize);
                }
            }
        }
        move_patterns.push(move_pattern);
    }

    move_patterns
}

pub fn get_pawn_move_patterns() -> Vec<((Bitboard, Bitboard), (Bitboard, Bitboard))> {
    let mut move_patterns = Vec::<((Bitboard, Bitboard), (Bitboard, Bitboard))>::new();

    for square in 0..64 {
        let mut white_moves = Bitboard(0);
        let mut black_moves = Bitboard(0);
        let mut white_captures = Bitboard(0);
        let mut black_captures = Bitboard(0);

        let row = square / 8;
        let col = square % 8;

        // White pawn moves
        if row < 7 {
            // Normal move
            white_moves.set_bit(square + 8);

            // Initial double move
            if row == 1 {
                white_moves.set_bit(square + 16);
            }

            // Captures
            if col > 0 {
                white_captures.set_bit(square + 7);
            } // Left capture
            if col < 7 {
                white_captures.set_bit(square + 9);
            } // Right capture
        }

        // Black pawn moves
        if row > 0 {
            // Normal move
            black_moves.set_bit(square - 8);

            // Initial double move
            if row == 6 {
                black_moves.set_bit(square - 16);
            }

            // Captures
            if col > 0 {
                black_captures.set_bit(square - 9);
            } // Left capture
            if col < 7 {
                black_captures.set_bit(square - 7);
            } // Right capture
        }

        move_patterns.push(((white_moves, black_moves), (white_captures, black_captures)));
    }

    move_patterns
}
