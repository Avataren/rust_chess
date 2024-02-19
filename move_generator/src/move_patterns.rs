use crate::Bitboard;

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
        move_pattern.clear_bit(square);
        move_patterns.push(move_pattern);
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
        move_pattern.clear_bit(square);
        move_patterns.push(move_pattern);
    }

    move_patterns
}
