const PAWN_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, // 8th rank
    75, 75, 75, 75, 75, 75, 75, 75, // 7th rank
    10, 10, 40, 50, 50, 40, 10, 10, // 6th rank
    5, 5, 10, 25, 25, 10, 5, 5, // 5th rank
    0, 0, 0, 20, 20, 0, 0, 0, // 4th rank
    5, -5, -10, 0, 0, -10, -5, 5, // 3rd rank
    5, 10, 10, -20, -20, 10, 10, 5, // 2nd rank
    0, 0, 0, 0, 0, 0, 0, 0, // 1st rank
];

pub fn evaluate_pawn_position(square: usize, is_white: bool) -> i32 {
    if is_white {
        PAWN_TABLE[square] // Directly use the index for white
    } else {
        PAWN_TABLE[63 - square] // Flip index for black
    }
}
