use crate::coord::Coord;

pub fn file_index(square_index: u16) -> u16 {
    square_index & 0b000111
}

pub fn rank_index(square_index: u16) -> u16 {
    square_index >> 3
}

pub fn index_from_coord(coord: &Coord) -> i32 {
    coord.rank_index * 8 + coord.file_index
}

pub fn square_index_to_board_row_col(index: i32) -> (usize, usize) {
    let col = index % 8;
    let row = 7 - index / 8;
    (row as usize, col as usize)
}
pub fn board_row_col_to_square_index(row: usize, col: usize) -> u16 {
    ((7 - row) * 8 + col) as u16
}
