use crate::coord::Coord;

pub fn file_index(square_index: i32) -> i32 {
    square_index & 0b000111
}

pub fn rank_index(square_index: i32) -> i32 {
    square_index >> 3
}

pub fn index_from_coord(coord: &Coord) -> i32 {
    coord.rank_index * 8 + coord.file_index
}
