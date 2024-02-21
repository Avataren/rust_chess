use crate::Coord;

pub const ROOK_DIRECTIONS: [Coord; 4] = [
    Coord::new(0, 1),
    Coord::new(0, -1),
    Coord::new(1, 0),
    Coord::new(-1, 0),
];
pub const BISHOP_DIRECTIONS: [Coord; 4] = [
    Coord::new(1, 1),
    Coord::new(1, -1),
    Coord::new(-1, 1),
    Coord::new(-1, -1),
];
