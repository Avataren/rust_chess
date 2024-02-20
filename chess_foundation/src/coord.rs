#[derive(Debug, Clone, Copy, Eq)]
pub struct Coord {
    pub file_index: i32,
    pub rank_index: i32,
}

impl Coord {

    // Primary constructor that directly takes file_index and rank_index
    pub const fn new(file_index: i32, rank_index: i32) -> Self {
        Self { file_index, rank_index }
    }

    // Constructor from square_index, assuming BoardHelper functions are implemented
    pub  fn from_square_index(square_index: i32) -> Self {
        Self {
            file_index: board_helper::file_index(square_index),
            rank_index: board_helper::rank_index(square_index),
        }
    }

    // Method to check if the square is light
    pub fn is_light_square(&self) -> bool {
        (self.file_index + self.rank_index) % 2 != 0
    }

    // Check if the Coord is a valid square
    pub fn is_valid_square(&self) -> bool {
        self.file_index >= 0 && self.file_index < 8 && self.rank_index >= 0 && self.rank_index < 8
    }

    // Assuming BoardHelper::index_from_coord is implemented
    pub fn square_index(&self) -> i32 {
        board_helper::index_from_coord(self)
    }
}

// Implementing PartialEq to enable comparisons
impl PartialEq for Coord {
    fn eq(&self, other: &Self) -> bool {
        self.file_index == other.file_index && self.rank_index == other.rank_index
    }
}

// Implementing PartialOrd to provide a basic comparison
impl PartialOrd for Coord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Coord {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.file_index == other.file_index {
            self.rank_index.cmp(&other.rank_index)
        } else {
            self.file_index.cmp(&other.file_index)
        }
    }
}

// Implementing operator overloads for Coord
use std::{
    cmp::Ordering,
    ops::{Add, Mul, Sub},
};

use crate::board_helper;

impl Add for Coord {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::new(
            self.file_index + other.file_index,
            self.rank_index + other.rank_index,
        )
    }
}

impl Sub for Coord {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self::new(
            self.file_index - other.file_index,
            self.rank_index - other.rank_index,
        )
    }
}

impl Mul<i32> for Coord {
    type Output = Self;

    fn mul(self, m: i32) -> Self::Output {
        Self::new(self.file_index * m, self.rank_index * m)
    }
}

impl Mul<Coord> for i32 {
    type Output = Coord;

    fn mul(self, a: Coord) -> Self::Output {
        Coord::new(a.file_index * self, a.rank_index * self)
    }
}
