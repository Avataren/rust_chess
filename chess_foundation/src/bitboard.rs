use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, Not};

/// Represents a chessboard using a 64-bit integer, where each bit corresponds to a square.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    const TOP_EDGE: Bitboard =
        Bitboard(0b01111110_00000000_00000000_00000000_00000000_00000000_00000000_00000000);
    const BOTTOM_EDGE: Bitboard =
        Bitboard(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_01111110);
    const LEFT_EDGE: Bitboard =
        Bitboard(0b00000000_10000000_10000000_10000000_10000000_10000000_10000000_00000000);
    const RIGHT_EDGE: Bitboard =
        Bitboard(0b00000000_00000001_00000001_00000001_00000001_00000001_00000001_00000000);

    /// Creates a new, empty bitboard.
    pub fn new() -> Self {
        Bitboard(0)
    }

    pub fn max() -> Self {
        Bitboard(0xFFFFFFFFFFFFFFFF)
    }

    pub fn from_square_index(index: u16) -> Self {
        Bitboard(1 << index)
    }

    pub fn from_edges() -> Self {
        Self::TOP_EDGE | Self::BOTTOM_EDGE | Self::LEFT_EDGE | Self::RIGHT_EDGE
    }

    pub fn edge_mask(piece_position: Bitboard) -> Bitboard {
        match piece_position {
            p if (p & Self::TOP_EDGE).0 != 0 => Self::TOP_EDGE,
            p if (p & Self::BOTTOM_EDGE).0 != 0 => Self::BOTTOM_EDGE,
            p if (p & Self::LEFT_EDGE).0 != 0 => Self::LEFT_EDGE,
            p if (p & Self::RIGHT_EDGE).0 != 0 => Self::RIGHT_EDGE,
            _ => Bitboard::max(), // No edges excluded if the rook is not on an edge
        }
    }

    pub fn count_ones(&self) -> u32 {
        self.0.count_ones()
    }

    pub fn default() -> Self {
        Bitboard(0)
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    pub fn contains_square(&self, index: i32) -> bool {
        self.is_set(index as usize)
    }

    // clear the least significant bit and return its index
    pub fn pop_lsb(&mut self) -> usize {
        let lsb = self.0.trailing_zeros() as usize;
        self.clear_bit(lsb);
        lsb
    }
    /// Sets a bit at the given index (0-63), where 0 is the least significant bit.
    pub fn set_bit(&mut self, index: usize) {
        // if (index > 63) {
        //     panic!("Index out of range");
        // }
        self.0 |= 1 << index;
    }

    /// Clears a bit at the given index (0-63).
    pub fn clear_bit(&mut self, index: usize) {
        self.0 &= !(1 << index);
    }

    /// Checks if a bit at the given index (0-63) is set.
    pub fn is_set(&self, index: usize) -> bool {
        (self.0 & (1 << index)) != 0
    }

    pub fn and(&self, other: Bitboard) -> Bitboard {
        Bitboard(self.0 & other.0)
    }

    pub fn or(&self, other: Bitboard) -> Bitboard {
        Bitboard(self.0 | other.0)
    }

    pub fn xor(&self, other: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ other.0)
    }

    pub fn not(&self) -> Bitboard {
        Bitboard(!self.0)
    }

    pub fn set_from_board_array(&mut self, board: &Vec<u8>) {
        let mut bitboard = Bitboard::new();
        for i in 0..64 {
            if board[i] != 0 {
                bitboard.set_bit(i);
            }
        }
        *self = bitboard;
    }

    pub fn from_board_array(board: &Vec<u8>) -> Self {
        let mut bitboard = Bitboard::new();
        for (i, &piece) in board.iter().enumerate() {
            if piece != 0 {
                bitboard.set_bit(i);
            }
        }
        bitboard
    }
    pub fn print_bitboard(&self) {
        println!("dec value: {:}", self.0);
        println!("hex value: {:x}", self.0);
        println!("binary value: ");
        for rank in (0..8).rev() {
            for file in 0..8 {
                let i = rank * 8 + file; // Calculate index from top to bottom, left to right
                print!("{}", if self.is_set(i) { "1" } else { "0" });
            }
            println!(); // New line at the end of each rank
        }
        println!(); // Ensure a newline at the end of the output
    }
}

impl BitAnd for Bitboard {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl BitOrAssign for Bitboard {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitXor for Bitboard {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl Not for Bitboard {
    type Output = Self;

    fn not(self) -> Self::Output {
        Bitboard(!self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitboard_operations() {
        let mut bitboard1 = Bitboard::new();
        let mut bitboard2 = Bitboard::new();

        bitboard1.set_bit(0);
        assert_eq!(bitboard1.is_set(0), true);
        assert_eq!(bitboard1, Bitboard(0b1));
        bitboard2.set_bit(1);
        println!("{:?}", bitboard1.or(bitboard2));

        assert_eq!(bitboard1.and(bitboard2), Bitboard(0b00));
        assert_eq!(bitboard1.or(bitboard2), Bitboard(0b11));
    }

    #[test]
    fn test_color_exctraction() {
        let bitboard = Bitboard(0b1111_1111_1111_1111);
        let white = bitboard.and(Bitboard(0b0000_0000_1111_1111));
        let black = bitboard.and(Bitboard(0b1111_1111_0000_0000));
        assert_eq!(white, Bitboard(0b0000_0000_1111_1111));
        assert_eq!(black, Bitboard(0b1111_1111_0000_0000));
    }

    #[test]
    fn test_pop_lsb() {
        let mut bitboard = Bitboard(0b1000_0000_0000_0000);
        assert_eq!(bitboard.pop_lsb(), 15);
        assert_eq!(bitboard.0, 0);

        bitboard = Bitboard(0b0000_0000_0000_0001);
        assert_eq!(bitboard.pop_lsb(), 0);
        assert_eq!(bitboard.0, 0);

        bitboard = Bitboard(0b0000_0000_0000_1111);
        assert_eq!(bitboard.pop_lsb(), 0);
        assert_eq!(bitboard.0, 0b0000_0000_0000_1110);
        assert_eq!(bitboard.pop_lsb(), 1);
        assert_eq!(bitboard.0, 0b0000_0000_0000_1100);
        assert_eq!(bitboard.pop_lsb(), 2);
        assert_eq!(bitboard.0, 0b0000_0000_0000_1000);
        assert_eq!(bitboard.pop_lsb(), 3);
        assert_eq!(bitboard.0, 0);
    }
}
