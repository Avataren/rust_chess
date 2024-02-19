/// Represents a chessboard using a 64-bit integer, where each bit corresponds to a square.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// Creates a new, empty bitboard.
    pub fn new() -> Self {
        Bitboard(0)
    }

    /// Sets a bit at the given index (0-63), where 0 is the least significant bit.
    pub fn set_bit(&mut self, index: usize) {
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

    pub fn print_bitboard(&self) {
        println! ("hex value: {:x}", self.0);
        println! ("binary value: ");
        for i in 0..64 {
            if i % 8 == 0 && i != 0 {
                println!();
            }
            print!("{}", if self.is_set(i) { "1" } else { "0" });
        }
        println!(); // Ensure a newline at the end of the output
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
    
}
