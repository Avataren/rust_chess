use core::panic;

use chess_board::ChessBoard;
use chess_foundation::bitboard::Bitboard;
use chess_foundation::{ChessMove, Coord};
use rand::Rng;

use crate::magic_constants::{BISHOP_MAGICS, ROOK_MAGICS};
use crate::masks::{BISHOP_MASKS, ROOK_MASKS};
use crate::{get_king_move_patterns, get_knight_move_patterns};

pub struct Magic {
    //pub pawn_lut: Vec<Bitboard>,
    pub knight_lut: Vec<Bitboard>,
    pub king_lut: Vec<Bitboard>,
    rook_table: Vec<Vec<Bitboard>>,
    bishop_table: Vec<Vec<Bitboard>>,
}

impl Magic {
    const RBITS: [i32; 64] = [
        12, 11, 11, 11, 11, 11, 11, 12, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10,
        11, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10,
        10, 11, 11, 10, 10, 10, 10, 10, 10, 11, 12, 11, 11, 11, 11, 11, 11, 12,
    ];

    const BBITS: [i32; 64] = [
        6, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 7, 9, 9, 7,
        5, 5, 5, 5, 7, 9, 9, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5,
        5, 5, 5, 6,
    ];

    pub fn new() -> Self {
        let king_lut = get_king_move_patterns();
        let knight_lut = get_knight_move_patterns();

        let rook_table = Self::init_rook_table();
        let bishop_table = Self::init_bishop_table();

        Magic {
            knight_lut,
            king_lut,
            rook_table,
            bishop_table,
        }
    }

    pub fn print_masks() {
        println!("pub const ROOK_MASKS: [Bitboard; 64] = [");
        //for r in rook_masks.iter() {
        for square in 0..64 {
            println!("Bitboard(0x{:X}), ", Self::rmask(square));
        }
        println!("];");

        println!("pub const BISHOP_MASKS: [Bitboard; 64] = [");
        //for r in bishop_masks.iter() {
        for square in 0..64 {
            println!("Bitboard(0x{:X}), ", Self::bmask(square));
        }
        println!("\n];");
    }

    pub fn print_magics() {
        let mut bishop_magix = Vec::new();
        let mut rook_magix = Vec::new();

        // Fill the bishop_magix vector
        for square in 0..64 {
            let m = Self::find_magic(square, Self::BBITS[square as usize], true);
            bishop_magix.push(m);
        }

        // Fill the rook_magix vector
        for square in 0..64 {
            let m = Self::find_magic(square, Self::RBITS[square as usize], false);
            rook_magix.push(m);
        }

        // Format and print bishop_magix as a const array in uppercase hexadecimal
        println!("pub const BISHOP_MAGICS: [u64; 64] = [");
        for m in bishop_magix.iter() {
            print!("0x{:X}, ", m);
        }
        println!("\n];");

        // Format and print rook_magix as a const array in uppercase hexadecimal
        println!("pub const ROOK_MAGICS: [u64; 64] = [");
        for m in rook_magix.iter() {
            print!("0x{:X}, ", m);
        }
        println!("\n];");
    }

    fn rook_magic_index(square: usize, blockers: Bitboard) -> usize {
        let mask = ROOK_MASKS[square];
        let magic = ROOK_MAGICS[square];
        let shift = Self::RBITS[square as usize];
        ((blockers & mask).0.wrapping_mul(magic) >> (64 - shift)) as usize
    }

    fn bishop_magic_index(square: usize, blockers: Bitboard) -> usize {
        let mask = BISHOP_MASKS[square];
        let magic = BISHOP_MAGICS[square];
        let shift = Self::BBITS[square as usize];
        ((blockers & mask).0.wrapping_mul(magic) >> (64 - shift)) as usize
    }

    fn init_rook_table() -> Vec<Vec<Bitboard>> {
        let mut rook_table: Vec<Vec<Bitboard>> = vec![vec![Bitboard::default(); 4096]; 64];
        for square in 0..64 {
            let all_blocker_configs = Self::generate_blocker_bitboards(ROOK_MASKS[square]);
            for blockers in all_blocker_configs {
                let magic_index = Self::rook_magic_index(square as usize, blockers);
                let legal_move_bitboard =
                    Self::generate_legal_moves_from_blockers(square as u16, &blockers, true); // Assuming false for rooks
                rook_table[square as usize][magic_index] = legal_move_bitboard;
            }
        }
        rook_table
    }

    fn init_bishop_table() -> Vec<Vec<Bitboard>> {
        let mut bishop_table: Vec<Vec<Bitboard>> = vec![vec![Bitboard::default(); 1024]; 64];

        for square in 0..64 {
            let all_blocker_configs = Self::generate_blocker_bitboards(BISHOP_MASKS[square]);
            for blockers in all_blocker_configs {
                let magic_index = Self::bishop_magic_index(square as usize, blockers);
                let legal_move_bitboard =
                    Self::generate_legal_moves_from_blockers(square as u16, &blockers, false); // Assuming false for rooks
                bishop_table[square as usize][magic_index] = legal_move_bitboard;
            }
        }
        bishop_table
    }

    pub fn get_rook_moves(&self, square:u16, relevant_blockers: Bitboard, chess_board: &ChessBoard) -> Vec<ChessMove> {
        let magic_index = Self::rook_magic_index(square as usize, chess_board.get_all_pieces());
        let mut moves_bitboard = self.rook_table[square as usize][magic_index] & !relevant_blockers;
        let mut move_list = Vec::new();
        while !moves_bitboard.is_empty() {
            let target_square = moves_bitboard.pop_lsb();
            move_list.push(ChessMove::new(square, target_square as u16));
        }
        move_list
    }


    pub fn get_bishop_moves(&self, square:u16, relevant_blockers: Bitboard, chess_board: &ChessBoard) -> Vec<ChessMove> {
        let magic_index = Self::bishop_magic_index(square as usize, chess_board.get_all_pieces());
        let mut moves_bitboard = self.bishop_table[square as usize][magic_index] & !relevant_blockers;
        let mut move_list = Vec::new();
        while !moves_bitboard.is_empty() {
            let target_square = moves_bitboard.pop_lsb();
            move_list.push(ChessMove::new(square, target_square as u16));
        }
        move_list
    }

    pub fn get_knight_moves(
        &self,
        square: u16,
        friendly_pieces_bitboard: Bitboard,
    ) -> Vec<ChessMove> {
        let knight_moves_bitboard = self.knight_lut[square as usize];
        let valid_moves_bitboard = knight_moves_bitboard & !friendly_pieces_bitboard;
        let mut move_list = Vec::new();

        let mut moves_bitboard = valid_moves_bitboard;
        while !moves_bitboard.is_empty() {
            let target_square = moves_bitboard.pop_lsb();
            move_list.push(ChessMove::new(square, target_square as u16));
        }

        move_list
    }

    pub fn get_king_moves(
        &self,
        square: u16,
        friendly_pieces_bitboard: Bitboard,
    ) -> Vec<ChessMove> {
        let king_moves_bitboard = self.king_lut[square as usize];
        let valid_moves_bitboard = king_moves_bitboard & !friendly_pieces_bitboard;
        let mut move_list = Vec::new();

        let mut moves_bitboard = valid_moves_bitboard;
        while !moves_bitboard.is_empty() {
            let target_square = moves_bitboard.pop_lsb();
            move_list.push(ChessMove::new(square, target_square as u16));
        }

        move_list
    }

    fn random_uint64() -> u64 {
        let mut rng = rand::thread_rng();
        let u1: u64 = rng.gen::<u16>() as u64;
        let u2: u64 = rng.gen::<u16>() as u64;
        let u3: u64 = rng.gen::<u16>() as u64;
        let u4: u64 = rng.gen::<u16>() as u64;
        u1 | (u2 << 16) | (u3 << 32) | (u4 << 48)
    }

    fn random_uint64_fewbits() -> u64 {
        Self::random_uint64() & Self::random_uint64() & Self::random_uint64()
    }

    /// Transforms a 64-bit block using a magic number and the number of bits to use.
    fn transform(b: u64, magic: u64, bits: i32) -> i32 {
        (((b.wrapping_mul(magic)) >> (64 - bits)) & ((1 << bits) - 1) as u64) as i32
    }

    /// Counts the number of set bits in a 64-bit unsigned integer.
    fn count_1s(mut b: u64) -> i32 {
        let mut r = 0;
        while b != 0 {
            r += 1;
            b &= b - 1;
        }
        r
    }

    fn index_to_uint64(index: i32, bits: i32, mut mask: u64) -> u64 {
        let mut result: u64 = 0;
        let mut j: i32 = 0;
        for i in 0..bits {
            let mut bit = mask & !(mask - 1); // Isolate the lowest bit of the mask
            mask &= mask - 1; // Clear the lowest bit of the mask
            if (index & (1 << i)) != 0 {
                result |= bit;
            }
            while bit != 0 {
                bit >>= 1;
                j += 1;
            }
        }
        result
    }

    fn rmask(sq: i32) -> u64 {
        let mut result: u64 = 0;
        let rk = sq / 8;
        let fl = sq % 8;
        for r in (rk + 1)..7 {
            result |= 1 << (fl + r * 8);
        }
        for r in (1..rk).rev() {
            result |= 1 << (fl + r * 8);
        }
        for f in (fl + 1)..7 {
            result |= 1 << (f + rk * 8);
        }
        for f in (1..fl).rev() {
            result |= 1 << (f + rk * 8);
        }
        result
    }

    fn bmask(sq: i32) -> u64 {
        let mut result: u64 = 0;
        let rk = sq / 8;
        let fl = sq % 8;
        let mut r;
        let mut f;
        r = rk + 1;
        f = fl + 1;
        while r <= 6 && f <= 6 {
            result |= 1 << (f + r * 8);
            r += 1;
            f += 1;
        }
        r = rk + 1;
        f = fl - 1;
        while r <= 6 && f >= 1 {
            result |= 1 << (f + r * 8);
            r += 1;
            f -= 1;
        }
        r = rk - 1;
        f = fl + 1;
        while r >= 1 && f <= 6 {
            result |= 1 << (f + r * 8);
            r -= 1;
            f += 1;
        }
        r = rk - 1;
        f = fl - 1;
        while r >= 1 && f >= 1 {
            result |= 1 << (f + r * 8);
            r -= 1;
            f -= 1;
        }
        result
    }

    fn ratt(sq: i32, block: u64) -> u64 {
        let mut result: u64 = 0;
        let rk = sq / 8; // Rank
        let fl = sq % 8; // File

        // Positive rank direction
        for r in (rk + 1)..8 {
            result |= 1 << (fl + r * 8);
            if block & (1 << (fl + r * 8)) != 0 {
                break;
            }
        }

        // Negative rank direction
        for r in (0..rk).rev() {
            result |= 1 << (fl + r * 8);
            if block & (1 << (fl + r * 8)) != 0 {
                break;
            }
        }

        // Positive file direction
        for f in (fl + 1)..8 {
            result |= 1 << (f + rk * 8);
            if block & (1 << (f + rk * 8)) != 0 {
                break;
            }
        }

        // Negative file direction
        for f in (0..fl).rev() {
            result |= 1 << (f + rk * 8);
            if block & (1 << (f + rk * 8)) != 0 {
                break;
            }
        }

        result
    }

    fn batt(sq: i32, block: u64) -> u64 {
        let mut result: u64 = 0;
        let rk = sq / 8; // Rank
        let fl = sq % 8; // File

        // Diagonal: bottom left to top right
        let mut r = rk + 1;
        let mut f = fl + 1;
        while r < 8 && f < 8 {
            result |= 1 << (f + r * 8);
            if block & (1 << (f + r * 8)) != 0 {
                break;
            }
            r += 1;
            f += 1;
        }

        // Diagonal: top left to bottom right
        r = rk + 1;
        f = fl - 1;
        while r < 8 && f >= 0 {
            result |= 1 << (f + r * 8);
            if block & (1 << (f + r * 8)) != 0 {
                break;
            }
            r += 1;
            f -= 1;
        }

        // Diagonal: top right to bottom left
        r = rk - 1;
        f = fl + 1;
        while r >= 0 && f < 8 {
            result |= 1 << (f + r * 8);
            if block & (1 << (f + r * 8)) != 0 {
                break;
            }
            r -= 1;
            f += 1;
        }

        // Diagonal: bottom right to top left
        r = rk - 1;
        f = fl - 1;
        while r >= 0 && f >= 0 {
            result |= 1 << (f + r * 8);
            if block & (1 << (f + r * 8)) != 0 {
                break;
            }
            r -= 1;
            f -= 1;
        }

        result
    }

    /// Finds a suitable magic number for the given square and mask.
    fn find_magic(sq: i32, m: i32, bishop: bool) -> u64 {
        let mask = if bishop {
            BISHOP_MASKS[sq as usize]
        } else {
            ROOK_MASKS[sq as usize]
        };
        let n = Self::count_1s(mask.0);
        let mut b = vec![0; 1 << n];
        let mut a = vec![0; 1 << n];
        let mut used = vec![0; 1 << n];

        for i in 0..(1 << n) {
            b[i] = Self::index_to_uint64(i as i32, n, mask.0);
            a[i] = if bishop {
                Self::batt(sq, b[i])
            } else {
                Self::ratt(sq, b[i])
            };
        }

        for _ in 0..100_000_000 {
            let magic = Self::random_uint64_fewbits();
            if Self::count_1s((mask.0.wrapping_mul(magic)) & 0xFF00000000000000) < 6 {
                continue;
            }

            used.iter_mut().for_each(|x| *x = 0);

            let mut fail = false;
            for i in 0..(1 << n) {
                let j = Self::transform(b[i], magic, m) as usize;
                if used[j] == 0 {
                    used[j] = a[i];
                } else if used[j] != a[i] {
                    fail = true;
                    break;
                }
            }

            if !fail {
                return magic;
            }
        }

        panic!("Failed to find a magic number");
    }

    fn generate_legal_moves_from_blockers(
        square: u16,
        blocker_bitboard: &Bitboard,
        ortho: bool,
    ) -> Bitboard {
        let directions = if ortho {
            chess_foundation::piece_directions::ROOK_DIRECTIONS
        } else {
            chess_foundation::piece_directions::BISHOP_DIRECTIONS
        };
        let start_coord = Coord::from_square_index(square);
        let mut bitboard = Bitboard::default();
        for dir in directions {
            for i in 1..8 {
                let coord = start_coord + dir * i;
                if coord.is_valid_square() {
                    bitboard.set_bit(coord.square_index() as usize);
                    if blocker_bitboard.contains_square(coord.square_index()) {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        bitboard
    }

    fn generate_blocker_bitboards(movement_mask: Bitboard) -> Vec<Bitboard> {
        let move_square_indices: Vec<usize> =
            (0..64).filter(|&i| movement_mask.is_set(i)).collect();

        let num_patterns = 1 << move_square_indices.len();
        let mut blocker_bitboards = vec![Bitboard::default(); num_patterns];
        for pattern_index in 0..num_patterns {
            for bit_index in 0..move_square_indices.len() {
                let bit = (pattern_index >> bit_index) & 1;
                let shift = move_square_indices[bit_index];
                if bit == 1 {
                    blocker_bitboards[pattern_index].set_bit(shift);
                }
            }
        }

        blocker_bitboards
    }
}
