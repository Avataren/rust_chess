use core::num;
use std::collections::HashMap;
use std::io::{self, Write};
use std::process::exit;
use std::{collections::HashSet, fs::File}; // Ensure the Write trait is in scope

use crate::move_patterns;
use crate::move_patterns::MovePatterns;
use chess_board::ChessBoard;
use chess_foundation::bitboard::Bitboard;
use chess_foundation::{coord, ChessMove, Coord};
use rand::Rng;

use crate::piece_patterns::get_bishop_move_patterns;
use crate::piece_patterns::get_king_move_patterns;
use crate::piece_patterns::get_knight_move_patterns;
use crate::piece_patterns::get_pawn_move_patterns;
use crate::piece_patterns::get_rook_move_patterns;

extern crate nalgebra as na;
use na::Vector2;

const MAX_MAGIC_NUMBER_ATTEMPTS: u64 = 1000000;

pub struct Magic {
    pub rook_lut: HashMap<(i32, Bitboard), Bitboard>,
    pub chess_board: ChessBoard,
}

impl Magic {
    pub fn new(chess_board: ChessBoard) -> Magic {
        let rook_lut = Self::generate_rook_lut();
        Magic {
            rook_lut: rook_lut,
            chess_board: chess_board,
        }
    }

    pub fn get_move_list_from_square(&self, square: i32) -> Vec<ChessMove> {
        if !Coord::from_square_index(square).is_valid_square() {
            println!("Invalid square index{}", square);
            return Vec::new();
        }
        println!("Getting moves from square index{}", square);
        let mut move_list = Vec::new();
        //if !(self.chess_board.get_rooks().and(Bitboard::contains_square(square as usize))).is_empty()
        if self.chess_board.get_rooks().contains_square(square as i32) {
            let all_pieces_bitboard = self
                .chess_board
                .get_white()
                .or(self.chess_board.get_black());
            //let mut blocker_bitboard = all_pieces_bitboard.and(movement_mask);

            let legal_move_bitboard =
                Self::generate_legal_moves_from_blockers(square as i32, all_pieces_bitboard, true);

            // let key = (square as i32, legal_move_bitboard);
            // // Attempt to retrieve the value associated with the key from the lookup table
            // if let Some(moves_bitboard_from_lut) = self.rook_lut.get(&key) {
            //     // If the key is found and the bitboard is not empty, iterate over its bits
            //     let mut moves_bitboard = moves_bitboard_from_lut.clone();
            //     while !moves_bitboard.is_empty() {
            //         let target_square = moves_bitboard.pop_lsb(); // Assuming `pop_lsb` is a method that modifies `moves_bitboard`
            //         move_list.push(Coord::new(square as i32, target_square as i32));
            //     }
            // } else {
            //     // If the key is not found, print a message
            //     println!("No keys found at square {}", square);
            // }
            let mut moves_bitboard = legal_move_bitboard.clone();
            while !moves_bitboard.is_empty() {
                let target_square = moves_bitboard.pop_lsb(); // Assuming `pop_lsb` is a method that modifies `moves_bitboard`
                move_list.push(ChessMove::new(square as u16, target_square as u16));
            }
        }
        move_list
    }
    // {
    //     let mut move_list = Vec::new();
    //     let all_pieces_bitboard = self.chess_board.get_white().or(self.chess_board.get_black());
    //     let mut blocker_bitboard = all_pieces_bitboard.and(movement_mask);
    //     let key = (square as i32, blocker_bitboard);
    //     let moves_bitboard = self.rook_lut.get(&key);

    //     while !blocker_bitboard.is_empty(){
    //         let target_square = blocker_bitboard.pop_lsb();
    //         move_list.push(Coord::new(square as i32, target_square as i32));
    //     }
    //     move_list
    // }

    fn generate_rook_lut() -> HashMap<(i32, Bitboard), Bitboard> {
        let movement_mask = get_rook_move_patterns();
        let mut rook_moves_lut = HashMap::new();
        for square in 0..64 {
            let blocker_bitboards = Self::generate_blocker_bitboards(movement_mask[square]);
            for blocker_bitboard in blocker_bitboards {
                //let legal_move_bitboard = self.generate_rook_moves(square, blocker_bitboard);
                let legal_move_bitboard =
                    Self::generate_legal_moves_from_blockers(square as i32, blocker_bitboard, true);
                //self.generate_rook_legal_move_bitboard(square, blocker_bitboard);
                rook_moves_lut.insert((square as i32, blocker_bitboard), legal_move_bitboard);
            }
        }
        rook_moves_lut
    }

    fn generate_legal_moves_from_blockers(
        square: i32,
        blocker_bitboard: Bitboard,
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
    // public static ulong LegalMoveBitboardFromBlockers(int startSquare, ulong blockerBitboard, bool ortho)
    // {
    //     ulong bitboard = 0;

    //     Coord[] directions = ortho ? BoardHelper.RookDirections : BoardHelper.BishopDirections;
    //     Coord startCoord = new Coord(startSquare);

    //     foreach (Coord dir in directions)
    //     {
    //         for (int dst = 1; dst < 8; dst++)
    //         {
    //             Coord coord = startCoord + dir * dst;

    //             if (coord.IsValidSquare())
    //             {
    //                 BitBoardUtility.SetSquare(ref bitboard, coord.SquareIndex);
    //                 if (BitBoardUtility.ContainsSquare(blockerBitboard, coord.SquareIndex))
    //                 {
    //                     break;
    //                 }
    //             }
    //             else { break; }
    //         }
    //     }

    //     return bitboard;
    // }

    // fn generate_rook_legal_move_bitboard(&self, square: u32, blocker_bitboard: Bitboard) -> Bitboard {
    //     // let mut all_pieces_bitboard = self.chess_board.get_white().or(self.chess_board.get_black());
    //     // let blocker_bitboard = all_pieces_bitboard.and(movement_mask);
    //     // moves_bitboard
    //     //self.move_patterns.rook_move_patterns.get(square).unwrap()
    // }

    fn generate_blocker_bitboards(movement_mask: Bitboard) -> Vec<Bitboard> {
        let move_square_indices: Vec<usize> =
            (0..64).filter(|&i| movement_mask.is_set(i)).collect();

        let num_patterns = 1 << move_square_indices.len();
        let mut blocker_bitboards = vec![Bitboard::default(); num_patterns];
        for pattern_index in 0..num_patterns {
            for bit_index in 0..move_square_indices.len() {
                let bit = (pattern_index >> bit_index) & 1;
                blocker_bitboards[pattern_index].set_bit(bit << move_square_indices[bit_index]);
            }
        }

        blocker_bitboards
    }

    // fn generate_rook_moves(square: usize, movement_mask: Bitboard) -> Vec<(u32, u32)> {

    //     let mut move_list = Vec::new();
    //     let all_pieces_bitboard = self.chess_board.get_white().or(self.chess_board.get_black());
    //     let mut blocker_bitboard = all_pieces_bitboard.and(movement_mask);
    //     let key = (square as i32, blocker_bitboard);
    //     let moves_bitboard = self.rook_lut.get(&key);

    //     while !blocker_bitboard.is_empty(){
    //         let target_square = blocker_bitboard.pop_lsb();
    //         move_list.push((square as u32, target_square as u32));
    //     }
    //     move_list
    // }
}

//     pub fn find_and_write_magic_numbers(&self) -> io::Result<()> {
//         let mut bishop_magic_numbers = Vec::new();
//         let mut rook_magic_numbers = Vec::new();

//         // Calculate and print progress
//         let total_squares = 64;

//         for square in 0..total_squares {
//             // Important: Get occupancy variations for the CURRENT square
//             let occupancy_variations = &self.occupancy_variations_bishop[square];
//             let magic_number = find_magic_number(
//                 occupancy_variations,
//                 MAX_MAGIC_NUMBER_ATTEMPTS,
//                 self.bits_for_occupancy_bishop,
//             );
//             if (magic_number == 0) {
//                 println!("Failed to find a magic number for bishop {}", square);
//                 exit(1);
//             }
//             // ... (Error handling)

//             print!(
//                 "*********** Bishop magic number for square {}: {} ***********",
//                 square, magic_number
//             );
//             bishop_magic_numbers.push(magic_number);

//             // Update and print the progress as we loop through squares
//             let progress = (square + 1) as f64 / total_squares as f64 * 100.0;
//             println!("Bishop progress: {:.2}%", progress);

//             // Similar process for rooks:

//             let occupancy_variations = &self.occupancy_variations_rook[square];
//             let magic_number = find_magic_number(
//                 occupancy_variations,
//                 MAX_MAGIC_NUMBER_ATTEMPTS,
//                 self.bits_for_occupancy_rook + 1,
//             );

//             if (magic_number == 0) {
//                 println!("Failed to find a magic number for rook {}", square);
//                 exit(1);
//             }
//             // ... (Error handling)

//             print!(
//                 "*********** Rook magic number for square {}: {} ***********",
//                 square, magic_number
//             );
//             rook_magic_numbers.push(magic_number);

//             let progress = (square + 1) as f64 / total_squares as f64 * 100.0;
//             println!("Rook progress: {:.2}%", progress);
//         }

//         // Write to a Rust file
//         let mut file = File::create("move_generator/src/magic_bitboards.rs")?;
//         writeln!(file, "// Magic Numbers for Bishops and Rooks")?;
//         writeln!(
//             file,
//             "pub const BISHOP_MAGIC_NUMBERS: [u64; 64] = {:?};",
//             bishop_magic_numbers
//         )?;
//         writeln!(
//             file,
//             "pub const ROOK_MAGIC_NUMBERS: [u64; 64] = {:?};",
//             rook_magic_numbers
//         )?;

//         // Optionally, also write the occupancy variations if needed
//         // ...

//         Ok(())
//     }
// }

// fn calculate_lookup_table_size(occupancy_variations: &Vec<u64>) -> (usize, f64) {
//     // Since we're working with a single Vec<u64>, we directly calculate its byte size.
//     let total_size_bytes = occupancy_variations.len() * std::mem::size_of::<u64>();
//     // The average size per square doesn't apply in the context of a single square's variations.
//     // If needed, you can calculate the average variation size, but it might not be meaningful.
//     // Here, we'll set the average size per square to the size of one u64, as an example.
//     let avg_size_per_square = std::mem::size_of::<u64>() as f64;

//     (total_size_bytes, avg_size_per_square)
// }

// fn find_magic_number(
//     occupancy_variations: &Vec<u64>,
//     max_attempts: u64,
//     bits_to_shift: u32,
// ) -> u64 {
//     let mut rng = rand::thread_rng();
//     let mut best_magic_number = 0;
//     let mut best_index_size = usize::MAX; // Start with the maximum possible value

//     for _ in 0..max_attempts {
//         let candidate_magic = rng.gen::<u64>() & rng.gen::<u64>() & rng.gen::<u64>(); // Random sparse number

//         if let Some(index_size) =
//             test_magic_number(candidate_magic, occupancy_variations, bits_to_shift)
//         {
//             // Update the best candidate if the current one is better
//             if index_size < best_index_size {
//                 best_magic_number = candidate_magic;
//                 best_index_size = index_size;

//                 // Optional: If the candidate meets your specific requirements, return it immediately
//                 if index_size < (1 << bits_to_shift) {
//                     println!(
//                         "Found a suitable magic number with index size <= {}: {} at index {}",
//                         (1 << bits_to_shift),
//                         candidate_magic,
//                         index_size
//                     );
//                     return candidate_magic;
//                 }
//             }
//         }
//     }

//     // After all attempts, return the best magic number found, even if it doesn't meet the specific condition
//     if best_index_size != usize::MAX {
//         println!(
//             "Returning the best found magic number: {} for index {}",
//             best_magic_number, best_index_size
//         );
//         best_magic_number
//     } else {
//         // If no valid magic number was found at all, return 0
//         0
//     }
// }

// fn test_magic_number(
//     magic_number: u64,
//     occupancy_variations: &[u64],
//     bits_to_shift: u32,
// ) -> Option<usize> {
//     let mut used_indices = HashSet::new();

//     for &occupancy in occupancy_variations {
//         let index = (occupancy.wrapping_mul(magic_number) >> (64 - bits_to_shift)) as usize; // Example shift, adjust as needed

//         if !used_indices.insert(index) {
//             return None; // Collision found, not a suitable magic number
//         }
//     }

//     Some(used_indices.len()) // Return the size of the index table as the metric
// }
