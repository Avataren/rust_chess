
use std::collections::HashMap;


 // Ensure the Write trait is in scope



use chess_board::ChessBoard;
use chess_foundation::bitboard::Bitboard;
use chess_foundation::{ChessMove, Coord};


use crate::{get_king_move_patterns, get_knight_move_patterns, get_pawn_move_patterns};
use crate::piece_patterns::get_bishop_move_patterns;



use crate::piece_patterns::get_rook_move_patterns;

extern crate nalgebra as na;


const MAX_MAGIC_NUMBER_ATTEMPTS: u64 = 1000000;

pub struct Magic {
    //pub pawn_lut: Vec<Bitboard>,
    pub knight_lut: Vec<Bitboard>,
    pub rook_lut: HashMap<(i32, Bitboard), Bitboard>,
    pub bishop_lut: HashMap<(i32, Bitboard), Bitboard>,
    pub king_lut: Vec<Bitboard>,
    pub all_rook_move_patterns: Vec<Vec<Bitboard>>
}

impl Magic {
    pub fn new() -> Self {

        //********** Generate all rook move patterns **********
        let mut all_rook_move_patterns = Vec::new();
        let movement_mask = get_rook_move_patterns();
        for square in 0..64 {
            all_rook_move_patterns.push(Self::generate_blocker_bitboards(movement_mask[square]));
        }
        let rook_lut = Self::generate_rook_lut(&all_rook_move_patterns);
        //********** Generate all bishop move patterns **********
        let mut all_bishop_move_patterns = Vec::new();
        let movement_mask = get_bishop_move_patterns();
        for square in 0..64 {
            all_bishop_move_patterns.push(Self::generate_blocker_bitboards(movement_mask[square]));
        }
        let bishop_lut = Self::generate_bishop_lut(&all_bishop_move_patterns);
        //********** Generate all king move patterns **********
        let king_lut = get_king_move_patterns();
        let knight_lut = get_knight_move_patterns();
        //let pawn_lut = get_pawn_move_patterns();

        Magic {
            // pawn_lut,
            knight_lut,
            rook_lut,
            bishop_lut,
            all_rook_move_patterns,
            king_lut
        }
    }



    pub fn get_move_list_from_square(&self, square: u16, chess_board: &ChessBoard, is_white:bool) -> Vec<ChessMove> {
        let mut move_list = Vec::new();

        let all_pieces_bitboard = 
        chess_board
        .get_white()
        .or(chess_board.get_black());

        let friendly_pieces_bitboard = if is_white {
            chess_board.get_white()
        } else {
            chess_board.get_black()
        };

        if chess_board.get_rooks().contains_square(square as i32) {
            move_list = self.get_valid_rook_moves(square, all_pieces_bitboard, friendly_pieces_bitboard);
        } else if chess_board.get_bishops().contains_square(square as i32) {
            move_list = self.get_valid_bishop_moves(square, all_pieces_bitboard, friendly_pieces_bitboard);
        } else if chess_board.get_queens().contains_square(square as i32) {
            move_list = self.get_valid_rook_moves(square, all_pieces_bitboard, friendly_pieces_bitboard);
            move_list.extend(self.get_valid_bishop_moves(square, all_pieces_bitboard, friendly_pieces_bitboard));
        } else if chess_board.get_kings().contains_square(square as i32) {
            move_list = self.get_valid_king_moves(square, friendly_pieces_bitboard);
        } else if chess_board.get_knights().contains_square(square as i32) {
            move_list = self.get_valid_knight_moves(square, friendly_pieces_bitboard);
        }
        
        move_list
    }


    fn get_valid_knight_moves(&self, square: u16, friendly_pieces_bitboard: Bitboard) -> Vec<ChessMove> {
        // Retrieve the king's move pattern for the given square from the pre-calculated lookup table
        let knight_moves_bitboard = self.knight_lut[square as usize];
        // Exclude squares occupied by friendly pieces from the king's potential moves
        let valid_moves_bitboard = knight_moves_bitboard & !friendly_pieces_bitboard;
        // Initialize an empty list to hold the valid moves
        let mut move_list = Vec::new();

        // Iterate over each bit in the valid_moves_bitboard
        let mut moves_bitboard = valid_moves_bitboard;
        while !moves_bitboard.is_empty() {
            // Get the next possible move square
            let target_square = moves_bitboard.pop_lsb(); // Assuming `pop_lsb` is a method that modifies `moves_bitboard`

            // Add this move to the list of valid moves
            move_list.push(ChessMove::new(square, target_square as u16));
        }

        move_list
    }

    fn get_valid_king_moves(&self, square: u16, friendly_pieces_bitboard: Bitboard) -> Vec<ChessMove> {
        // Retrieve the king's move pattern for the given square from the pre-calculated lookup table
        let king_moves_bitboard = self.king_lut[square as usize];
        // Exclude squares occupied by friendly pieces from the king's potential moves
        let valid_moves_bitboard = king_moves_bitboard & !friendly_pieces_bitboard;
        // Initialize an empty list to hold the valid moves
        let mut move_list = Vec::new();

        // Iterate over each bit in the valid_moves_bitboard
        let mut moves_bitboard = valid_moves_bitboard;
        while !moves_bitboard.is_empty() {
            // Get the next possible move square
            let target_square = moves_bitboard.pop_lsb(); // Assuming `pop_lsb` is a method that modifies `moves_bitboard`

            // Add this move to the list of valid moves
            move_list.push(ChessMove::new(square, target_square as u16));
        }

        move_list
    }

    fn get_valid_rook_moves(&self, square: u16, all_pieces_bitboard: Bitboard, friendly_pieces_bitboard: Bitboard) -> Vec<ChessMove> {
        let blocker_bb = all_pieces_bitboard & get_rook_move_patterns()[square as usize];
        let key = (square as i32, blocker_bb);
        let mut move_list = Vec::new();
        
        // Attempt to retrieve the moves bitboard from the lookup table
        if let Some(moves_bitboard_from_lut) = self.rook_lut.get(&key) {
            // Exclude squares occupied by friendly pieces
            let valid_moves_bitboard = moves_bitboard_from_lut.and(!friendly_pieces_bitboard);
            
            // Iterate over the valid moves
            let mut moves_bitboard = valid_moves_bitboard.clone();
            while !moves_bitboard.is_empty() {
                let target_square = moves_bitboard.pop_lsb();
                move_list.push(ChessMove::new(square, target_square as u16));
            }
        } else {
            println!("No keys found for rook at square {}", square);
        }
    
        move_list
    }
    
    fn get_valid_bishop_moves(&self, square: u16, all_pieces_bitboard: Bitboard, friendly_pieces_bitboard: Bitboard) -> Vec<ChessMove> {
        let blocker_bb = all_pieces_bitboard & get_bishop_move_patterns()[square as usize];
        let key = (square as i32, blocker_bb);
        let mut move_list = Vec::new();
        
        // Attempt to retrieve the moves bitboard from the lookup table
        if let Some(moves_bitboard_from_lut) = self.bishop_lut.get(&key) {
            // Exclude squares occupied by friendly pieces
            let valid_moves_bitboard = moves_bitboard_from_lut.and(!friendly_pieces_bitboard);
            
            // Iterate over the valid moves
            let mut moves_bitboard = valid_moves_bitboard.clone();
            while !moves_bitboard.is_empty() {
                let target_square = moves_bitboard.pop_lsb();
                move_list.push(ChessMove::new(square, target_square as u16));
            }
        } else {
            println!("No keys found for bishop at square {}", square);
        }
    
        move_list
    }
      

    fn generate_rook_lut(all_rook_move_patterns: &Vec<Vec<Bitboard>>) -> HashMap<(i32, Bitboard), Bitboard> {
        
        let mut rook_moves_lut = HashMap::new();
        for square in 0..64 {
            let blocker_bitboards = all_rook_move_patterns[square].clone();
            for blocker_bitboard in blocker_bitboards {
                //let legal_move_bitboard = self.generate_rook_moves(square, blocker_bitboard);
                let legal_move_bitboard =
                    Self::generate_legal_moves_from_blockers(square as u16, &blocker_bitboard, true);
                //self.generate_rook_legal_move_bitboard(square, blocker_bitboard);
                rook_moves_lut.insert((square as i32, blocker_bitboard), legal_move_bitboard);
            }
        }
        rook_moves_lut
    }

    fn generate_bishop_lut(all_rook_move_patterns: &Vec<Vec<Bitboard>>) -> HashMap<(i32, Bitboard), Bitboard> {
        
        let mut rook_moves_lut = HashMap::new();
        for square in 0..64 {
            let blocker_bitboards = all_rook_move_patterns[square].clone();
            for blocker_bitboard in blocker_bitboards {
                //let legal_move_bitboard = self.generate_rook_moves(square, blocker_bitboard);
                let legal_move_bitboard =
                    Self::generate_legal_moves_from_blockers(square as u16, &blocker_bitboard, false);
                //self.generate_rook_legal_move_bitboard(square, blocker_bitboard);
                rook_moves_lut.insert((square as i32, blocker_bitboard), legal_move_bitboard);
            }
        }
        rook_moves_lut
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
                let shift = move_square_indices[bit_index];
                // println! ("setting bit {} at shift {}", bit, shift);
                //blocker_bitboards[pattern_index].set_bit( shift);
                if bit == 1 {
                    blocker_bitboards[pattern_index].set_bit(shift);
                } else {
                    blocker_bitboards[pattern_index].clear_bit(shift);
                }
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
