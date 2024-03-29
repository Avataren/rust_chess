use chess_board::chessboard::CastlingRights;
use chess_board::ChessBoard;
use chess_foundation::bitboard::Bitboard;
use chess_foundation::{ChessMove, Coord};

use crate::magic_constants::{BISHOP_MAGICS, ROOK_MAGICS};
use crate::masks::{BISHOP_MASKS, ROOK_MASKS};
use crate::{get_king_move_patterns, get_knight_move_patterns};

#[derive (Clone)]
pub struct PieceConductor {
    //pub pawn_lut: Vec<Bitboard>,
    pub knight_lut: Vec<Bitboard>,
    pub king_lut: Vec<Bitboard>,
    rook_table: Vec<Vec<Bitboard>>,
    bishop_table: Vec<Vec<Bitboard>>,
    white_pawn_attack_masks: [Bitboard; 64],
    black_pawn_attack_masks: [Bitboard; 64],
}

impl PieceConductor {
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

        let mut white_pawn_attack_masks = [Bitboard::default(); 64];
        for square in 0..64 {
            white_pawn_attack_masks[square] = Self::get_pawn_attacks(square, true);
        }

        let mut black_pawn_attack_masks = [Bitboard::default(); 64];
        for square in 0..64 {
            black_pawn_attack_masks[square] = Self::get_pawn_attacks(square, false);
        }

        PieceConductor {
            knight_lut,
            king_lut,
            rook_table,
            bishop_table,
            white_pawn_attack_masks,
            black_pawn_attack_masks,
        }
    }

    // for precalculating masks and magics
    // pub fn print_masks() {
    //     println!("pub const ROOK_MASKS: [Bitboard; 64] = [");
    //     //for r in rook_masks.iter() {
    //     for square in 0..64 {
    //         println!("Bitboard(0x{:X}), ", Self::rmask(square));
    //     }
    //     println!("];");

    //     println!("pub const BISHOP_MASKS: [Bitboard; 64] = [");
    //     //for r in bishop_masks.iter() {
    //     for square in 0..64 {
    //         println!("Bitboard(0x{:X}), ", Self::bmask(square));
    //     }
    //     println!("\n];");
    // }

    // pub fn print_magics() {
    //     let mut bishop_magix = Vec::new();
    //     let mut rook_magix = Vec::new();

    //     // Fill the bishop_magix vector
    //     for square in 0..64 {
    //         let m = Self::find_magic(square, Self::BBITS[square as usize], true);
    //         bishop_magix.push(m);
    //     }

    //     // Fill the rook_magix vector
    //     for square in 0..64 {
    //         let m = Self::find_magic(square, Self::RBITS[square as usize], false);
    //         rook_magix.push(m);
    //     }

    //     // Format and print bishop_magix as a const array in uppercase hexadecimal
    //     println!("pub const BISHOP_MAGICS: [u64; 64] = [");
    //     for m in bishop_magix.iter() {
    //         print!("0x{:X}, ", m);
    //     }
    //     println!("\n];");

    //     // Format and print rook_magix as a const array in uppercase hexadecimal
    //     println!("pub const ROOK_MAGICS: [u64; 64] = [");
    //     for m in rook_magix.iter() {
    //         print!("0x{:X}, ", m);
    //     }
    //     println!("\n];");
    // }

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

    pub fn get_rook_moves(
        &self,
        square: u16,
        relevant_blockers: Bitboard,
        chess_board: &ChessBoard,
    ) -> Vec<ChessMove> {
        let magic_index = Self::rook_magic_index(square as usize, chess_board.get_all_pieces());
        let mut moves_bitboard = self.rook_table[square as usize][magic_index] & !relevant_blockers;
        let mut move_list = Vec::new();
        while !moves_bitboard.is_empty() {
            let target_square = moves_bitboard.pop_lsb();
            move_list.push(ChessMove::new(square, target_square as u16));
        }
        move_list
    }

    pub fn get_pawn_moves(
        &self,
        square: u16,
        is_white: bool,
        chess_board: &ChessBoard,
    ) -> Vec<ChessMove> {
        let mut move_list = Vec::new();

        // Calculate rank and file for the pawn
        let rank = square >> 3;
        let file = square & 7;

        // Single move forward
        let single_step = if is_white { 8 } else { -8 };
        let single_move_pos = square as i32 + single_step;

        // Prevent underflow/overflow
        if single_move_pos >= 0 && single_move_pos < 64 {
            if chess_board
                .get_all_free_squares()
                .contains_square(single_move_pos as i32)
            {
                move_list.push(ChessMove::new(square, single_move_pos as u16));
            }
        }

        // Double move forward
        let starting_rank = if is_white { 1 } else { 6 };
        if rank == starting_rank {
            let double_move_pos = single_move_pos + single_step;
            // Prevent underflow/overflow
            if double_move_pos >= 0 && double_move_pos < 64 {
                if chess_board
                    .get_all_free_squares()
                    .contains_square(single_move_pos as i32)
                    && chess_board
                        .get_all_free_squares()
                        .contains_square(double_move_pos as i32)
                {
                    move_list.push(ChessMove::new_with_flag(
                        square,
                        double_move_pos as u16,
                        ChessMove::PAWN_TWO_UP_FLAG,
                    ));
                }
            }
        }

        // Captures
        if file > 0 {
            // Can capture to the left
            let capture_left = if is_white {
                square + 7 // For white pawns
            } else {
                // For black pawns, ensure no underflow
                square.checked_sub(9).unwrap_or_else(|| u16::MAX)
            };

            // Check if capture_left is a legal board position
            if capture_left < 64
                && chess_board
                    .get_opponent_pieces(is_white)
                    .contains_square(capture_left as i32)
            {
                move_list.push(ChessMove::new(square, capture_left));
            }
        }
        if file < 7 {
            // Can capture to the right
            let capture_right = if is_white {
                // For white pawns, ensure no overflow
                square.checked_add(9).unwrap_or_else(|| u16::MAX)
            } else {
                // For black pawns, ensure no underflow
                square.checked_sub(7).unwrap_or_else(|| u16::MAX)
            };

            // Check if capture_right is a legal board position
            if capture_right < 64
                && chess_board
                    .get_opponent_pieces(is_white)
                    .contains_square(capture_right as i32)
            {
                move_list.push(ChessMove::new(square, capture_right));
            }
        }

        // En Passant
        let last_move = chess_board.get_last_move();
        if let Some(last_move) = last_move {
            if last_move.has_flag(ChessMove::PAWN_TWO_UP_FLAG) {
                let last_move_to = last_move.target_square();
                let last_move_to_file = last_move_to & 7;
                let pawn_rank = square >> 3;
                let pawn_file = square & 7;
                let en_passant_rank = if is_white { 4 } else { 3 };
                let is_adjacent_file = (pawn_file as i32 - last_move_to_file as i32).abs() == 1;

                if pawn_rank == en_passant_rank && is_adjacent_file {
                    let en_passant_capture_square = if is_white {
                        last_move_to + 8
                    } else {
                        last_move_to - 8
                    };
                    move_list.push(ChessMove::new_with_flag(
                        square,
                        en_passant_capture_square as u16,
                        ChessMove::EN_PASSANT_CAPTURE_FLAG,
                    ));
                }
            }
        }

        //check if any moves are promotions
        let promotion_rank = if is_white { 7 } else { 0 };
        //for cm in move_list.iter_mut() {
        let promotion_moves = move_list
            .iter()
            .filter(|cm| cm.target_square() >> 3 == promotion_rank)
            .cloned()
            .collect::<Vec<ChessMove>>();

        if promotion_moves.len() > 0 {
            move_list.clear();
        }

        for cm in promotion_moves {
            move_list.push(ChessMove::new_capture_with_flag(
                cm.start_square(),
                cm.capture,
                cm.target_square(),
                ChessMove::PROMOTE_TO_QUEEN_FLAG,
            ));
            move_list.push(ChessMove::new_capture_with_flag(
                cm.start_square(),
                cm.capture,
                cm.target_square(),
                ChessMove::PROMOTE_TO_ROOK_FLAG,
            ));
            move_list.push(ChessMove::new_capture_with_flag(
                cm.start_square(),
                cm.capture,
                cm.target_square(),
                ChessMove::PROMOTE_TO_BISHOP_FLAG,
            ));
            move_list.push(ChessMove::new_capture_with_flag(
                cm.start_square(),
                cm.capture,
                cm.target_square(),
                ChessMove::PROMOTE_TO_KNIGHT_FLAG,
            ));
        }

        move_list
    }
    pub fn get_rook_attacks(
        &self,
        square: usize,
        _relevant_blockers: Bitboard,
        all_pieces: Bitboard,
    ) -> Bitboard {
        let magic_index = Self::rook_magic_index(square, all_pieces);
        self.rook_table[square][magic_index]
    }

    pub fn get_bishop_attacks(
        &self,
        square: usize,
        relevant_blockers: Bitboard,
        all_pieces: Bitboard,
    ) -> Bitboard {
        let magic_index = Self::bishop_magic_index(square, all_pieces);
        self.bishop_table[square][magic_index] & !relevant_blockers
    }

    fn get_pawn_attacks(square: usize, is_white: bool) -> Bitboard {
        let mut attacks = Bitboard::default();

        if is_white {
            // Ensure pawn is not on 1st rank (no attacks from there)
            if square > 7 {
                if (square & 7) != 0 {
                    // Pawn is not on A-file, can attack left
                    attacks.set_bit(square - 9);
                }
                if (square & 7) != 7 {
                    // Pawn is not on H-file, can attack right
                    attacks.set_bit(square - 7);
                }
            }
        } else {
            // Ensure pawn is not on 8th rank (no attacks from there)
            if square < 56 {
                if (square & 7) != 0 {
                    // Pawn is not on A-file, can attack left
                    attacks.set_bit(square + 7);
                }
                if (square & 7) != 7 {
                    // Pawn is not on H-file, can attack right
                    attacks.set_bit(square + 9);
                }
            }
        }
        attacks
    }

    pub fn get_bishop_moves(
        &self,
        square: u16,
        relevant_blockers: Bitboard,
        chess_board: &ChessBoard,
    ) -> Vec<ChessMove> {
        let magic_index = Self::bishop_magic_index(square as usize, chess_board.get_all_pieces());
        let mut moves_bitboard =
            self.bishop_table[square as usize][magic_index] & !relevant_blockers;
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
        friendly_pieces_bb: Bitboard,
        chess_board: &ChessBoard,
        is_white: bool,
    ) -> Vec<ChessMove> {
        let king_moves_bitboard = self.king_lut[square as usize];
        let valid_moves_bitboard = king_moves_bitboard & !friendly_pieces_bb;
        let mut move_list = Vec::new();

        let mut moves_bitboard = valid_moves_bitboard;
        while !moves_bitboard.is_empty() {
            let target_square = moves_bitboard.pop_lsb();
            move_list.push(ChessMove::new(square, target_square as u16));
        }

        move_list.extend(self.get_castling_moves(chess_board, square, is_white));
        move_list
    }

    pub fn get_castling_moves(
        &self,
        chess_board: &ChessBoard,
        square: u16,
        is_white: bool,
    ) -> Vec<ChessMove> {
        let mut castling_moves = Vec::new();

        // Assume the king is in its original square; otherwise, castling isn't possible.
        let original_king_square = if is_white { 4 } else { 60 };
        if square != original_king_square {
            return castling_moves; // Return an empty list if the king isn't in its original square.
        }

        // Determine castling rights based on the color
        let king_side_rights = if is_white {
            CastlingRights::WhiteKingSide
        } else {
            CastlingRights::BlackKingSide
        } as u8;
        let queen_side_rights = if is_white {
            CastlingRights::WhiteQueenSide
        } else {
            CastlingRights::BlackQueenSide
        } as u8;

        // King's potential castling squares
        let king_side_rook_square = if is_white { 7 } else { 63 };
        let queen_side_rook_square = if is_white { 0 } else { 56 };

        // Check if castling rights are available for each side
        let mut can_castle_king_side = chess_board.castling_rights & king_side_rights != 0;
        let mut can_castle_queen_side = chess_board.castling_rights & queen_side_rights != 0;

        // Check if the path is clear for castling
        let king_side_path_clear = chess_board.is_path_clear(square, king_side_rook_square);
        let queen_side_path_clear = chess_board.is_path_clear(square, queen_side_rook_square);

        // Check if the king is in check or the squares it passes through are under attack
        let king_not_in_check = !self.is_king_in_check(chess_board, is_white);
        let threat_map = self.generate_threat_map(chess_board, is_white);
        let king_side_squares_safe = self.are_squares_safe([square + 1, square + 2], threat_map);
        let queen_side_squares_safe = self.are_squares_safe([square - 1, square - 2], threat_map);

        //check that rook actually exists!
        let rooks_bb = chess_board.get_rooks();
        let color_bb = if is_white {
            chess_board.get_white()
        } else {
            chess_board.get_black()
        };

        let colored_rooks_bb = rooks_bb & color_bb;

        if !(colored_rooks_bb.contains_square(king_side_rook_square as i32)) {
            can_castle_king_side = false;
        }
        if !(colored_rooks_bb.contains_square(queen_side_rook_square as i32)) {
            can_castle_queen_side = false;
        }

        // Add kingside castling move if applicable
        if can_castle_king_side
            && king_side_path_clear
            && king_not_in_check
            && king_side_squares_safe
        {
            castling_moves.push(ChessMove::new_with_flag(
                square,
                square + 2,
                ChessMove::CASTLE_FLAG,
            ));
        }

        // Add queenside castling move if applicable
        if can_castle_queen_side
            && queen_side_path_clear
            && king_not_in_check
            && queen_side_squares_safe
        {
            castling_moves.push(ChessMove::new_with_flag(
                square,
                square - 2,
                ChessMove::CASTLE_FLAG,
            ));
        }

        castling_moves
    }

    pub fn generate_threat_map(
        &self,
        chess_board: &ChessBoard,
        // relevant_blockers: Bitboard,
        is_white: bool,
    ) -> Bitboard {
        let all_pieces = chess_board.get_all_pieces();

        let mut enemy_pieces_bb = if is_white {
            chess_board.get_black()
        } else {
            chess_board.get_white()
        };

        let relevant_blockers = enemy_pieces_bb;

        let mut threats_bb = Bitboard::default();

        while enemy_pieces_bb != Bitboard::default() {
            let square = enemy_pieces_bb.pop_lsb() as usize;
            match chess_board.get_piece_type(square as u16) {
                Some(piece_type) => match piece_type {
                    chess_foundation::piece::PieceType::Rook => {
                        threats_bb |= self.get_rook_attacks(square, relevant_blockers, all_pieces);
                    }
                    chess_foundation::piece::PieceType::Bishop => {
                        threats_bb |=
                            self.get_bishop_attacks(square, relevant_blockers, all_pieces);
                    }
                    chess_foundation::piece::PieceType::Queen => {
                        threats_bb |= self.get_rook_attacks(square, relevant_blockers, all_pieces);
                        threats_bb |=
                            self.get_bishop_attacks(square, relevant_blockers, all_pieces);
                    }
                    chess_foundation::piece::PieceType::King => {
                        threats_bb |= self.king_lut[square];
                    }
                    chess_foundation::piece::PieceType::Knight => {
                        threats_bb |= self.knight_lut[square];
                    }
                    chess_foundation::piece::PieceType::Pawn => {
                        threats_bb |= if is_white {
                            self.white_pawn_attack_masks[square]
                        } else {
                            self.black_pawn_attack_masks[square]
                        };
                    }
                    _ => {}
                },
                None => {}
            }
        }
        threats_bb
    }

    pub fn are_squares_safe(&self, squares: [u16; 2], threat_map: Bitboard) -> bool {
        for &square in squares.iter() {
            let square_bb = Bitboard::from_square_index(square);
            // Check if the square is under attack by seeing if it intersects with the threat map
            if (threat_map & square_bb) != Bitboard::default() {
                return false; // If any square is under attack, return false
            }
        }
        true // If none of the squares are under attack, return true
    }

    pub fn is_king_in_check(&self, chess_board: &ChessBoard, is_white: bool) -> bool {
        let king_bb = chess_board.get_king(is_white);
        let threats = self.generate_threat_map(chess_board, is_white);
        (king_bb & threats) != Bitboard::default()
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

#[cfg(test)]
mod tests {
    use crate::move_generator::get_all_legal_moves_for_color;

    use super::*;
    use chess_board::ChessBoard;
    use chess_foundation::bitboard::Bitboard;

    pub fn perft(
        depth: i32,
        chess_board: &mut ChessBoard,
        magic: &PieceConductor,
    ) -> (u64, u64, u64, u64, u64) {
        if depth == 0 {
            return (1, 0, 0, 0, 0); // Leaf node, count as a single position
        }
        let mut nodes = 0;
        let legal_moves =
            get_all_legal_moves_for_color(chess_board, magic, chess_board.is_white_active());
        let mut captures = 0;
        let mut castles = 0;
        let mut promotions = 0;
        let mut ep = 0;

        for mut m in legal_moves {
            // Make the move on the chess board
            let move_was_made = chess_board.make_move(&mut m); // Ensure make_move returns a bool indicating success

            if move_was_made {
                if depth == 1 {
                    // Increment counts based on move flags
                    if m.has_flag(ChessMove::CASTLE_FLAG) {
                        castles += 1;
                    }
                    if m.promotion_piece_type().is_some() {
                        promotions += 1;
                    }
                    if m.has_flag(ChessMove::EN_PASSANT_CAPTURE_FLAG) {
                        ep += 1;
                    }
                    if m.capture.is_some() {
                        captures += 1; // Only count as capture if not en passant
                    }
                }

                // Recursive call to perft for the next depth
                let results = perft(depth - 1, chess_board, magic);
                nodes += results.0;
                captures += results.1;
                ep += results.2;
                castles += results.3;
                promotions += results.4;

                // Undo the move to backtrack
                chess_board.undo_move(); // Ensure undo_move uses the move to undo correctly
            }
        }
        (nodes, captures, ep, castles, promotions)
    }

    // Group related tests into a submodule

    mod perft_tests {
        use super::*;
        use std::{
            fs::File,
            io::{BufWriter, Write},
            time::Instant,
        };

        #[test]
        fn perft_test() {
            let mut magic = PieceConductor::new();
            let mut output = Vec::new(); // Use a vector to collect output
            let headers = format!(
                "| {:>5} | {:>12} | {:>10} | {:>8} | {:>7} | {:>10} | {:<12} |\n",
                "Depth", "Nodes", "Captures", "EP", "Castles", "Promotions", "Time Taken"
            );

            let mut seperator = String::new();
            for _ in 0..headers.len() - 1 {
                seperator.push('-');
            }
            output.push(seperator.clone() + "\n");
            output.push(headers.clone());
            output.push(seperator.clone() + "\n");

            for depth in 0..6 {
                let mut chess_board = ChessBoard::new();
                chess_board
                    .set_from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");

                let start = Instant::now(); // Start timing
                let result = perft(depth, &mut chess_board, &mut magic);
                let duration = start.elapsed(); // End timing
                let line = format!(
                    "| {:^5} | {:>12} | {:>10} | {:>8} | {:>7} | {:>10} | {:>11.7}s |\n",
                    depth,
                    result.0,
                    result.1,
                    result.2,
                    result.3,
                    result.4,
                    duration.as_secs_f64()
                );
                output.push(line); // Collect each line of output
            }
            output.push(seperator.clone() + "\n");
            if cfg!(debug_assertions) {
                for line in &output {
                    println!("{}", line);
                }
            }
            // Write the collected output to a file
            let file_path = "perft_test_results.txt"; // Specify your file path here
            let file = File::create(file_path).expect("Failed to create file");
            let mut writer = BufWriter::new(file);

            for line in output {
                writer
                    .write_all(line.as_bytes())
                    .expect("Failed to write to file");
            }
            println!("Test results with timing written to {}", file_path);
        }
    }

    mod threat_map_tests {
        use super::*;

        #[test]
        fn test_threat_map_queen() {
            let mut chess_board = ChessBoard::new();
            chess_board.clear();
            let magic = PieceConductor::new();
            let is_white = false;

            // Generate the threat map for the square
            chess_board.set_piece_at_square(0, chess_foundation::piece::PieceType::Queen, is_white);
            let mut threat_map = magic.generate_threat_map(
                &mut chess_board,
                // relevant_blockers,
                is_white,
            );
            println!("Threat map:");
            threat_map.print_bitboard();
            let mut expected_threat_map = Bitboard(0x81412111090503fe);
            assert_eq!(
                threat_map, expected_threat_map,
                "The generated threat map does not match the expected map."
            );
            // // // test for a different square
            chess_board.clear();
            chess_board.set_piece_at_square(
                34,
                chess_foundation::piece::PieceType::Queen,
                is_white,
            );
            threat_map = magic.generate_threat_map(
                &mut chess_board,
                // relevant_blockers,
                is_white,
            );
            println!("Threat map:");
            threat_map.print_bitboard();
            expected_threat_map = Bitboard(0x24150efb0e152444);
            assert_eq!(
                threat_map, expected_threat_map,
                "The generated threat map does not match the expected map."
            );
        }

        #[test]
        fn test_threat_map_bishop() {
            let mut chess_board = ChessBoard::new();
            chess_board.clear();
            let magic = PieceConductor::new();
            let is_white = false;

            // Generate the threat map for the square
            chess_board.set_piece_at_square(
                0,
                chess_foundation::piece::PieceType::Bishop,
                is_white,
            );
            let mut threat_map = magic.generate_threat_map(
                &mut chess_board,
                // relevant_blockers,
                is_white,
            );
            println!("Threat map:");
            threat_map.print_bitboard();
            let mut expected_threat_map = Bitboard(0x8040201008040200);
            assert_eq!(
                threat_map, expected_threat_map,
                "The generated threat map does not match the expected map."
            );
            // // test for a different square
            chess_board.clear();
            chess_board.set_piece_at_square(
                34,
                chess_foundation::piece::PieceType::Bishop,
                is_white,
            );
            chess_board.set_piece_at_square(
                36,
                chess_foundation::piece::PieceType::Bishop,
                is_white,
            );
            threat_map = magic.generate_threat_map(
                &mut chess_board,
                // relevant_blockers,
                is_white,
            );
            println!("Threat map:");
            threat_map.print_bitboard();
            expected_threat_map = Bitboard(0xa2552a002a55a241);
            assert_eq!(
                threat_map, expected_threat_map,
                "The generated threat map does not match the expected map."
            );
        }

        #[test]
        fn test_threat_map_rook() {
            let mut chess_board = ChessBoard::new();
            chess_board.clear();
            let magic = PieceConductor::new();
            let is_white = false;

            // Generate the threat map for the square
            chess_board.set_piece_at_square(0, chess_foundation::piece::PieceType::Rook, is_white);
            let mut threat_map = magic.generate_threat_map(
                &mut chess_board,
                // relevant_blockers,
                is_white,
            );
            println!("Threat map:");
            threat_map.print_bitboard();
            let mut expected_threat_map = Bitboard(0x1010101010101fe);
            assert_eq!(
                threat_map, expected_threat_map,
                "The generated threat map does not match the expected map."
            );
            // test for a different square
            chess_board.clear();
            chess_board.set_piece_at_square(63, chess_foundation::piece::PieceType::Rook, is_white);
            chess_board.set_piece_at_square(0, chess_foundation::piece::PieceType::Rook, is_white);
            threat_map = magic.generate_threat_map(
                &mut chess_board,
                // relevant_blockers,
                is_white,
            );
            println!("Threat map:");
            threat_map.print_bitboard();
            expected_threat_map = Bitboard(0x7f818181818181fe);
            assert_eq!(
                threat_map, expected_threat_map,
                "The generated threat map does not match the expected map."
            );
        }

        #[test]
        fn test_threat_map_pawn() {
            let mut chess_board = ChessBoard::new();
            chess_board.clear();
            let magic = PieceConductor::new();
            let is_white = true;
            let square = 48;
            chess_board.set_piece_at_square(
                square,
                chess_foundation::piece::PieceType::Pawn,
                false,
            );

            // chess_board.set_piece_at_square(9, chess_foundation::piece::PieceType::Pawn, is_white);
            // chess_board.set_piece_at_square(10, chess_foundation::piece::PieceType::Pawn, is_white);
            // chess_board.set_piece_at_square(11, chess_foundation::piece::PieceType::Pawn, is_white);
            let threat_map = magic.generate_threat_map(&mut chess_board, is_white);
            println!("pawn map:");
            Bitboard::from_square_index(square).print_bitboard();
            println!("Threat map:");
            threat_map.print_bitboard();

            // // Generate the threat map for the square
            // chess_board.set_piece_at_square(60, chess_foundation::piece::PieceType::Pawn, false);
            // chess_board.set_piece_at_square(61, chess_foundation::piece::PieceType::Pawn, false);
            // chess_board.set_piece_at_square(62, chess_foundation::piece::PieceType::Pawn, false);
            // chess_board.set_piece_at_square(63, chess_foundation::piece::PieceType::Pawn, false);
            // // chess_board.set_piece_at_square(9, chess_foundation::piece::PieceType::Pawn, is_white);
            // // chess_board.set_piece_at_square(10, chess_foundation::piece::PieceType::Pawn, is_white);
            // // chess_board.set_piece_at_square(11, chess_foundation::piece::PieceType::Pawn, is_white);
            // let mut threat_map = magic.generate_threat_map(
            //     &mut chess_board,
            //     is_white,
            // );

            // println!("Threat map:");
            // threat_map.print_bitboard();
            // let mut expected_threat_map = Bitboard(0xf8000000000000);
            // assert_eq!(
            //     threat_map, expected_threat_map,
            //     "The generated threat map does not match the expected map."
            // );
        }

        #[test]
        fn test_threat_from_fen() {
            let mut chess_board = ChessBoard::new();
            let magic = PieceConductor::new();
            let is_white = false;

            chess_board.set_from_fen("8/p2k4/1N6/8/8/8/8/4K3 w  - 0 1");
            let threat_map = magic.generate_threat_map(&mut chess_board, is_white);

            println!("Threat map:");
            threat_map.print_bitboard();

            // chess_board.set_from_fen("8/8/8/8/7q/8/6k1/4K3 w  - 0 1");

            // chess_board.set_from_fen("4K3/4Q3/3q4/1k6/8/8/8/8 w  - 0 1");
            // let mut threat_map = magic.generate_threat_map(
            //     &mut chess_board,
            //     is_white,
            // );

            // println!("Threat map:");
            // threat_map.print_bitboard();

            // let mut expected_threat_map = Bitboard(0xa1cf71d2f498808);
            // assert_eq!(
            //     threat_map, expected_threat_map,
            //     "The generated threat map does not match the expected map."
            // );

            // chess_board.set_from_fen("8/8/5p1/4p3/4K3/8/8/8 w KQkq - 0 1");
            // let mut threat_map = magic.generate_threat_map(
            //     &mut chess_board,
            //     is_white,
            // );

            // println!("Threat map:");
            // threat_map.print_bitboard();

            // let mut expected_threat_map = Bitboard(0xa1cf71d2f498808);
            // assert_eq!(
            //     threat_map, expected_threat_map,
            //     "The generated threat map does not match the expected map."
            // );
        }
    }
}
