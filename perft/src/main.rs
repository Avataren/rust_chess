use chess_board::ChessBoard;
use chess_foundation::ChessMove;
use move_generator::move_generator::get_all_legal_moves_for_color;
use move_generator::piece_conductor::PieceConductor;
use std::collections::HashMap;
use std::env;
use std::time::Instant;

fn perft(depth: i32, chess_board: &mut ChessBoard, magic: &PieceConductor, is_root: bool) -> u64 {
    if depth == 0 {
        return 1; // At leaf node, count as a single position
    }

    let mut total_nodes = 0u64;
    let legal_moves = get_all_legal_moves_for_color(chess_board, magic, chess_board.is_white_active());

    for mut m in legal_moves {
        // Make the move on the chess board
        let move_was_made = chess_board.make_move(&mut m); // Ensure make_move returns a bool indicating success

        if move_was_made {
            let nodes = perft(depth - 1, chess_board, magic, false);
            if is_root {
                // At the root level, print each move and its node count
                println!("{} {}", m.to_san_simple(), nodes);
            }
            total_nodes += nodes; // Aggregate nodes for all moves at this depth

            // Undo the move to backtrack
            chess_board.undo_move(); // Ensure undo_move uses the move to undo correctly
        }
    }

    if is_root {
        println!(); // Print a blank line after the list of moves
        println!("{}", total_nodes); // Print the total node count at the root level
    }

    total_nodes // Return the total node count for this call
}

// fn perform_perft(perft_depth: i32) {
//     let mut magic = PieceConductor::new();
//     let mut output = Vec::new(); // Use a vector to collect output
//     let headers = format!(
//         "| {:>5} | {:>12} | {:>10} | {:>8} | {:>7} | {:>10} | {:<12} |\n",
//         "Depth", "Nodes", "Captures", "EP", "Castles", "Promotions", "Time Taken"
//     );

//     let mut seperator = String::new();
//     for _ in 0..headers.len() - 1 {
//         seperator.push('-');
//     }
//     output.push(seperator.clone() + "\n");
//     output.push(headers.clone());
//     output.push(seperator.clone() + "\n");

//     for depth in 0..perft_depth {
//         let mut chess_board = ChessBoard::new();
//         chess_board.set_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

//         let start = Instant::now(); // Start timing
//         let result = perft(depth, &mut chess_board, &mut magic);
//         let duration = start.elapsed(); // End timing
//         let line = format!(
//             "| {:^5} | {:>12} | {:>10} | {:>8} | {:>7} | {:>10} | {:>11.7}s |\n",
//             depth,
//             result.0,
//             result.1,
//             result.2,
//             result.3,
//             result.4,
//             duration.as_secs_f64()
//         );
//         output.push(line); // Collect each line of output
//     }
//     output.push(seperator.clone() + "\n");
//     for line in &output {
//         print!("{}", line);
//     }
// }

fn main() {

    // let m = ChessMove::from_san("a1a2");
    // println!("a1a2 to chessmove: {:?} - {:?}", m.start_square(), m.target_square());


    let args: Vec<String> = env::args().skip(1).collect();

    if args.len() < 2 {
        eprintln!("Usage: ./your_program depth FEN [moves]");
        return;
    }

    let depth: i32 = args[0].parse().unwrap_or_else(|_| {
        eprintln!("Invalid depth.");
        std::process::exit(1);
    });

    let fen = &args[1];
    let moves = if args.len() > 2 { &args[2..] } else { &[] };

    let mut magic = PieceConductor::new();
    let mut chess_board = ChessBoard::new();
    if (fen.len() > 1) {
        chess_board.set_from_fen(fen);
    }

    
    for mov in moves {
        let mut chess_move = ChessMove::from_san(mov); 
        chess_board.make_move(&mut chess_move);
    }

    // Perform perft and print results
    //let result = perft(depth, &mut chess_board, &mut magic);
    perft(depth, &mut chess_board, &mut magic, true);
    //println!("{}", result.0); // Print the total node count
}

// let args: Vec<String> = env::args().skip(1).collect();

// // Iterate over the arguments and print them
// for (index, arg) in args.iter().enumerate() {
//     println!("Argument {}: {}", index, arg);
// }

// perform_perft(6);
