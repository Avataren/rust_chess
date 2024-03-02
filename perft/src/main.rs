use chess_board::ChessBoard;
use chess_foundation::ChessMove;
use move_generator::move_generator::get_all_legal_moves_for_color;
use move_generator::piece_conductor::PieceConductor;
use std::collections::HashMap;
use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

fn perft(
    depth: i32,
    chess_board: &mut ChessBoard,
    conductor: &PieceConductor,
    is_root: bool,
) -> u64 {
    if depth == 0 {
        return 1; // At leaf node, count as a single position
    }

    let mut total_nodes = 0u64;
    let legal_moves =
        get_all_legal_moves_for_color(chess_board, conductor, chess_board.is_white_active());

    for mut m in legal_moves {
        // Make the move on the chess board
        let move_was_made = chess_board.make_move(&mut m); // Ensure make_move returns a bool indicating success

        if move_was_made {
            let nodes = perft(depth - 1, chess_board, conductor, false);
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

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let mut log_file = OpenOptions::new()
        .append(true) // Append to the file if it already exists
        .create(true) // Create the file if it doesn't exist
        .open("arguments_log.txt") // Specify the log file name
        .expect("Failed to open log file");

    let start = SystemTime::now(); // Get the current system time
    let since_the_epoch = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    let timestamp = since_the_epoch.as_secs(); // Convert time to a simple timestamp

    // Convert the arguments to a single string, separated by spaces
    let args_str = args.join(" ");

    // Write the timestamp and the arguments to the log file
    writeln!(log_file, "{}: {}", timestamp, args_str).expect("Failed to write to log file");

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

    let mut conductor = PieceConductor::new();
    let mut chess_board = ChessBoard::new();
    if (fen.len() > 1) {
        chess_board.set_from_fen(fen);
    }

    for mov in moves.iter() {
        let mut chess_move = ChessMove::from_san(mov);
        chess_board.make_move(&mut chess_move);
    }

    // Perform perft and print results
    perft(depth, &mut chess_board, &mut conductor, true);
}
