use bevy::ecs::{
    event::{EventReader, EventWriter},
    system::ResMut,
};
use move_generator::move_generator::get_all_legal_moves_for_color;
use rand::seq::SliceRandom;

use crate::{
    game_events::{ChessAction, ChessEvent, RefreshPiecesFromBoardEvent},
    ChessBoardRes, PieceConductorRes,
};

pub fn handle_chess_events(
    mut chess_ew: EventReader<ChessEvent>,
    mut chess_board: ResMut<ChessBoardRes>,
    mut move_generator: ResMut<PieceConductorRes>,
    mut refresh_pieces_events: EventWriter<RefreshPiecesFromBoardEvent>,
) {
    for event in chess_ew.read() {
        match event.action {
            ChessAction::Undo => {
                println!("Undoing move");
                chess_board.chess_board.undo_move();
                refresh_pieces_events.send(RefreshPiecesFromBoardEvent);
            }
            ChessAction::MakeMove => {
                println!("Making black move");
                let moves = get_all_legal_moves_for_color(&mut chess_board.chess_board, &mut move_generator.magic, false);
                if moves.len() == 0 {
                    println!("Checkmate or Stalemate! Game over!");
                    continue;
                }
                if let Some(random_move) = moves.choose(&mut rand::thread_rng())
                {
                    chess_board.chess_board.make_move(&mut random_move.clone());
                    refresh_pieces_events.send(RefreshPiecesFromBoardEvent);
                }
            }

        }
    }
}
