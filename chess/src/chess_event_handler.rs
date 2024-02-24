use bevy::ecs::{
    event::{EventReader, EventWriter},
    system::ResMut,
};

use crate::{
    game_events::{ChessAction, ChessEvent, RefreshPiecesFromBoardEvent},
    ChessBoardRes,
};

pub fn handle_chess_events(
    mut chess_ew: EventReader<ChessEvent>,
    mut chess_board: ResMut<ChessBoardRes>,
    mut refresh_pieces_events: EventWriter<RefreshPiecesFromBoardEvent>,
) {
    for event in chess_ew.read() {
        match event.action {
            ChessAction::Undo => {
                println!("Undoing move");
                chess_board.chess_board.undo_move();
                refresh_pieces_events.send(RefreshPiecesFromBoardEvent);
            }
            _ => {}
        }
    }
}
