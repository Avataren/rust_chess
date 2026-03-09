use bevy::prelude::*;
use bevy::ecs::message::MessageWriter;
use chess_board::FENParser;

use crate::{
    game_events::{ChessAction, ChessEvent},
    ChessBoardRes,
};

pub fn handle_keyboard_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut chess_ew: MessageWriter<ChessEvent>,
    chess_board: Res<ChessBoardRes>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyU) {
        chess_ew.write(ChessEvent::new(ChessAction::Undo));
    }
    if keyboard_input.just_pressed(KeyCode::KeyP) {
        println!("{}", FENParser::board_to_fen(&chess_board.chess_board))
    }
    if keyboard_input.just_pressed(KeyCode::KeyR) {
        chess_ew.write(ChessEvent::new(ChessAction::Restart));
    }
}
