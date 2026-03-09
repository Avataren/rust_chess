use bevy::prelude::*;
use bevy::ecs::message::MessageWriter;
use chess_board::FENParser;

use crate::{
    game_events::{ChessAction, ChessEvent},
    game_resources::{GameOverState, GamePhase},
    ChessBoardRes,
};

pub fn handle_keyboard_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut chess_ew: MessageWriter<ChessEvent>,
    chess_board: Res<ChessBoardRes>,
    game_phase: Res<GamePhase>,
    game_over_state: Res<GameOverState>,
) {
    // Only handle gameplay keys when actually playing and not game over
    if *game_phase != GamePhase::Playing || *game_over_state != GameOverState::Playing {
        return;
    }
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
