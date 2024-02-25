use bevy::{
    ecs::{
        event::{EventReader, EventWriter},
        system::Res,
    },
    input::{
        keyboard::{KeyCode, KeyboardInput},
        ButtonInput, ButtonState,
    },
};
use chess_board::FENParser;

use crate::{
    game_events::{ChessAction, ChessEvent},
    ChessBoardRes,
};

pub fn handle_keyboard_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut chess_ew: EventWriter<ChessEvent>,
    chess_board: Res<ChessBoardRes>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyU) {
        chess_ew.send(ChessEvent::new(ChessAction::Undo));
    }
    if keyboard_input.just_pressed(KeyCode::KeyP) {
        println!("{}", FENParser::board_to_fen(&chess_board.chess_board))
    }
}
