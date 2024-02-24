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

use crate::game_events::{ChessAction, ChessEvent};

pub fn handle_keyboard_input(
    keyboard_input: Res<ButtonInput<KeyCode>>,
    mut chess_ew: EventWriter<ChessEvent>,
) {
    if keyboard_input.just_pressed(KeyCode::KeyU) {
        chess_ew.send(ChessEvent::new(ChessAction::Undo));
    }
}
