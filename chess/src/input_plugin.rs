use bevy::prelude::*;

use crate::{
    chess_event_handler,
    game_events::{
        ChessEvent, DragPieceEvent, DropPieceEvent, PickUpPieceEvent, RefreshPiecesFromBoardEvent,
    },
    keyboard_input, piece_picker, pieces,
};

pub struct ChessInputPlugin;

impl Plugin for ChessInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<ChessEvent>()
            .add_message::<ChessEvent>()
            .add_message::<PickUpPieceEvent>()
            .add_message::<DragPieceEvent>()
            .add_message::<DropPieceEvent>()
            .add_message::<RefreshPiecesFromBoardEvent>()
            //.insert_resource(AsyncTaskRunner::<(i32, Option<ChessMove>)>::default) 
            .add_systems(
                PreUpdate,
                (
                    piece_picker::handle_mouse_input,
                    piece_picker::handle_touch_input,
                    keyboard_input::handle_keyboard_input,
                )
                    .chain(),
            )
            .add_systems(
                Update,
                (
                    pieces::spawn_chess_pieces,
                    piece_picker::pick_up_piece,
                    piece_picker::drag_piece,
                    piece_picker::drop_piece,
                    chess_event_handler::handle_async_moves
                )
                    .chain(),
            )
            .add_systems(PostUpdate, chess_event_handler::handle_chess_events);

    }
}
