use bevy::ecs::message::Message;
use chess_foundation::ChessMove;

pub enum ChessAction {
    MakeMove,
    Undo,
    Restart,
}

/// Sent by the search handler to trigger the piece tween animation for the AI move.
/// Handled by `apply_ai_move_animation` which owns `piece_query` and `Commands`.
#[derive(Message)]
pub struct AiMoveAnimEvent {
    pub engine_move: ChessMove,
}

#[derive(Message)]
pub struct ChessEvent {
    pub action: ChessAction,
}

impl ChessEvent {
    pub fn new(action: ChessAction) -> Self {
        ChessEvent { action }
    }
}

#[derive(Message)]
pub struct RefreshPiecesFromBoardEvent;

#[derive(Message)]
pub struct PickUpPieceEvent {
    pub position: bevy::math::Vec2,
}
#[derive(Message)]
pub struct DragPieceEvent {
    pub position: bevy::math::Vec2,
}
#[derive(Message)]
pub struct DropPieceEvent {
    pub position: bevy::math::Vec2,
}
