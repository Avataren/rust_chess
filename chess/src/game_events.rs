use bevy::ecs::message::Message;

pub enum ChessAction {
    MakeMove,
    Undo,
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
