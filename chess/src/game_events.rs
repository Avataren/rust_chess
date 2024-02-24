use bevy::ecs::event::Event;

pub enum ChessAction {
    Undo,
}

#[derive(Event)]
pub struct ChessEvent {
    pub action: ChessAction,
}

impl ChessEvent {
    pub fn new(action: ChessAction) -> Self {
        ChessEvent { action }
    }
}

#[derive(Event)]
pub struct RefreshPiecesFromBoardEvent;
