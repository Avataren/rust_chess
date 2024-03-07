use bevy::ecs::event::Event;

pub enum ChessAction {
    MakeMove,
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

// pub enum ChessInputState {
//     PickUp,
//     Drag,
//     Drop,
//     None,
// }
// #[derive(Event)]
// pub struct ChessInputEvent {
//     pub state: ChessInputState,
//     pub vec2: bevy::math::Vec2,
// }

// impl ChessInputEvent {
//     pub fn new(state: ChessInputState, vec2: bevy::math::Vec2) -> Self {
//         ChessInputEvent { state, vec2 }
//     }
// }

#[derive(Event)]
pub struct PickUpPieceEvent {
    pub position: bevy::math::Vec2,
}
#[derive(Event)]
pub struct DragPieceEvent {
    pub position: bevy::math::Vec2,
}
#[derive(Event)]
pub struct DropPieceEvent {
    pub position: bevy::math::Vec2,
}
