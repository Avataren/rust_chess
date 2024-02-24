use bevy::{
    input::{mouse::MouseButton, touch::TouchPhase},
    prelude::*,
    window::PrimaryWindow,
};
use move_generator::move_generator::get_move_list_from_square;

use crate::{
    board::{BoardDimensions, ChessBoardTransform},
    board_accessories::{DebugSquare, EnableDebugMarkers},
    game_events::{DragPieceEvent, DropPieceEvent, PickUpPieceEvent, RefreshPiecesFromBoardEvent},
    pieces::{chess_coord_to_board, get_board_coords_from_cursor, ChessPieceComponent},
    sound::{spawn_sound, SoundEffects},
    ChessBoardRes, MagicRes,
};
use chess_foundation::{board_helper::board_row_col_to_square_index, Bitboard, ChessMove};

#[derive(Resource)]
pub struct PieceIsPickedUp {
    pub piece_type: Option<char>,
    pub piece_entity: Option<Entity>,
    pub original_row_col: (usize, usize),
    // pub target_row_col: (usize, usize),
    pub current_position: Vec3,
    pub is_dragging: bool,
}

impl Default for PieceIsPickedUp {
    fn default() -> Self {
        PieceIsPickedUp {
            piece_type: None,
            piece_entity: None,
            original_row_col: (0, 0),
            // target_row_col: (0, 0),
            current_position: Vec3::new(0.0, 0.0, 0.0),
            is_dragging: false,
        }
    }
}

pub fn handle_touch_input(
    mut chess_pickup_ew: EventWriter<PickUpPieceEvent>,
    mut chess_drag_ew: EventWriter<DragPieceEvent>,
    mut chess_drop_ew: EventWriter<DropPieceEvent>,
    piece_is_picked_up: Res<PieceIsPickedUp>,
    mut touch_er: EventReader<TouchInput>,
) {
    for ev in touch_er.read() {
        match ev.phase {
            TouchPhase::Started => {
                if !piece_is_picked_up.is_dragging {
                    chess_pickup_ew.send(PickUpPieceEvent {
                        position: ev.position,
                    });
                }
            }
            TouchPhase::Moved => {
                chess_drag_ew.send(DragPieceEvent {
                    position: ev.position,
                });
            }
            TouchPhase::Ended => {
                if piece_is_picked_up.is_dragging {
                    chess_drop_ew.send(DropPieceEvent {
                        position: ev.position,
                    });
                }
            }
            TouchPhase::Canceled => {
                if piece_is_picked_up.is_dragging {
                    chess_drop_ew.send(DropPieceEvent {
                        position: ev.position,
                    });
                }
            }
        }
    }
}

pub fn handle_mouse_input(
    q_windows: Query<&Window, With<PrimaryWindow>>,
    mouse_button_input: ResMut<'_, ButtonInput<MouseButton>>,
    mut chess_pickup_ew: EventWriter<PickUpPieceEvent>,
    mut chess_drag_ew: EventWriter<DragPieceEvent>,
    mut chess_drop_ew: EventWriter<DropPieceEvent>,
) {
    if let Some(window) = q_windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if mouse_button_input.just_pressed(MouseButton::Left) {
                chess_pickup_ew.send(PickUpPieceEvent {
                    position: cursor_position,
                });
            } else if mouse_button_input.pressed(MouseButton::Left) {
                chess_drag_ew.send(DragPieceEvent {
                    position: cursor_position,
                });
            } else if mouse_button_input.just_released(MouseButton::Left) {
                chess_drop_ew.send(DropPieceEvent {
                    position: cursor_position,
                });
            }
        }
    }
}

fn board_coords_to_chess_coords(board_coords: Vec2, square_size: f32) -> (usize, usize) {
    let col = (board_coords.x / square_size).floor() as usize;
    let row = 7 - (board_coords.y / square_size).floor() as usize; // Assuming an 8x8 chess board
    (col, row)
}

pub fn pick_up_piece(
    q_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    board_transform: Res<ChessBoardTransform>,
    board_dimensions: Res<BoardDimensions>,
    chess_board: Res<ChessBoardRes>,
    mut piece_is_picked_up: ResMut<PieceIsPickedUp>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    magic_res: Res<MagicRes>,
    mut commands: Commands,
    mut chess_input_er: EventReader<PickUpPieceEvent>,
) {
    if let Ok((camera, camera_transform)) = q_camera.get_single() {
        let mut position = Vec2::ZERO;
        for inp in chess_input_er.read() {
            position = inp.position;

            let board_coords = get_board_coords_from_cursor(
                position,
                camera,
                camera_transform,
                &board_transform,
                &board_dimensions,
            )
            .expect("Failed to get board coordinates"); // Consider handling this more gracefully

            let (col, row) =
                board_coords_to_chess_coords(board_coords, board_dimensions.square_size);

            // spawn debug pieces
            if let Some((entity, transform, chess_piece)) = piece_query
                .iter_mut()
                .find(|(_, _, chess_piece)| chess_piece.row == row && chess_piece.col == col)
            {
                piece_is_picked_up.piece_type = Some(chess_piece.piece_type);
                piece_is_picked_up.original_row_col = (row, col);
                piece_is_picked_up.current_position = transform.translation;
                piece_is_picked_up.piece_entity = Some(entity);
                piece_is_picked_up.is_dragging = true;

                let is_white = true;
                // Continue with your logic, using `is_white` as needed
                let valid_moves = get_move_list_from_square(
                    board_row_col_to_square_index(row, col),
                    &chess_board.chess_board,
                    is_white,
                    &magic_res.magic,
                );

                commands.spawn(EnableDebugMarkers::new(valid_moves.clone()));

                println!("Picked up piece: {:?}", chess_piece.piece_type);
            }
        }
    }
}

pub fn drag_piece(
    q_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    board_transform: Res<ChessBoardTransform>,
    board_dimensions: Res<BoardDimensions>,
    piece_is_picked_up: ResMut<PieceIsPickedUp>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    mut chess_input_er: EventReader<DragPieceEvent>,
) {
    if let Ok((camera, camera_transform)) = q_camera.get_single() {
        if let Some(piece_entity) = piece_is_picked_up.piece_entity {
            if let Ok((_, mut transform, _)) = piece_query.get_mut(piece_entity) {
                for inp in chess_input_er.read() {
                    let board_coords = get_board_coords_from_cursor(
                        inp.position,
                        camera,
                        camera_transform,
                        &board_transform,
                        &board_dimensions,
                    )
                    .expect("Failed to get board coordinates"); // Consider handling this more gracefully

                    transform.translation = Vec3::new(
                        board_coords.x - board_dimensions.board_size.x / 2.0,
                        board_coords.y - board_dimensions.board_size.y / 2.0,
                        1.0,
                    );
                }
            }
        }
    }
}

pub fn drop_piece(
    q_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    board_transform: Res<ChessBoardTransform>,
    board_dimensions: Res<BoardDimensions>,
    mut piece_is_picked_up: ResMut<PieceIsPickedUp>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    mut chess_input_er: EventReader<DropPieceEvent>,
    mut debug_squares_query: Query<(Entity, &DebugSquare)>,
    mut commands: Commands,
    sound_effects: Res<SoundEffects>,
    mut chess_board: ResMut<ChessBoardRes>,
    mut refresh_pieces_events: EventWriter<RefreshPiecesFromBoardEvent>,
) {
    if let Ok((camera, camera_transform)) = q_camera.get_single() {
        for inp in chess_input_er.read() {
            //hide debug squares
            for (entity, _) in debug_squares_query.iter_mut() {
                commands.entity(entity).insert(Visibility::Hidden);
            }

            if let Some(piece_entity) = piece_is_picked_up.piece_entity {
                if let Ok((_, mut transform, mut chess_piece)) = piece_query.get_mut(piece_entity) {
                    let board_coords = get_board_coords_from_cursor(
                        inp.position,
                        camera,
                        camera_transform,
                        &board_transform,
                        &board_dimensions,
                    )
                    .expect("Failed to get board coordinates"); // Consider handling this more gracefully

                    let (col, row) =
                        board_coords_to_chess_coords(board_coords, board_dimensions.square_size);

                    print!("board_coords: {:?}", board_coords);
                    if (board_coords.x < 0.0)
                        || (board_coords.x > board_dimensions.board_size.x)
                        || (board_coords.y < 0.0)
                        || (board_coords.y > board_dimensions.board_size.y)
                        || (piece_is_picked_up.original_row_col.0 == row
                            && piece_is_picked_up.original_row_col.1 == col)
                    {
                        //trying to drop piece outside board, or same position as picked up
                        piece_is_picked_up.is_dragging = false;
                        let (original_row, original_col) = piece_is_picked_up.original_row_col;
                        transform.translation = chess_coord_to_board(
                            original_row,
                            original_col,
                            board_dimensions.square_size,
                            Vec3::new(
                                -board_dimensions.board_size.x / 2.0,
                                board_dimensions.board_size.y / 2.0,
                                0.5,
                            ),
                        );
                        return;
                    }

                    chess_piece.row = row;
                    chess_piece.col = col;
                    piece_is_picked_up.is_dragging = false;

                    let mut the_move = ChessMove::new(
                        board_row_col_to_square_index(
                            piece_is_picked_up.original_row_col.0,
                            piece_is_picked_up.original_row_col.1,
                        ),
                        board_row_col_to_square_index(row, col),
                    );

                    let is_capture = !(chess_board.chess_board.get_all_pieces()
                        & Bitboard::from_square_index(the_move.target_square()))
                    .is_empty();

                    if chess_board.chess_board.make_move(&mut the_move) {
                        //if chess_board.chess_board.get_piece_at(col)
                        if is_capture {
                            spawn_sound(&mut commands, &sound_effects, "capture.ogg");
                        } else {
                            spawn_sound(&mut commands, &sound_effects, "move-self.ogg");
                        }
                        println!("Released piece at row: {}, col: {}", row, col);
                    }
                    refresh_pieces_events.send(RefreshPiecesFromBoardEvent);
                }
            }
        }
    }
}
