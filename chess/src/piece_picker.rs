use bevy::{input::mouse::MouseButton, prelude::*, window::PrimaryWindow};

use crate::{
    board::{BoardDimensions, ChessBoardTransform},
    pieces::{chess_coord_to_board, get_board_coords_from_cursor, ChessPiece},
    sound::{spawn_sound, SoundEffects},
};

#[derive(Resource)]
pub struct PieceIsPickedUp {
    pub piece_type: Option<char>,
    pub piece_entity: Option<Entity>,
    pub original_row_col: (usize, usize),
    pub target_row_col: (usize, usize),
    pub current_position: Vec3,
    pub is_dragging: bool,
}

impl Default for PieceIsPickedUp {
    fn default() -> Self {
        PieceIsPickedUp {
            piece_type: None,
            piece_entity: None,
            original_row_col: (0, 0),
            target_row_col: (0, 0),
            current_position: Vec3::new(0.0, 0.0, 0.0),
            is_dragging: false,
        }
    }
}

pub fn handle_pick_and_drag_piece(
    q_windows: Query<&Window, With<PrimaryWindow>>,
    q_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPiece)>,
    board_transform: Res<ChessBoardTransform>,
    board_dimensions: Res<BoardDimensions>,
    mut piece_is_picked_up: ResMut<PieceIsPickedUp>,
    mouse_button_input: ResMut<'_, ButtonInput<MouseButton>>,
    sound_effects: Res<SoundEffects>,
    mut commands: Commands,
) {
    if let Some(window) = q_windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = q_camera.get_single() {
                handle_mouse_input(
                    &mouse_button_input,
                    cursor_position,
                    (camera, camera_transform),
                    &mut piece_is_picked_up,
                    &board_transform,
                    &board_dimensions,
                    &mut piece_query,
                    &sound_effects,
                    &mut commands,
                );
            }
        }
    }
}

fn handle_mouse_input(
    mouse_button_input: &ResMut<'_, ButtonInput<MouseButton>>,
    cursor_position: Vec2,
    camera_bundle: (&Camera, &GlobalTransform),
    piece_is_picked_up: &mut PieceIsPickedUp,
    board_transform: &ChessBoardTransform,
    board_dimensions: &BoardDimensions,
    piece_query: &mut Query<(Entity, &mut Transform, &mut ChessPiece)>,
    sound_effects: &SoundEffects,
    commands: &mut Commands,
) {
    if mouse_button_input.pressed(MouseButton::Left) {
        if piece_is_picked_up.is_dragging {
            drag_piece(
                cursor_position,
                camera_bundle,
                board_transform,
                board_dimensions,
                piece_is_picked_up,
                piece_query,
            );
        } else {
            pick_up_piece(
                cursor_position,
                camera_bundle,
                board_transform,
                board_dimensions,
                piece_is_picked_up,
                piece_query,
            );
        }
    } else if mouse_button_input.just_released(MouseButton::Left) {
        spawn_sound(commands, &sound_effects, "move-self.ogg");
        release_piece(
            cursor_position,
            camera_bundle,
            board_transform,
            board_dimensions,
            piece_is_picked_up,
            piece_query,
        );
    }
}

fn board_coords_to_chess_coords(board_coords: Vec2, square_size: f32) -> (usize, usize) {
    let col = (board_coords.x / square_size).floor() as usize;
    let row = 7 - (board_coords.y / square_size).floor() as usize; // Assuming an 8x8 chess board
    (col, row)
}

fn pick_up_piece(
    cursor_position: Vec2,
    (camera, camera_transform): (&Camera, &GlobalTransform),
    board_transform: &ChessBoardTransform,
    board_dimensions: &BoardDimensions,
    piece_is_picked_up: &mut PieceIsPickedUp,
    piece_query: &mut Query<(Entity, &mut Transform, &mut ChessPiece)>,
) {
    let board_coords = get_board_coords_from_cursor(
        cursor_position,
        camera,
        camera_transform,
        board_transform,
        board_dimensions,
    )
    .expect("Failed to get board coordinates"); // Consider handling this more gracefully

    let (col, row) = board_coords_to_chess_coords(board_coords, board_dimensions.square_size);

    if let Some((entity, transform, chess_piece)) = piece_query
        .iter_mut()
        .find(|(_, _, chess_piece)| chess_piece.row == row && chess_piece.col == col)
    {
        piece_is_picked_up.piece_type = Some(chess_piece.piece_type);
        piece_is_picked_up.original_row_col = (row, col);
        piece_is_picked_up.current_position = transform.translation;
        piece_is_picked_up.piece_entity = Some(entity);
        piece_is_picked_up.is_dragging = true;

        println!("Picked up piece: {:?}", chess_piece.piece_type);
    }
}

fn drag_piece(
    cursor_position: Vec2,
    (camera, camera_transform): (&Camera, &GlobalTransform),
    board_transform: &ChessBoardTransform,
    board_dimensions: &BoardDimensions,
    piece_is_picked_up: &mut PieceIsPickedUp,
    piece_query: &mut Query<(Entity, &mut Transform, &mut ChessPiece)>,
) {
    if let Some(piece_entity) = piece_is_picked_up.piece_entity {
        if let Ok((_, mut transform, _)) = piece_query.get_mut(piece_entity) {
            let board_coords = get_board_coords_from_cursor(
                cursor_position,
                camera,
                camera_transform,
                board_transform,
                board_dimensions,
            )
            .expect("Failed to get board coordinates"); // Consider handling this more gracefully

            transform.translation = Vec3::new(
                board_coords.x - board_dimensions.board_size.x / 2.0,
                board_coords.y - board_dimensions.board_size.y / 2.0,
                transform.translation.z,
            );
        }
    }
}

fn release_piece(
    cursor_position: Vec2,
    (camera, camera_transform): (&Camera, &GlobalTransform),
    board_transform: &ChessBoardTransform,
    board_dimensions: &BoardDimensions,
    piece_is_picked_up: &mut PieceIsPickedUp,
    piece_query: &mut Query<(Entity, &mut Transform, &mut ChessPiece)>,
) {
    if let Some(piece_entity) = piece_is_picked_up.piece_entity {
        if let Ok((_, mut transform, mut chess_piece)) = piece_query.get_mut(piece_entity) {
            let board_coords = get_board_coords_from_cursor(
                cursor_position,
                camera,
                camera_transform,
                board_transform,
                board_dimensions,
            )
            .expect("Failed to get board coordinates"); // Consider handling this more gracefully

            let (col, row) =
                board_coords_to_chess_coords(board_coords, board_dimensions.square_size);

            transform.translation = chess_coord_to_board(
                row,
                col,
                board_dimensions.square_size,
                Vec3::new(
                    -board_dimensions.board_size.x / 2.0,
                    board_dimensions.board_size.y / 2.0,
                    transform.translation.z,
                ),
            );

            chess_piece.row = row;
            chess_piece.col = col;
            piece_is_picked_up.is_dragging = false;

            println!("Released piece at row: {}, col: {}", row, col);
        }
    }
}