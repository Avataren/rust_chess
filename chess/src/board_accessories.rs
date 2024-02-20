use bevy::{
    math::vec3,
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    utils::HashMap,
    window::PrimaryWindow,
};

use crate::{
    board::{BoardDimensions, BoardTag, ChessBoardTransform},
    pieces::{chess_coord_to_board, get_board_coords_from_cursor},
    ChessBoardRes,
};

/// Updates the position of the marker square based on the current cursor position.

#[derive(Component)]
pub struct MarkerSquare;

#[derive(Component)]
pub struct DebugSquare {
    pub row: usize,
    pub col: usize,
}

pub fn spawn_board_accessories(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    board_dimensions: Res<BoardDimensions>,
    query: Query<Entity, With<BoardTag>>,
) {
    let parent = query.iter().next().unwrap();
    let board_offset = Vec3::new(
        -board_dimensions.board_size.x / 2.0,
        board_dimensions.board_size.y / 2.0,
        0.0,
    );
    let mut world_position = chess_coord_to_board(1, 1, board_dimensions.square_size, board_offset);
    world_position.z = 0.1;
    let square_size = board_dimensions.square_size;
    let rect = meshes.add(Rectangle::new(square_size, square_size));

    let child_mesh = commands
        .spawn(MaterialMesh2dBundle {
            mesh: Mesh2dHandle(rect),
            material: materials.add(Color::rgba(0.0, 1.0, 0.0, 0.15)),
            transform: Transform {
                translation: world_position,
                ..Default::default()
            },
            ..Default::default()
        })
        .insert(MarkerSquare)
        .id();
    commands.entity(parent).push_children(&[child_mesh]);
}

pub fn spawn_debug_markers(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    board_dimensions: Res<BoardDimensions>,
    query: Query<Entity, With<BoardTag>>,
) {
    let parent = query.iter().next().unwrap();
    let board_offset = Vec3::new(
        -board_dimensions.board_size.x / 2.0,
        board_dimensions.board_size.y / 2.0,
        0.0,
    );
    let square_size = board_dimensions.square_size;
    let rect = meshes.add(Rectangle::new(square_size, square_size));

    for col in 0..8 {
        for row in 0..8 {
            let mut world_position =
                chess_coord_to_board(row, col, board_dimensions.square_size, board_offset);
            world_position.z = 0.25;

            let child_mesh = commands
                .spawn(MaterialMesh2dBundle {
                    mesh: Mesh2dHandle(rect.clone()),
                    material: materials.add(Color::rgba(0.0, 0.4, 0.8, 0.5)),
                    transform: Transform {
                        translation: world_position,
                        ..Default::default()
                    },
                    visibility: Visibility::Hidden,
                    ..Default::default()
                })
                .insert(DebugSquare { row, col })
                .id();
            commands.entity(parent).push_children(&[child_mesh]);
        }
    }
}

pub fn update_marker_square(
    q_windows: Query<&Window, With<PrimaryWindow>>,
    q_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    mut query: Query<&mut Transform, With<MarkerSquare>>,
    board_transform: Res<ChessBoardTransform>,
    board_dimensions: Res<BoardDimensions>,
) {
    if let Some(window) = q_windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if let Ok((camera, camera_transform)) = q_camera.get_single() {
                update_marker_position(
                    cursor_position,
                    &board_transform,
                    &board_dimensions,
                    camera,
                    camera_transform,
                    &mut query,
                );
            }
        }
    }
}

fn update_marker_position(
    cursor_position: Vec2,
    board_transform: &ChessBoardTransform,
    board_dimensions: &BoardDimensions,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    query: &mut Query<&mut Transform, With<MarkerSquare>>,
) {
    if let Some(board_coords) = get_board_coords_from_cursor(
        cursor_position,
        camera,
        camera_transform,
        board_transform,
        board_dimensions,
    ) {
        let square_size = board_dimensions.square_size;
        let col = (board_coords.x / square_size).floor() as i32;
        let row = 7 - ((board_coords.y / square_size).floor() as i32);

        if (0..=7).contains(&col) && (0..=7).contains(&row) {
            let square_coords = chess_coord_to_board(
                row as usize,
                col as usize,
                square_size,
                Vec3::new(
                    -board_dimensions.board_size.x / 2.0,
                    board_dimensions.board_size.y / 2.0,
                    0.0,
                ),
            );

            if let Some(mut marker_square) = query.iter_mut().next() {
                marker_square.translation = Vec3::new(square_coords.x, square_coords.y, 0.1);
            }
        }
    }
}
