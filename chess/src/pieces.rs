
use bevy::{
    math::vec3, prelude::*, sprite::{MaterialMesh2dBundle, Mesh2dHandle}, utils::HashMap, window::{PrimaryWindow}
};

use crate::{board::{BoardDimensions, BoardTag, ChessBoardTransform}, ChessBoardRes};

const CHESSPIECE_SCALE: f32 = 0.8;

#[derive(Resource)]
pub struct PieceTextures {
    pub textures: HashMap<String, Handle<Image>>,
}

#[derive(Component)]
pub struct MarkerSquare;

#[derive(Component)]
pub struct ChessPiece {
    pub piece_type: char,
    pub row: usize,
    pub col: usize
}

impl Default for PieceTextures {
    fn default() -> Self {
        PieceTextures {
            textures: HashMap::default(),
        }
    }
}

pub fn preload_piece_sprites(
    asset_server: Res<AssetServer>,
    mut piece_textures: ResMut<PieceTextures>,

) {
    let pieces = ["bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr"];
    for &piece in pieces.iter() {
        let texture_handle = asset_server.load(format!("{}.png", piece));
        piece_textures.textures.insert(piece.to_string(), texture_handle);
    }
}

pub fn spawn_board_accessories(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    board_dimensions: Res<BoardDimensions>,
    query: Query<Entity, With<BoardTag>>
){
    let parent = query.iter().next().unwrap();
    let board_offset = Vec3::new(-board_dimensions.board_size.x / 2.0, board_dimensions.board_size.y / 2.0, 0.0);    
    let mut world_position = chess_coord_to_board(1, 1, board_dimensions.square_size, board_offset);
    world_position.z = 0.1;
    let square_size = board_dimensions.square_size;
    let rect = meshes.add(Rectangle::new(square_size, square_size));

    let child_mesh = commands.spawn(MaterialMesh2dBundle {
        mesh: Mesh2dHandle(rect),
        material: materials.add(Color::rgba(0.0, 1.0, 0.0, 0.15)),
        transform:
         Transform{
            translation: world_position,
            ..Default::default()
        },
        ..Default::default()
    }).insert(MarkerSquare).id();
    commands.entity(parent).push_children(&[child_mesh]);
}

// Helper function to get board coordinates from window cursor position
pub fn get_board_coords_from_cursor(
    cursor_position: Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    board_transform: &ChessBoardTransform,
    board_dimensions: &BoardDimensions,
) -> Option<Vec2> {
    camera.viewport_to_world(camera_transform, cursor_position)
        .map(|ray| ray.origin.truncate())
        .map(|world_position| {
            let matrix = board_transform.transform.inverse();
            let world_coords = matrix.transform_point3(world_position.extend(0.0)).truncate();
            let board_offset = Vec3::new(
                -board_dimensions.board_size.x / 2.0,
                board_dimensions.board_size.y / 2.0,
                0.0,
            );
            world_coords + Vec2::new(-board_offset.x, board_offset.y)
        })
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

/// Updates the position of the marker square based on the current cursor position.
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

pub fn spawn_chess_pieces(
    mut commands: Commands,
    chess_board_res: Res<ChessBoardRes>,
    piece_textures: Res<PieceTextures>,
    board_dimensions: Res<BoardDimensions>,
    query: Query<Entity, With<BoardTag>>
) {
    let square_size = board_dimensions.square_size;
    let board_offset = Vec3::new(-board_dimensions.board_size.x / 2.0, board_dimensions.board_size.y / 2.0, 0.0);

    let parent = query.iter().next().unwrap();

    for i in 0..64 {
        let piece_char = chess_board_res.chess_board.get_piece_character(i);
        if piece_char != '.' {
            let row = 7 - i / 8;
            let col = i % 8;
            let mut world_position = chess_coord_to_board(row, col, square_size, board_offset);
            world_position.z = 100.0;
            let piece_texture_key = match piece_char {
                'P' => "wp",
                'R' => "wr",
                'N' => "wn",
                'B' => "wb",
                'Q' => "wq",
                'K' => "wk",
                'p' => "bp",
                'r' => "br",
                'n' => "bn",
                'b' => "bb",
                'q' => "bq",
                'k' => "bk",
                _ => continue,
            };
            if let Some(texture_handle) = piece_textures.textures.get(piece_texture_key) {


                let child_sprite = commands.spawn(SpriteBundle {
                    texture: texture_handle.clone(),
                    transform: Transform{
                        translation: world_position,
                        scale: vec3(CHESSPIECE_SCALE, CHESSPIECE_SCALE, 1.0),
                        ..Default::default()
                    },
                    ..Default::default()
                }).insert(ChessPiece {
                    piece_type: piece_char,
                    row: row,
                    col: col
                }).id();

                commands.entity(parent).push_children(&[child_sprite]);
            }
        }
    }
}

pub fn chess_coord_to_board(row: usize, col: usize, square_size: f32, board_offset: Vec3) -> Vec3 {
    Vec3::new(
        col as f32 * square_size + board_offset.x + square_size / 2.0, // Center in square horizontally
        -(row as f32) * square_size + board_offset.y  - square_size / 2.0, // Center in square vertically
        0.5,
    )
}
