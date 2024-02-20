use bevy::{
    math::vec3,
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    utils::HashMap,
    window::PrimaryWindow,
};

use crate::{
    board::{BoardDimensions, BoardTag, ChessBoardTransform},
    ChessBoardRes,
};

const CHESSPIECE_SCALE: f32 = 0.8;

#[derive(Resource)]
pub struct PieceTextures {
    pub textures: HashMap<String, Handle<Image>>,
}

impl Default for PieceTextures {
    fn default() -> Self {
        PieceTextures {
            textures: HashMap::default(),
        }
    }
}

#[derive(Component)]
pub struct ChessPiece {
    pub piece_type: char,
    pub row: usize,
    pub col: usize,
}
pub fn preload_piece_sprites(
    asset_server: Res<AssetServer>,
    mut piece_textures: ResMut<PieceTextures>,
) {
    let pieces = [
        "bb", "bk", "bn", "bp", "bq", "br", "wb", "wk", "wn", "wp", "wq", "wr",
    ];
    for &piece in pieces.iter() {
        let texture_handle = asset_server.load(format!("{}.png", piece));
        piece_textures
            .textures
            .insert(piece.to_string(), texture_handle);
    }
}

// Helper function to get board coordinates from window cursor position
pub fn get_board_coords_from_cursor(
    cursor_position: Vec2,
    camera: &Camera,
    camera_transform: &GlobalTransform,
    board_transform: &ChessBoardTransform,
    board_dimensions: &BoardDimensions,
) -> Option<Vec2> {
    camera
        .viewport_to_world(camera_transform, cursor_position)
        .map(|ray| ray.origin.truncate())
        .map(|world_position| {
            let matrix = board_transform.transform.inverse();
            let world_coords = matrix
                .transform_point3(world_position.extend(0.0))
                .truncate();
            let board_offset = Vec3::new(
                -board_dimensions.board_size.x / 2.0,
                board_dimensions.board_size.y / 2.0,
                0.0,
            );
            world_coords + Vec2::new(-board_offset.x, board_offset.y)
        })
}

pub fn spawn_chess_pieces(
    mut commands: Commands,
    chess_board_res: Res<ChessBoardRes>,
    piece_textures: Res<PieceTextures>,
    board_dimensions: Res<BoardDimensions>,
    query: Query<Entity, With<BoardTag>>,
) {
    let square_size = board_dimensions.square_size;
    let board_offset = Vec3::new(
        -board_dimensions.board_size.x / 2.0,
        board_dimensions.board_size.y / 2.0,
        0.0,
    );

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
                let child_sprite = commands
                    .spawn(SpriteBundle {
                        texture: texture_handle.clone(),
                        transform: Transform {
                            translation: world_position,
                            scale: vec3(CHESSPIECE_SCALE, CHESSPIECE_SCALE, 1.0),
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                    .insert(ChessPiece {
                        piece_type: piece_char,
                        row: row,
                        col: col,
                    })
                    .id();

                commands.entity(parent).push_children(&[child_sprite]);
            }
        }
    }
}

pub fn chess_coord_to_board(row: usize, col: usize, square_size: f32, board_offset: Vec3) -> Vec3 {
    Vec3::new(
        col as f32 * square_size + board_offset.x + square_size / 2.0, // Center in square horizontally
        -(row as f32) * square_size + board_offset.y - square_size / 2.0, // Center in square vertically
        0.5,
    )
}
