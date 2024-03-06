use bevy::{math::vec3, prelude::*, utils::HashMap};
use bevy_svg::prelude::*;

use crate::{
    board::{BoardDimensions, BoardTag, ChessBoardTransform},
    game_events::RefreshPiecesFromBoardEvent,
    ChessBoardRes,
};

const CHESSPIECE_SCALE: f32 = 1.6;
//const CHESSPIECE_SCALE: f32 = 0.8;

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
pub struct ChessPieceComponent {
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

#[derive(Resource)]
pub struct PieceSVGs {
    pub svgs: HashMap<String, Handle<Svg>>,
}

impl Default for PieceSVGs {
    fn default() -> Self {
        PieceSVGs {
            svgs: HashMap::default(),
        }
    }
}

pub fn preload_piece_svgs(asset_server: Res<AssetServer>, mut piece_svgs: ResMut<PieceSVGs>) {
    let pieces = [
        "bB", "bK", "bN", "bP", "bQ", "bR", "wB", "wK", "wN", "wP", "wQ", "wR",
    ];
    for &piece in pieces.iter() {
        let svg = asset_server.load(format!("kosal/{}.svg", piece));
        piece_svgs.svgs.insert(piece.to_string(), svg);
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
    piece_textures: Res<PieceSVGs>,
    board_dimensions: Res<BoardDimensions>,
    query: Query<Entity, With<BoardTag>>,
    query_existing: Query<Entity, With<ChessPieceComponent>>,
    mut refresh_pieces_events: EventReader<RefreshPiecesFromBoardEvent>,
) {
    for _ev in refresh_pieces_events.read() {
        // //first clear all existing pieces
        let entities_to_despawn = query_existing.iter().collect::<Vec<Entity>>();
        for entity in entities_to_despawn {
            commands.entity(entity).despawn_recursive();
        }

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
                world_position.z = 0.5;
                world_position.x -= square_size / 2.0;
                world_position.y += square_size / 2.0;
                let piece_texture_key = match piece_char {
                    'P' => "wP",
                    'R' => "wR",
                    'N' => "wN",
                    'B' => "wB",
                    'Q' => "wQ",
                    'K' => "wK",
                    'p' => "bP",
                    'r' => "bR",
                    'n' => "bN",
                    'b' => "bB",
                    'q' => "bQ",
                    'k' => "bK",
                    _ => continue,
                };
                if let Some(texture_handle) = piece_textures.svgs.get(piece_texture_key) {
                    // let child_sprite = commands
                    //     .spawn(SpriteBundle {
                    //         texture: texture_handle.clone(),
                    //         transform: Transform {
                    //             translation: world_position,
                    //             scale: vec3(CHESSPIECE_SCALE, CHESSPIECE_SCALE, 1.0),
                    //             ..Default::default()
                    //         },
                    //         ..Default::default()
                    //     })
                    //     .insert(ChessPieceComponent {
                    //         piece_type: piece_char,
                    //         row: row,
                    //         col: col,
                    //     })
                    //     .id();

                    let child_sprite = commands
                        .spawn(Svg2dBundle {
                            svg: texture_handle.clone(),
                            //origin: Origin::Center, // Origin::TopLeft is the default
                            transform: Transform {
                                translation: world_position,
                                scale: vec3(CHESSPIECE_SCALE, CHESSPIECE_SCALE, 1.0),
                                // rotation: Quat::from_rotation_x(-std::f32::consts::PI / 5.0),
                                ..Default::default()
                            },
                            ..Default::default()
                        })
                        .insert(ChessPieceComponent {
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
}

pub fn chess_coord_to_board(row: usize, col: usize, square_size: f32, board_offset: Vec3) -> Vec3 {
    Vec3::new(
        col as f32 * square_size + board_offset.x + square_size / 2.0, // Center in square horizontally
        -(row as f32) * square_size + board_offset.y - square_size / 2.0, // Center in square vertically
        0.5,
    )
}
