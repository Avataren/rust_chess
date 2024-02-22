use std::env;

use bevy::{
    audio::{AudioPlugin, SpatialScale},
    prelude::*,
    window::WindowResolution,
};

mod board;
mod board_accessories;

mod piece_picker;
mod pieces;
mod sound;
use board::ResolutionInfo;

use move_generator::{magic::Magic};
use pieces::RefreshPiecesFromBoardEvent;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    #[wasm_bindgen(js_name = resizeCanvasAndApplyStyles)]
    fn resize_canvas_and_apply_styles();
}

#[derive(Resource)]
struct ChessBoardRes {
    chess_board: chess_board::ChessBoard,
}

#[derive(Resource)]
struct MagicRes {
    magic: move_generator::magic::Magic,
}

const AUDIO_SCALE: f32 = 1. / 100.0;

fn main() {
    env::set_var("WGPU_BACKEND", "dx12");
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        canvas: Some("#game-canvas".to_string()),
                        title: "XavChess".to_string(),
                        resizable: true,
                        resolution: WindowResolution::new(1280., 1024.),
                        prevent_default_event_handling: false,
                        ..default()
                    }),
                    ..default()
                })
                .set(AudioPlugin {
                    default_spatial_scale: SpatialScale::new_2d(AUDIO_SCALE),
                    ..default()
                }),
        )
        // .add_plugins(WindowResizePlugin)
        .add_systems(
            Startup,
            (
                setup,
                pieces::preload_piece_sprites,
                sound::preload_sounds,
                initialize_game,
                board_accessories::spawn_board_accessories,
                board_accessories::spawn_debug_markers,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                board::handle_resize_event,
                board::resize_board,
                piece_picker::handle_pick_and_drag_piece,
                board_accessories::update_marker_square,
                board_accessories::update_debug_squares,
                sound::manage_sounds,
                pieces::spawn_chess_pieces,

            )
                .chain(),
        )
        .run();
}

fn initialize_game(mut refresh_pieces_events: EventWriter<RefreshPiecesFromBoardEvent>) {
    refresh_pieces_events.send(RefreshPiecesFromBoardEvent);
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2dBundle::default());
    setup_ui(&mut commands);
    // Resources
    commands.insert_resource(piece_picker::PieceIsPickedUp::default());
    commands.insert_resource(board::BoardDimensions::default());
    commands.insert_resource(pieces::PieceTextures::default());
    commands.insert_resource(sound::SoundEffects::default());
    commands.insert_resource(ResolutionInfo {
        width: 1280.0,
        height: 1080.0,
    });
    commands.insert_resource(ChessBoardRes {
        chess_board: chess_board::ChessBoard::new(),
    });

    let _chessboard = chess_board::ChessBoard::new();
    let magic = Magic::new();
    commands.insert_resource(MagicRes { magic });
    commands.insert_resource(Events::<RefreshPiecesFromBoardEvent>::default());
    board::spawn_board(&mut commands, asset_server);

    #[cfg(target_arch = "wasm32")]
    resize_canvas_and_apply_styles();
}

fn setup_ui(commands: &mut Commands) {
    // Node that fills entire background
    commands
        .spawn(NodeBundle {
            style: Style {
                width: Val::Percent(100.),
                ..default()
            },
            ..default()
        })
        .with_children(|root| {
            // Text where we display current resolution
            root.spawn((
                TextBundle::from_section(
                    "Resolution",
                    TextStyle {
                        font_size: 20.0,
                        ..default()
                    },
                ),
                board::ResolutionText,
            ));
        });
}
