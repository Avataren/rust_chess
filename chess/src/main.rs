use std::env;

use bevy::{
    audio::{AudioPlugin, SpatialScale},
    prelude::*,
    window::{WindowMode, WindowResolution},
};

mod board;
mod board_accessories;
mod chess_event_handler;
mod game_events;
mod keyboard_input;
mod piece_picker;
mod pieces;
mod sound;
mod game_resources;
use board::ResolutionInfo;

use game_events::{
    ChessEvent, DragPieceEvent, DropPieceEvent, PickUpPieceEvent, RefreshPiecesFromBoardEvent,
};
use move_generator::magic::Magic;

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
    //env::set_var("WGPU_BACKEND", "dx12");
    //env::set_var("RUST_BACKTRACE", "1");
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        canvas: Some("#game-canvas".to_string()),
                        title: "XavChess".to_string(),
                        resizable: true,
                        //mode: WindowMode::BorderlessFullscreen,
                        resolution: WindowResolution::new(1280., 1024.),
                        prevent_default_event_handling: false,
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..default()
                    }),
                    ..default()
                })
                // .set(AudioPlugin {
                //     default_spatial_scale: SpatialScale::new_2d(AUDIO_SCALE),
                //     ..default()
                // }),
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
                piece_picker::handle_mouse_input,
                piece_picker::handle_touch_input,
                keyboard_input::handle_keyboard_input,
                chess_event_handler::handle_chess_events,
                piece_picker::pick_up_piece,
                piece_picker::drag_piece,
                piece_picker::drop_piece,
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
    commands.insert_resource(game_resources::ValidMoves::new());
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
    commands.insert_resource(Events::<ChessEvent>::default());
    commands.insert_resource(Events::<PickUpPieceEvent>::default());
    commands.insert_resource(Events::<DragPieceEvent>::default());
    commands.insert_resource(Events::<DropPieceEvent>::default());
    board::spawn_board(&mut commands, asset_server);
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
