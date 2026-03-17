use bevy::{ecs::message::MessageWriter, prelude::*, window::WindowResolution};
use bevy_fps_counter::{FpsCounterPlugin, FpsCounterText};

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;
mod board;
mod board_accessories;
mod chess_event_handler;
mod clock_ui;
mod game_events;
mod game_over_ui;
mod game_resources;
mod opening_name_ui;
mod start_screen_ui;
mod keyboard_input;
mod piece_picker;
mod pieces;
mod sound;
mod embed_plugin;
mod material_ui;
mod preload_assets_plugin;
mod input_plugin;
use bevy_tweening::TweeningPlugin;
use board::ResolutionInfo;

use embed_plugin::EmbeddedAssetPlugin;
use game_events::{
   RefreshPiecesFromBoardEvent,
};
use chess_evaluation::{init_neural_eval_from_bytes, set_neural_eval_enabled};
use game_resources::{
    CurrentOpening, GameClocks, GameOverState, GamePhase, GameSettings, IsAiThinking,
    OpeningBookRes, PendingGameOver, PendingMoveSound, PlayerColor, PonderState, TimeControl,
};
use input_plugin::ChessInputPlugin;
use move_generator::piece_conductor::PieceConductor;
use preload_assets_plugin::PreloadAssetsPlugin;


#[derive(Resource)]
struct ChessBoardRes {
    chess_board: chess_board::ChessBoard,
}

#[derive(Resource)]
struct PieceConductorRes {
    magic: move_generator::piece_conductor::PieceConductor,
}



fn reposition_fps_counter(mut q: Query<&mut Node, With<FpsCounterText>>) {
    for mut node in q.iter_mut() {
        node.position_type = PositionType::Absolute;
        node.top   = Val::Px(8.0);
        node.right = Val::Px(14.0);
        node.left  = Val::Auto;
    }
}

/// NNUE weights embedded at compile time.
/// Update by copying a new .npz to chess_evaluation/src/eval.npz and rebuilding.
static NNUE_WEIGHTS: &[u8] = include_bytes!("../../chess_evaluation/src/eval.npz");

fn init_neural_eval() {
    match init_neural_eval_from_bytes(NNUE_WEIGHTS) {
        Ok(()) => {
            set_neural_eval_enabled(true);
            eprintln!("Neural eval loaded ({} KB)", NNUE_WEIGHTS.len() / 1024);
        }
        Err(e) => eprintln!("Neural eval failed to load: {e}"),
    }
}

fn main() {
    init_neural_eval();
    if cfg!(debug_assertions) {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                canvas: Some("#game-canvas".to_string()),
                title: "XavChess".to_string(),
                resizable: true,
                resolution: WindowResolution::new(1280, 1280),
                prevent_default_event_handling: true,
                present_mode: bevy::window::PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EmbeddedAssetPlugin)
        .add_plugins(TweeningPlugin)
        .add_plugins(FpsCounterPlugin)
        .add_plugins(PreloadAssetsPlugin)
        .add_plugins(ChessInputPlugin)
        .add_systems(PreStartup, setup)
        .add_systems(PostStartup, reposition_fps_counter)
        .add_systems(
            Startup,
            (
                initialize_game,
                board_accessories::spawn_board_accessories,
                board_accessories::spawn_last_move_highlights,
                board_accessories::spawn_debug_markers,
                game_over_ui::spawn_game_over_ui,
                start_screen_ui::spawn_start_screen,
                opening_name_ui::spawn_opening_name_ui,
                opening_name_ui::spawn_thinking_indicator,
                material_ui::spawn_material_ui,
                clock_ui::spawn_clock_ui,
            )
        )
        .add_systems(
            Update,
            (
                board::handle_resize_event,
                board::resize_board,
                pieces::sync_piece_rotations,

                board_accessories::update_marker_square,
                board_accessories::update_last_move_highlights,
                board_accessories::update_debug_squares,
                chess_event_handler::on_tween_completed,
                game_over_ui::update_game_over_ui,
                start_screen_ui::update_start_screen_visibility,
                opening_name_ui::detect_opening,
                opening_name_ui::update_opening_name_ui,
                opening_name_ui::update_thinking_ui,
                material_ui::update_material_ui,
                clock_ui::update_clocks,
                clock_ui::update_clock_ui,
            ).chain(),
        )
        .add_systems(
            Update,
            (
                game_over_ui::handle_game_over_input,
                game_over_ui::handle_restart_button,
                start_screen_ui::handle_time_control_buttons,
                start_screen_ui::handle_strength_buttons,
                start_screen_ui::handle_start_buttons,
            ),
        )
        .run();
}

fn initialize_game(mut refresh_pieces_events: MessageWriter<RefreshPiecesFromBoardEvent>) {
    refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2d);

    commands.insert_resource(piece_picker::PieceIsPickedUp::default());
    commands.insert_resource(board::BoardDimensions::default());
    commands.insert_resource(pieces::PieceTextures::default());
    commands.insert_resource(game_resources::ValidMoves::new());
    commands.insert_resource(game_resources::LastMove::default());
    commands.insert_resource(GameOverState::default());
    commands.insert_resource(PendingGameOver::default());
    commands.insert_resource(PlayerColor::default());
    commands.insert_resource(GamePhase::default());
    commands.insert_resource(CurrentOpening::default());
    commands.insert_resource(IsAiThinking::default());
    commands.insert_resource(PendingMoveSound::default());
    commands.insert_resource(GameSettings::default());
    commands.insert_resource(GameClocks::new(TimeControl::Blitz));
    commands.insert_resource(PonderState::default());
    commands.insert_resource(ResolutionInfo {
        width: 1280.0,
        height: 1080.0,
    });

    let chessboard = chess_board::ChessBoard::new();
    commands.insert_resource(ChessBoardRes {
        chess_board: chessboard,
    });

    let magic = PieceConductor::new();
    let book = chess_evaluation::OpeningBook::build(&magic);
    commands.insert_resource(OpeningBookRes { book });
    commands.insert_resource(PieceConductorRes { magic });

    board::spawn_board(&mut commands, asset_server);
}
