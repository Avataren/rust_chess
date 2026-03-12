use bevy::{ecs::message::MessageWriter, prelude::*, window::WindowResolution};
use bevy_fps_counter::{FpsCounterPlugin, FpsCounterText};

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;
mod board;
mod board_accessories;
mod chess_event_handler;
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
use game_resources::{CurrentOpening, Difficulty, GameOverState, GamePhase, IsAiThinking, OpeningBookRes, PendingGameOver, PendingMoveSound, PlayerColor, PonderState};
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

fn main() {
    if cfg!(debug_assertions) {
        std::env::set_var("RUST_BACKTRACE", "1");
    }
    //env::set_var("WGPU_BACKEND", "dx12");
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                canvas: Some("#game-canvas".to_string()),
                title: "XavChess".to_string(),
                resizable: true,
                //mode: WindowMode::BorderlessFullscreen,
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
            ).chain(),
        )
        .add_systems(
            Update,
            (
                game_over_ui::handle_game_over_input,
                game_over_ui::handle_restart_button,
                start_screen_ui::handle_difficulty_buttons,
                start_screen_ui::handle_start_buttons,
            ),
        )
        //.add_systems(PreUpdate, ())
        .run();
}

fn initialize_game(mut refresh_pieces_events: MessageWriter<RefreshPiecesFromBoardEvent>) {
    refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2d);
    //setup_ui(&mut commands);
    // Resources

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
    commands.insert_resource(Difficulty::default());
    commands.insert_resource(PonderState::default());
    commands.insert_resource(ResolutionInfo {
        width: 1280.0,
        height: 1080.0,
    });

    let chessboard = chess_board::ChessBoard::new();
    //let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq";
    //let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    // chessboard.set_from_fen("8/8/8/8/7q/8/6k1/4K3 w  - 0 1");
    //chessboard.set_from_fen("5K2/4Q3/3q4/1k6/8/8/8/8 w  - 0 1");
    //chessboard.set_from_fen("4K3/4Q3/3q4/1k6/8/8/8/8 w  - 0 1");
    //chessboard.set_from_fen("3Q1K2/8/3q4/1k6/8/8/8/8 w  - 0 1");
    //chessboard.set_from_fen("8/8/5p1/4p3/4K3/8/8/8 w KQkq - 0 1");
    // chessboard.clear();
    // chessboard.set_piece_at_square(4, chess_foundation::piece::PieceType::King, true);
    // chessboard.set_piece_at_square(19, chess_foundation::piece::PieceType::Pawn, false);
    // chessboard.set_piece_at_square(20, chess_foundation::piece::PieceType::Queen, false);
    //chessboard.set_from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");
    commands.insert_resource(ChessBoardRes {
        chess_board: chessboard,
        //chess_board: chess_board::ChessBoard::new(),
    });

    let magic = PieceConductor::new();
    let book = chess_evaluation::OpeningBook::build(&magic);
    commands.insert_resource(OpeningBookRes { book });
    commands.insert_resource(PieceConductorRes { magic });

    board::spawn_board(&mut commands, asset_server);
}

// fn setup_ui(commands: &mut Commands) {
//     // Node that fills entire background
//     commands
//         .spawn(NodeBundle {
//             style: Style {
//                 width: Val::Percent(100.),
//                 top: Val::Px(50.0),
//                 ..default()
//             },
//             ..default()
//         })
//         .with_children(|root| {
//             // Text where we display current resolution
//             root.spawn((
//                 TextBundle::from_section(
//                     "Resolution",
//                     TextStyle {
//                         font_size: 20.0,
//                         ..default()
//                     },
//                 ),
//                 board::ResolutionText,
//             ));
//         });
// }
