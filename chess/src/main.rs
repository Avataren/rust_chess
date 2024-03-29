use bevy::{audio::AudioPlugin, prelude::*, window::WindowResolution};
use bevy_fps_counter::FpsCounterPlugin;
mod board;
mod board_accessories;
mod chess_event_handler;
mod game_events;
mod game_resources;
mod keyboard_input;
mod piece_picker;
mod pieces;
mod sound;
mod embed_plugin;
mod preload_assets_plugin;
mod input_plugin;
use bevy_tweening::TweeningPlugin;
use board::ResolutionInfo;

use embed_plugin::EmbeddedAssetPlugin;
use game_events::{
   RefreshPiecesFromBoardEvent,
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
                resolution: WindowResolution::new(1280., 1280.),
                prevent_default_event_handling: false,
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
        .add_systems(
            Startup,
            (
                initialize_game,
                board_accessories::spawn_board_accessories,
                board_accessories::spawn_debug_markers,
            )
        )
        .add_systems(
            Update,
            (
                board::handle_resize_event,
                board::resize_board,
                
                board_accessories::update_marker_square,
                board_accessories::update_debug_squares,
                chess_event_handler::on_tween_completed,
            ).chain(),
        )
        //.add_systems(PreUpdate, ())
        .run();
}

fn initialize_game(mut refresh_pieces_events: EventWriter<RefreshPiecesFromBoardEvent>) {
    refresh_pieces_events.send(RefreshPiecesFromBoardEvent);
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2dBundle::default());
    //setup_ui(&mut commands);
    // Resources

    commands.insert_resource(piece_picker::PieceIsPickedUp::default());
    commands.insert_resource(board::BoardDimensions::default());
    commands.insert_resource(pieces::PieceTextures::default());
    commands.insert_resource(game_resources::ValidMoves::new());
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
