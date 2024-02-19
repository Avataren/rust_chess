use bevy::{prelude::*, window::WindowResolution};
use bevy_wasm_window_resize::WindowResizePlugin;
mod board;
mod pieces;
mod piece_picker;

#[derive(Resource)]
struct ChessBoardRes {
    chess_board: chess_board::ChessBoard,
}

fn main() {
    print!("Starting chess game");

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                canvas: Some("#game-canvas".to_string()),
                resolution: WindowResolution::new(1280., 1024.),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(WindowResizePlugin)
        .add_systems(
            Startup,
            (
                setup, pieces::preload_piece_sprites, 
                pieces::spawn_chess_pieces, 
                pieces::spawn_board_accessories,
                // board::set_initial_board_size
            ).chain(),
        )
        .add_systems(Update, 
            (
                board::resize_board,
                piece_picker::handle_pick_and_drag_piece,
                pieces::update_marker_square
            ).chain())
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>, mut windows: Query<&mut Window>) {
    let mut window = windows.single_mut();
    window.set_maximized(true);    
    commands.spawn(Camera2dBundle::default());
    board::spawn_board(&mut commands, asset_server);
    // Resources
    commands.insert_resource(piece_picker::PieceIsPickedUp::default());
    commands.insert_resource(board::BoardDimensions::default()); 
    commands.insert_resource(pieces::PieceTextures::default());
    commands.insert_resource(ChessBoardRes {
        chess_board: chess_board::ChessBoard::new(),
    });
}
