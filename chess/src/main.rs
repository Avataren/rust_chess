use bevy::{prelude::*};

mod board;
mod pieces;
mod piece_picker;

#[derive(Resource)]
struct ChessBoardRes {
    chess_board: chess_board::ChessBoard,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            (
                setup, pieces::preload_piece_sprites, 
                pieces::spawn_chess_pieces, 
                pieces::spawn_board_accessories
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

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
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
