use bevy::{prelude::*, window::WindowResolution};
mod board;
mod piece_picker;
mod pieces;
use board::ResolutionInfo;


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

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                canvas: Some("#game-canvas".to_string()),
                title: "XavChess".to_string(),
                resizable: true,
                resolution: WindowResolution::new(1280., 1024.),
                prevent_default_event_handling: false,
                ..default()
            }),
            ..default()
        }))
        // .add_plugins(WindowResizePlugin)
        .add_systems(
            Startup,
            (
                setup,
                pieces::preload_piece_sprites,
                pieces::spawn_chess_pieces,
                pieces::spawn_board_accessories,
            )
                .chain(),
        )
        .add_systems(
            Update,
            (
                board::handle_resize_event,
                board::resize_board,
                piece_picker::handle_pick_and_drag_piece,
                pieces::update_marker_square,
            )
                .chain(),
        )
        .run();
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2dBundle::default());
    setup_ui(&mut commands);
    board::spawn_board(&mut commands, asset_server);
    // Resources
    commands.insert_resource(piece_picker::PieceIsPickedUp::default());
    commands.insert_resource(board::BoardDimensions::default());
    commands.insert_resource(pieces::PieceTextures::default());
    commands.insert_resource(ResolutionInfo{ width: 1280.0, height: 1080.0 });
    commands.insert_resource(ChessBoardRes {
        chess_board: chess_board::ChessBoard::new(),
    });
    #[cfg(target_arch = "wasm32")]
    resize_canvas_and_apply_styles();
}

fn setup_ui(commands: &mut Commands) {
    // Node that fills entire background
    commands.spawn(NodeBundle {
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
