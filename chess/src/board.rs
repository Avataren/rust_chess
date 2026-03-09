use bevy::{prelude::*, window::WindowResized};
use bevy::ecs::message::MessageReader;
use crate::game_resources::PlayerColor;

#[derive(Component)]
pub struct ResolutionText;

#[derive(Component)]
pub struct BoardTag;

#[derive(Resource)]
pub struct BoardDimensions {
    pub board_size: Vec2,
    pub square_size: f32,
    pub scale_factor: f32,
}

#[derive(Resource)]
pub struct ResolutionInfo {
    pub width: f32,
    pub height: f32,
}

#[derive(Resource, Deref, DerefMut)]
pub struct ChessBoardTransform {
    pub transform: Mat4,
}

impl Default for BoardDimensions {
    fn default() -> Self {
        BoardDimensions {
            board_size: Vec2::new(1024.0, 1024.0),
            square_size: 128.0,
            scale_factor: 1.0,
        }
    }
}

pub fn spawn_board(commands: &mut Commands, asset_server: Res<AssetServer>) {
    let board_texture_handle = asset_server.load("embedded://chess/assets/board.png");
    let board_transform = Transform::from_scale(Vec3::ONE);
    let board_transform_resource = ChessBoardTransform {
        transform: board_transform.to_matrix(),
    };

    commands.insert_resource(board_transform_resource);

    commands
        .spawn((
            Sprite {
                image: board_texture_handle,
                ..Default::default()
            },
            board_transform,
        ))
        .insert(BoardTag);
}

pub fn handle_resize_event(
    mut resolution: ResMut<ResolutionInfo>,
    mut events_reader: MessageReader<WindowResized>,
) {
    for event in events_reader.read() {
        resolution.width = event.width as f32;
        resolution.height = (event.height - 100.0) as f32;
    }
}

pub fn resize_board(
    resolution: ResMut<ResolutionInfo>,
    mut board_dimensions: ResMut<BoardDimensions>,
    mut board_query: Query<(&mut Transform, &Sprite), With<BoardTag>>,
    images: Res<Assets<Image>>,
    mut board_transform: ResMut<ChessBoardTransform>,
    player_color: Res<PlayerColor>,
    //mut q: Query<&mut Text, With<ResolutionText>>,
    _time: Res<Time>,
) {
    for (mut transform, sprite) in board_query.iter_mut() {
        if let Some(texture) = images.get(&sprite.image) {
            let texture_aspect_ratio = texture.size().x as f32 / texture.size().y as f32;
            let window_aspect_ratio = resolution.width / resolution.height;

            let scale = if window_aspect_ratio > texture_aspect_ratio {
                resolution.height / texture.size().y as f32
            } else {
                resolution.width / texture.size().x as f32
            };

            board_dimensions.scale_factor = scale;
            transform.scale = Vec3::new(scale, scale, 1.0);
            transform.rotation = if *player_color == PlayerColor::Black {
                Quat::from_rotation_z(std::f32::consts::PI)
            } else {
                Quat::IDENTITY
            };

            // let seconds = time.elapsed_seconds() as f32;
            // let angle_radians = (seconds * 6.0).to_radians();

            // transform.rotation = Quat::from_rotation_z(angle_radians);
            board_transform.transform = transform.to_matrix();
            board_dimensions.board_size =
                Vec2::new(texture.size().x as f32, texture.size().y as f32);
            board_dimensions.square_size = board_dimensions.board_size.x / 8.0;

            // let mut text = q.single_mut();
            // text.sections[0].value = format!("{:.1} x {:.1}", resolution.width, resolution.height);
        }
    }
}
