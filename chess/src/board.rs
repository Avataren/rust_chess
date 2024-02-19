use bevy::{prelude::*, window::WindowResized};

#[derive(Component)]
pub struct BoardTag;

#[derive(Resource)]
pub struct BoardDimensions {
    pub board_size: Vec2,
    pub square_size: f32,
    pub scale_factor: f32,
}

#[derive(Resource, Deref, DerefMut)]
pub struct ChessBoardTransform {
    pub transform: Mat4
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

pub fn spawn_board(
    commands: &mut Commands,
    asset_server: Res<AssetServer>,
) {
    let board_texture_handle = asset_server.load("board.png");
    // Spawn the board with a tag to identify it later
    //boardTransform.compute_matrix().inverse()
    let board_transform = Transform::from_scale(Vec3::ONE);
    let board_transform_resource = ChessBoardTransform{
        transform: board_transform.compute_matrix()
    };
    
    commands.insert_resource(board_transform_resource);

    commands.spawn(SpriteBundle {
        texture: board_texture_handle,
        transform: board_transform,
        ..Default::default()
    }).insert(BoardTag);
}

pub fn resize_board(
    mut board_dimensions: ResMut<BoardDimensions>,
    mut board_query: Query<(&mut Transform, &Handle<Image>), With<BoardTag>>,
    mut events_reader: EventReader<WindowResized>,
    images: Res<Assets<Image>>,
    mut board_transform: ResMut<ChessBoardTransform>
) {
    for event in events_reader.read() {
        println! ("resize_board reading event");
        let window_width = event.width;
        let window_height = event.height - 100.0;

        for (mut transform, texture_handle) in board_query.iter_mut() {
            if let Some(texture) = images.get(texture_handle) {
                let texture_aspect_ratio = texture.size().x as f32 / texture.size().y as f32;
                let window_aspect_ratio = window_width / window_height;

                let scale = if window_aspect_ratio > texture_aspect_ratio {
                    window_height as f32 / texture.size().y as f32
                } else {
                    window_width as f32 / texture.size().x as f32
                };

                board_dimensions.scale_factor = scale;
                transform.scale = Vec3::new(scale, scale, 1.0);
                board_transform.transform = transform.compute_matrix();
                board_dimensions.board_size = Vec2::new(texture.size().x as f32 , texture.size().y as f32);
                board_dimensions.square_size = board_dimensions.board_size.x / 8.0;
            }
        }
    }
}