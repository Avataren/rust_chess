use bevy::prelude::*;

use crate::{
    game_resources::{CurrentOpening, OpeningBookRes},
    ChessBoardRes,
};

#[derive(Component)]
pub struct OpeningNameText;

pub fn spawn_opening_name_ui(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(12.0),
            left: Val::Percent(50.0),
            ..default()
        },
        Text::new(""),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::srgba(0.9, 0.9, 0.9, 0.85)),
        OpeningNameText,
    ));
}

/// Every frame: probe the book for a name. When one is found, update the
/// resource (which the UI text picks up via change detection).
pub fn detect_opening(
    chess_board: Res<ChessBoardRes>,
    opening_book: Res<OpeningBookRes>,
    mut current_opening: ResMut<CurrentOpening>,
) {
    if let Some(name) = opening_book.book.probe_name(&chess_board.chess_board) {
        if current_opening.0 != name {
            current_opening.0 = name.to_string();
        }
    }
}

/// Sync the displayed text when `CurrentOpening` changes.
pub fn update_opening_name_ui(
    current_opening: Res<CurrentOpening>,
    mut query: Query<&mut Text, With<OpeningNameText>>,
) {
    if !current_opening.is_changed() {
        return;
    }
    for mut text in query.iter_mut() {
        **text = current_opening.0.clone();
    }
}
