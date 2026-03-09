use bevy::prelude::*;

use crate::game_resources::GameOverState;

#[derive(Component)]
pub struct GameOverOverlay;

#[derive(Component)]
pub struct GameOverText;

pub fn spawn_game_over_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(24.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.65)),
            Visibility::Hidden,
            GameOverOverlay,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new(""),
                TextFont {
                    font_size: 80.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                GameOverText,
            ));
            parent.spawn((
                Text::new("Press R to restart"),
                TextFont {
                    font_size: 36.0,
                    ..default()
                },
                TextColor(Color::srgba(0.8, 0.8, 0.8, 1.0)),
            ));
        });
}

pub fn update_game_over_ui(
    game_over_state: Res<GameOverState>,
    mut overlay_query: Query<&mut Visibility, With<GameOverOverlay>>,
    mut text_query: Query<&mut Text, With<GameOverText>>,
) {
    if !game_over_state.is_changed() {
        return;
    }
    let (visible, message) = match *game_over_state {
        GameOverState::Playing => (Visibility::Hidden, ""),
        GameOverState::PlayerWins => (Visibility::Visible, "You Win!"),
        GameOverState::OpponentWins => (Visibility::Visible, "You Lose!"),
        GameOverState::Stalemate => (Visibility::Visible, "Stalemate!"),
    };
    for mut vis in overlay_query.iter_mut() {
        *vis = visible;
    }
    for mut text in text_query.iter_mut() {
        **text = message.to_string();
    }
}
