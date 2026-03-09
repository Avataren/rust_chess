use bevy::{ecs::message::MessageWriter, prelude::*};

use crate::{
    game_events::{ChessAction, ChessEvent},
    game_resources::{GamePhase, PlayerColor},
};

#[derive(Component)]
pub struct StartScreenOverlay;

#[derive(Component)]
pub enum ColorButton {
    White,
    Black,
}

pub fn spawn_start_screen(mut commands: Commands) {
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                align_items: AlignItems::Center,
                justify_content: JustifyContent::Center,
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(40.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.08, 0.08, 0.12, 0.93)),
            StartScreenOverlay,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("XavChess"),
                TextFont {
                    font_size: 96.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));

            parent.spawn((
                Text::new("Choose your side"),
                TextFont {
                    font_size: 36.0,
                    ..default()
                },
                TextColor(Color::srgba(0.75, 0.75, 0.75, 1.0)),
            ));

            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(40.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_color_button(row, "♔  Play as White", ColorButton::White);
                    spawn_color_button(row, "♚  Play as Black", ColorButton::Black);
                });
        });
}

fn spawn_color_button(parent: &mut ChildSpawnerCommands, label: &str, button: ColorButton) {
    let (bg, text_color) = match button {
        ColorButton::White => (
            Color::srgb(0.93, 0.93, 0.88),
            Color::srgb(0.1, 0.1, 0.1),
        ),
        ColorButton::Black => (
            Color::srgb(0.18, 0.18, 0.22),
            Color::WHITE,
        ),
    };

    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(48.0), Val::Px(24.0)),
                border: UiRect::all(Val::Px(2.0)),
                ..default()
            },
            BackgroundColor(bg),
            button,
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                TextFont {
                    font_size: 34.0,
                    ..default()
                },
                TextColor(text_color),
            ));
        });
}

pub fn handle_start_buttons(
    interaction_query: Query<(&Interaction, &ColorButton), Changed<Interaction>>,
    mut overlay_query: Query<&mut Visibility, With<StartScreenOverlay>>,
    mut game_phase: ResMut<GamePhase>,
    mut player_color: ResMut<PlayerColor>,
    mut chess_ew: MessageWriter<ChessEvent>,
) {
    for (interaction, button) in interaction_query.iter() {
        if *interaction != Interaction::Pressed {
            continue;
        }
        match button {
            ColorButton::White => {
                *player_color = PlayerColor::White;
            }
            ColorButton::Black => {
                *player_color = PlayerColor::Black;
                // AI plays white and moves first
                chess_ew.write(ChessEvent::new(ChessAction::MakeMove));
            }
        }
        *game_phase = GamePhase::Playing;
        for mut vis in overlay_query.iter_mut() {
            *vis = Visibility::Hidden;
        }
    }
}

pub fn update_start_screen_visibility(
    game_phase: Res<GamePhase>,
    mut overlay_query: Query<&mut Visibility, With<StartScreenOverlay>>,
) {
    if !game_phase.is_changed() {
        return;
    }
    let visible = match *game_phase {
        GamePhase::StartScreen => Visibility::Visible,
        GamePhase::Playing => Visibility::Hidden,
    };
    for mut vis in overlay_query.iter_mut() {
        *vis = visible;
    }
}
