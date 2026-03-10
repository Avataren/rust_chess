use bevy::{ecs::message::MessageWriter, prelude::*};

use crate::{
    game_events::{ChessAction, ChessEvent},
    game_resources::{Difficulty, GamePhase, PlayerColor},
};

#[derive(Component)]
pub struct StartScreenOverlay;

#[derive(Component)]
pub enum ColorButton {
    White,
    Black,
}

#[derive(Component, PartialEq, Clone, Copy)]
pub enum DifficultyButton {
    Easy,
    Medium,
    Hard,
    VeryHard,
}

impl DifficultyButton {
    fn to_difficulty(self) -> Difficulty {
        match self {
            DifficultyButton::Easy     => Difficulty::Easy,
            DifficultyButton::Medium   => Difficulty::Medium,
            DifficultyButton::Hard     => Difficulty::Hard,
            DifficultyButton::VeryHard => Difficulty::VeryHard,
        }
    }
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
                row_gap: Val::Px(36.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.08, 0.08, 0.12, 0.93)),
            StartScreenOverlay,
        ))
        .with_children(|parent| {
            // Title
            parent.spawn((
                Text::new("XavChess"),
                TextFont { font_size: 96.0, ..default() },
                TextColor(Color::WHITE),
            ));

            // Difficulty label
            parent.spawn((
                Text::new("Difficulty"),
                TextFont { font_size: 28.0, ..default() },
                TextColor(Color::srgba(0.75, 0.75, 0.75, 1.0)),
            ));

            // Difficulty buttons row
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(24.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_difficulty_button(row, "Easy",      DifficultyButton::Easy,     false);
                    spawn_difficulty_button(row, "Medium",    DifficultyButton::Medium,   true);
                    spawn_difficulty_button(row, "Hard",      DifficultyButton::Hard,     false);
                    spawn_difficulty_button(row, "Very Hard", DifficultyButton::VeryHard, false);
                });

            // Color label
            parent.spawn((
                Text::new("Choose your side"),
                TextFont { font_size: 28.0, ..default() },
                TextColor(Color::srgba(0.75, 0.75, 0.75, 1.0)),
            ));

            // Color buttons row
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

fn spawn_difficulty_button(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    button: DifficultyButton,
    selected: bool,
) {
    let (bg, text_color) = difficulty_colors(button, selected);
    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(36.0), Val::Px(16.0)),
                ..default()
            },
            BackgroundColor(bg),
            button,
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                TextFont { font_size: 28.0, ..default() },
                TextColor(text_color),
            ));
        });
}

fn difficulty_colors(_button: DifficultyButton, selected: bool) -> (Color, Color) {
    if selected {
        // Bright white background, dark text — clearly active.
        (Color::srgb(0.95, 0.95, 0.95), Color::srgb(0.08, 0.08, 0.08))
    } else {
        // Neutral dark background, muted text — clearly inactive.
        (Color::srgb(0.18, 0.18, 0.20), Color::srgb(0.60, 0.60, 0.65))
    }
}

fn spawn_color_button(parent: &mut ChildSpawnerCommands, label: &str, button: ColorButton) {
    let (bg, text_color) = match button {
        ColorButton::White => (Color::srgb(0.93, 0.93, 0.88), Color::srgb(0.1, 0.1, 0.1)),
        ColorButton::Black => (Color::srgb(0.18, 0.18, 0.22), Color::WHITE),
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
                TextFont { font_size: 34.0, ..default() },
                TextColor(text_color),
            ));
        });
}

pub fn handle_difficulty_buttons(
    interaction_query: Query<(&Interaction, &DifficultyButton), Changed<Interaction>>,
    mut difficulty: ResMut<Difficulty>,
    mut button_query: Query<(&DifficultyButton, &mut BackgroundColor)>,
) {
    let mut changed = false;
    for (interaction, btn) in interaction_query.iter() {
        if *interaction == Interaction::Pressed {
            *difficulty = btn.to_difficulty();
            changed = true;
        }
    }
    if !changed {
        return;
    }
    for (btn, mut bg) in button_query.iter_mut() {
        let selected = btn.to_difficulty() == *difficulty;
        let (new_bg, _) = difficulty_colors(*btn, selected);
        *bg = BackgroundColor(new_bg);
    }
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
