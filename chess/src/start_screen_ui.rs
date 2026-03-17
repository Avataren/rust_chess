use bevy::{ecs::message::MessageWriter, prelude::*};

use crate::{
    game_events::{ChessAction, ChessEvent},
    game_resources::{GameClocks, GamePhase, GameSettings, PlayerColor, Strength, TimeControl},
};

#[derive(Component)]
pub struct StartScreenOverlay;

#[derive(Component)]
pub enum ColorButton {
    White,
    Black,
}

#[derive(Component, PartialEq, Clone, Copy)]
pub enum TimeControlButton {
    Bullet,
    Blitz,
    Rapid,
}

impl TimeControlButton {
    fn to_time_control(self) -> TimeControl {
        match self {
            TimeControlButton::Bullet => TimeControl::Bullet,
            TimeControlButton::Blitz  => TimeControl::Blitz,
            TimeControlButton::Rapid  => TimeControl::Rapid,
        }
    }
}

#[derive(Component, PartialEq, Clone, Copy)]
pub enum StrengthButton {
    S1,
    S2,
    S3,
    S4,
    S5,
}

impl StrengthButton {
    fn to_strength(self) -> Strength {
        match self {
            StrengthButton::S1 => Strength::S1,
            StrengthButton::S2 => Strength::S2,
            StrengthButton::S3 => Strength::S3,
            StrengthButton::S4 => Strength::S4,
            StrengthButton::S5 => Strength::S5,
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
                row_gap: Val::Px(16.0),
                overflow: Overflow::scroll_y(),
                padding: UiRect::axes(Val::Px(8.0), Val::Px(12.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.08, 0.08, 0.12, 0.93)),
            StartScreenOverlay,
        ))
        .with_children(|parent| {
            // Title
            parent.spawn((
                Text::new("XavChess"),
                TextFont { font_size: 60.0, ..default() },
                TextColor(Color::WHITE),
            ));

            // Time Control label
            parent.spawn((
                Text::new("Time Control"),
                TextFont { font_size: 22.0, ..default() },
                TextColor(Color::srgba(0.75, 0.75, 0.75, 1.0)),
            ));

            // Time Control buttons row
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    justify_content: JustifyContent::Center,
                    column_gap: Val::Px(12.0),
                    row_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_option_button(row, "Bullet  2+1", TimeControlButton::Bullet, false);
                    spawn_option_button(row, "Blitz   5+3", TimeControlButton::Blitz,  true);
                    spawn_option_button(row, "Rapid 10+5",  TimeControlButton::Rapid,  false);
                });

            // Strength label
            parent.spawn((
                Text::new("Strength"),
                TextFont { font_size: 22.0, ..default() },
                TextColor(Color::srgba(0.75, 0.75, 0.75, 1.0)),
            ));

            // Strength buttons row
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    justify_content: JustifyContent::Center,
                    column_gap: Val::Px(12.0),
                    row_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_option_button(row, "Beginner", StrengthButton::S1, false);
                    spawn_option_button(row, "Casual",   StrengthButton::S2, false);
                    spawn_option_button(row, "Club",     StrengthButton::S3, true);
                    spawn_option_button(row, "Strong",   StrengthButton::S4, false);
                    spawn_option_button(row, "Maximum",  StrengthButton::S5, false);
                });

            // Color label
            parent.spawn((
                Text::new("Choose your side"),
                TextFont { font_size: 22.0, ..default() },
                TextColor(Color::srgba(0.75, 0.75, 0.75, 1.0)),
            ));

            // Color buttons row
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    justify_content: JustifyContent::Center,
                    column_gap: Val::Px(20.0),
                    row_gap: Val::Px(10.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_color_button(row, "♚  Play as Black", ColorButton::Black);
                    spawn_color_button(row, "♔  Play as White", ColorButton::White);
                });
        });
}

fn button_colors(selected: bool) -> (Color, Color) {
    if selected {
        (Color::srgb(0.95, 0.95, 0.95), Color::srgb(0.08, 0.08, 0.08))
    } else {
        (Color::srgb(0.18, 0.18, 0.20), Color::srgb(0.60, 0.60, 0.65))
    }
}

fn spawn_option_button<C: Component>(
    parent: &mut ChildSpawnerCommands,
    label: &str,
    button: C,
    selected: bool,
) {
    let (bg, text_color) = button_colors(selected);
    parent
        .spawn((
            Button,
            Node {
                padding: UiRect::axes(Val::Px(20.0), Val::Px(12.0)),
                ..default()
            },
            BackgroundColor(bg),
            button,
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                TextFont { font_size: 22.0, ..default() },
                TextColor(text_color),
            ));
        });
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
                padding: UiRect::axes(Val::Px(28.0), Val::Px(16.0)),
                border: UiRect::all(Val::Px(2.0)),
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

pub fn handle_time_control_buttons(
    interaction_query: Query<(&Interaction, &TimeControlButton), Changed<Interaction>>,
    mut game_settings: ResMut<GameSettings>,
    mut game_clocks: ResMut<GameClocks>,
    mut button_query: Query<(&TimeControlButton, &mut BackgroundColor, &Children)>,
    mut text_query: Query<&mut TextColor>,
) {
    let mut changed = false;
    for (interaction, btn) in interaction_query.iter() {
        if *interaction == Interaction::Pressed {
            game_settings.time_control = btn.to_time_control();
            game_clocks.set_time_control(btn.to_time_control());
            changed = true;
        }
    }
    if !changed {
        return;
    }
    for (btn, mut bg, children) in button_query.iter_mut() {
        let selected = btn.to_time_control() == game_settings.time_control;
        let (new_bg, new_text_color) = button_colors(selected);
        *bg = BackgroundColor(new_bg);
        for child in children.iter() {
            if let Ok(mut tc) = text_query.get_mut(child) {
                *tc = TextColor(new_text_color);
            }
        }
    }
}

pub fn handle_strength_buttons(
    interaction_query: Query<(&Interaction, &StrengthButton), Changed<Interaction>>,
    mut game_settings: ResMut<GameSettings>,
    mut button_query: Query<(&StrengthButton, &mut BackgroundColor, &Children)>,
    mut text_query: Query<&mut TextColor>,
) {
    let mut changed = false;
    for (interaction, btn) in interaction_query.iter() {
        if *interaction == Interaction::Pressed {
            game_settings.strength = btn.to_strength();
            changed = true;
        }
    }
    if !changed {
        return;
    }
    for (btn, mut bg, children) in button_query.iter_mut() {
        let selected = btn.to_strength() == game_settings.strength;
        let (new_bg, new_text_color) = button_colors(selected);
        *bg = BackgroundColor(new_bg);
        for child in children.iter() {
            if let Ok(mut tc) = text_query.get_mut(child) {
                *tc = TextColor(new_text_color);
            }
        }
    }
}

pub fn handle_start_buttons(
    interaction_query: Query<(&Interaction, &ColorButton), Changed<Interaction>>,
    mut overlay_query: Query<&mut Visibility, With<StartScreenOverlay>>,
    mut game_phase: ResMut<GamePhase>,
    mut player_color: ResMut<PlayerColor>,
    mut chess_ew: MessageWriter<ChessEvent>,
    mut game_clocks: ResMut<GameClocks>,
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
        game_clocks.reset();
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
