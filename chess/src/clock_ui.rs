use bevy::prelude::*;
use std::time::Duration;

use crate::{
    game_resources::{GameClocks, GameOverState, GamePhase, PendingGameOver, PlayerColor},
    ChessBoardRes,
};

#[derive(Component)]
pub struct PlayerClockText;

#[derive(Component)]
pub struct OpponentClockText;

#[derive(Component)]
pub struct ClockPanel;

pub fn spawn_clock_ui(mut commands: Commands) {
    // Opponent clock — top right
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                right: Val::Px(10.0),
                padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.85)),
            Visibility::Hidden,
            ClockPanel,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("--:--"),
                TextFont { font_size: 32.0, ..default() },
                TextColor(Color::WHITE),
                OpponentClockText,
            ));
        });

    // Player clock — bottom right
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(10.0),
                right: Val::Px(10.0),
                padding: UiRect::axes(Val::Px(12.0), Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.85)),
            Visibility::Hidden,
            ClockPanel,
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("--:--"),
                TextFont { font_size: 32.0, ..default() },
                TextColor(Color::WHITE),
                PlayerClockText,
            ));
        });
}

fn format_clock(d: Duration) -> String {
    let secs = d.as_secs();
    if secs >= 10 {
        let m = secs / 60;
        let s = secs % 60;
        format!("{}:{:02}", m, s)
    } else {
        let tenths = d.subsec_millis() / 100;
        format!("{}.{}", secs, tenths)
    }
}

fn clock_color(d: Duration) -> Color {
    let secs = d.as_secs();
    if secs < 10 {
        Color::srgb(0.9, 0.2, 0.2)   // red
    } else if secs < 30 {
        Color::srgb(0.95, 0.65, 0.1) // orange
    } else {
        Color::WHITE
    }
}

pub fn update_clock_ui(
    game_clocks: Res<GameClocks>,
    player_color: Res<PlayerColor>,
    game_phase: Res<GamePhase>,
    mut panel_query: Query<&mut Visibility, With<ClockPanel>>,
    mut player_query: Query<(&mut Text, &mut TextColor), (With<PlayerClockText>, Without<OpponentClockText>)>,
    mut opponent_query: Query<(&mut Text, &mut TextColor), (With<OpponentClockText>, Without<PlayerClockText>)>,
) {
    let visible = *game_phase == GamePhase::Playing;
    for mut vis in panel_query.iter_mut() {
        *vis = if visible { Visibility::Visible } else { Visibility::Hidden };
    }
    if !visible {
        return;
    }

    let player_is_white = *player_color == PlayerColor::White;
    let player_rem = game_clocks.remaining(player_is_white);
    let opp_rem = game_clocks.remaining(!player_is_white);

    for (mut text, mut color) in player_query.iter_mut() {
        **text = format_clock(player_rem);
        *color = TextColor(clock_color(player_rem));
    }
    for (mut text, mut color) in opponent_query.iter_mut() {
        **text = format_clock(opp_rem);
        *color = TextColor(clock_color(opp_rem));
    }
}

/// Ticks the active side's clock and triggers a flag loss when time expires.
pub fn update_clocks(
    time: Res<Time>,
    mut game_clocks: ResMut<GameClocks>,
    chess_board: Res<ChessBoardRes>,
    game_phase: Res<GamePhase>,
    game_over_state: Res<GameOverState>,
    mut pending_game_over: ResMut<PendingGameOver>,
    player_color: Res<PlayerColor>,
) {
    if *game_phase != GamePhase::Playing || *game_over_state != GameOverState::Playing {
        return;
    }
    // Don't tick if a game-over is already pending (avoid double-flagging).
    if pending_game_over.0.is_some() {
        return;
    }

    let active_is_white = chess_board.chess_board.is_white_active();
    game_clocks.tick(active_is_white, time.delta());

    if game_clocks.is_flagged(active_is_white) {
        let player_is_white = *player_color == PlayerColor::White;
        let outcome = if active_is_white == player_is_white {
            // Player's clock hit zero
            GameOverState::OpponentWins
        } else {
            // AI's clock hit zero
            GameOverState::PlayerWins
        };
        pending_game_over.0 = Some(outcome);
    }
}
