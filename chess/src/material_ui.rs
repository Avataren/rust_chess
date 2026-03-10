use bevy::prelude::*;

use crate::{game_resources::PlayerColor, pieces::PieceTextures, ChessBoardRes};

#[derive(Component)]
pub struct TopMaterialLabel;

#[derive(Component)]
pub struct BottomMaterialLabel;

const ICON_SIZE: f32 = 30.0;

const START_PAWNS:   i32 = 8;
const START_KNIGHTS: i32 = 2;
const START_BISHOPS: i32 = 2;
const START_ROOKS:   i32 = 2;
const START_QUEENS:  i32 = 1;

struct PieceCounts {
    pawns:   i32,
    knights: i32,
    bishops: i32,
    rooks:   i32,
    queens:  i32,
}

impl PieceCounts {
    fn from_board(res: &ChessBoardRes, is_white: bool) -> Self {
        let b = &res.chess_board;
        let side = if is_white { b.get_white() } else { b.get_black() };
        Self {
            pawns:   (side & b.get_pawns()).count_ones()   as i32,
            knights: (side & b.get_knights()).count_ones() as i32,
            bishops: (side & b.get_bishops()).count_ones() as i32,
            rooks:   (side & b.get_rooks()).count_ones()   as i32,
            queens:  (side & b.get_queens()).count_ones()  as i32,
        }
    }

    fn material(&self) -> i32 {
        self.pawns
            + self.knights * 3
            + self.bishops * 3
            + self.rooks   * 5
            + self.queens  * 9
    }
}

/// Returns texture keys (e.g. "wq", "bp") for pieces captured *from* the given side,
/// sorted high-value first.
fn captured_keys(counts: &PieceCounts, captured_side_is_white: bool) -> Vec<&'static str> {
    let (q, r, b, n, p) = if captured_side_is_white {
        ("wq", "wr", "wb", "wn", "wp")
    } else {
        ("bq", "br", "bb", "bn", "bp")
    };

    let mut keys = Vec::new();
    for _ in 0..(START_QUEENS  - counts.queens ).max(0) { keys.push(q); }
    for _ in 0..(START_ROOKS   - counts.rooks  ).max(0) { keys.push(r); }
    for _ in 0..(START_BISHOPS - counts.bishops).max(0) { keys.push(b); }
    for _ in 0..(START_KNIGHTS - counts.knights).max(0) { keys.push(n); }
    for _ in 0..(START_PAWNS   - counts.pawns  ).max(0) { keys.push(p); }
    keys
}

fn spawn_side_children(
    commands:       &mut Commands,
    container:      Entity,
    piece_textures: &PieceTextures,
    keys:           &[&'static str],
    adv_str:        String,
) {
    commands.entity(container).despawn_related::<Children>();

    if keys.is_empty() && adv_str.is_empty() {
        return;
    }

    // Collect handles before entering the closure so we don't borrow piece_textures inside.
    let handles: Vec<Handle<Image>> = keys
        .iter()
        .filter_map(|k| piece_textures.textures.get(*k).cloned())
        .collect();

    commands.entity(container).with_children(move |parent| {
        for handle in handles {
            parent.spawn((
                Node {
                    width:  Val::Px(ICON_SIZE),
                    height: Val::Px(ICON_SIZE),
                    ..default()
                },
                ImageNode {
                    image: handle,
                    ..default()
                },
            ));
        }
        if !adv_str.is_empty() {
            parent.spawn((
                Text::new(adv_str),
                TextFont { font_size: 18.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.9)),
            ));
        }
    });
}

pub fn spawn_material_ui(mut commands: Commands) {
    let row_node = || Node {
        position_type:  PositionType::Absolute,
        flex_direction: FlexDirection::Row,
        align_items:    AlignItems::Center,
        column_gap:     Val::Px(1.0),
        left:           Val::Px(14.0),
        ..default()
    };

    commands.spawn((
        {
            let mut n = row_node();
            n.top = Val::Px(10.0);
            n
        },
        TopMaterialLabel,
    ));

    commands.spawn((
        {
            let mut n = row_node();
            n.bottom = Val::Px(10.0);
            n
        },
        BottomMaterialLabel,
    ));
}

pub fn update_material_ui(
    chess_board:    Res<ChessBoardRes>,
    player_color:   Res<PlayerColor>,
    piece_textures: Res<PieceTextures>,
    mut commands:   Commands,
    top_q:          Query<Entity, With<TopMaterialLabel>>,
    bot_q:          Query<Entity, With<BottomMaterialLabel>>,
) {
    if !chess_board.is_changed() && !player_color.is_changed() {
        return;
    }

    let white = PieceCounts::from_board(&chess_board, true);
    let black = PieceCounts::from_board(&chess_board, false);
    let adv   = white.material() - black.material(); // >0 white winning

    let from_black = captured_keys(&black, false); // black pieces taken by white
    let from_white = captured_keys(&white, true);  // white pieces taken by black

    let white_is_bottom = *player_color == PlayerColor::White;
    let (bottom_keys, top_keys): (Vec<&'static str>, Vec<&'static str>) = if white_is_bottom {
        (from_black, from_white)
    } else {
        (from_white, from_black)
    };

    let adv_label = if adv != 0 { format!(" +{}", adv.unsigned_abs()) } else { String::new() };
    let bottom_adv = if adv != 0 && (adv > 0) == white_is_bottom { adv_label.clone() } else { String::new() };
    let top_adv    = if adv != 0 && (adv > 0) != white_is_bottom { adv_label }          else { String::new() };

    if let Ok(top_entity) = top_q.single() {
        spawn_side_children(&mut commands, top_entity, &piece_textures, &top_keys, top_adv);
    }
    if let Ok(bot_entity) = bot_q.single() {
        spawn_side_children(&mut commands, bot_entity, &piece_textures, &bottom_keys, bottom_adv);
    }
}
