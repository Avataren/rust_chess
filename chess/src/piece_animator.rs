use bevy_tweening::{lens::TransformPositionLens, Animator, Tween, EaseFunction};
use bevy::prelude::*;
use std::time::Duration;

fn animate_piece_move(commands: &mut Commands, entity: Entity, start_pos: Vec3, end_pos: Vec3) {
    let tween = Tween::new(
        EaseFunction::QuadraticInOut,
        Duration::from_secs(1),
        TransformPositionLens {
            start: start_pos,
            end: end_pos,
        },
    );

    commands.entity(entity).insert(Animator::new(tween));
}
