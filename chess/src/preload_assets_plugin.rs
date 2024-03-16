use bevy::prelude::*;
use pieces::preload_piece_sprites;
use sound::preload_sounds;

use crate::{
    pieces,
    sound::{self, SoundsPreloaded},
};

pub struct PreloadAssetsPlugin;

impl Plugin for PreloadAssetsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SoundsPreloaded(false))
            .insert_resource(sound::SoundEffects::default())
            .add_systems(PostStartup, (preload_piece_sprites, preload_sounds));
    }
}
