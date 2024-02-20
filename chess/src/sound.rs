use bevy::{asset::Handle, prelude::*, utils::HashMap};

#[derive(Resource)]
pub struct SoundEffects {
    pub sounds: HashMap<String, Handle<AudioSource>>,
}

#[derive(Component)]
pub struct SoundEffect;

impl Default for SoundEffects {
    fn default() -> Self {
        SoundEffects {
            sounds: HashMap::default(),
        }
    }
}

pub fn preload_sounds(asset_server: Res<AssetServer>, mut sounds: ResMut<SoundEffects>) {
    let sound_files = [
        "notify.ogg",
        "move-self.ogg",
        "capture.ogg",
        "move-check.ogg",
        "castle.ogg",
    ];
    for &sound in sound_files.iter() {
        sounds
            .sounds
            .insert(sound.to_string(), asset_server.load(sound));
    }
}

pub fn spawn_sound(commands: &mut Commands, sound_effects: &SoundEffects, sound_name: &str) {
    if let Some(sound_handle) = sound_effects.sounds.get(sound_name) {
        commands
            .spawn(AudioBundle {
                source: sound_handle.clone(),
                ..Default::default()
            })
            .insert(SoundEffect);
    }
}

pub fn manage_sounds(
    mut commands: Commands,
    mut q_audio: Query<(Entity, &AudioSink), With<SoundEffect>>,
) {
    for (entity, audiosink) in q_audio.iter() {
        if audiosink.empty() {
            commands.entity(entity).despawn();
        }
    }
}
