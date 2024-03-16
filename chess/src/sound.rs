//use web_sys; // Add this line to import the web_sys crate
use bevy::{
    asset::Handle, audio::PlaybackMode,  prelude::*, utils::HashMap,
};

#[derive(Resource)]
pub struct SoundEffects {
    pub sounds: HashMap<String, Handle<AudioSource>>,
}

#[derive(Resource)]
pub struct SoundsPreloaded(pub bool);

#[derive(Component)]
pub struct SoundEffect;

impl Default for SoundEffects {
    fn default() -> Self {
        SoundEffects {
            sounds: HashMap::default(),
        }
    }
}

pub fn preload_sounds(
    asset_server: Res<AssetServer>,
    mut sounds: ResMut<SoundEffects>,
    // mut sounds_preloaded: ResMut<SoundsPreloaded>,
    // mut ev_mouse: EventReader<MouseButtonInput>,
    // mut ev_touch: EventReader<TouchInput>,
    // q_windows: Query<&Window, With<PrimaryWindow>>
) {
    // if sounds_preloaded.0 {
    //     return;
    // }

    // // Check if there are any mouse button or touch input events
    // if !ev_mouse.read().next().is_none() || !ev_touch.read().next().is_none() {
    //     sounds_preloaded.0 = true;
    // }
    // if !sounds_preloaded.0 {
    //     return;
    // }

    let sound_files = [
        "notify.ogg",
        "move-self.ogg",
        "capture.ogg",
        "move-check.ogg",
        "castle.ogg",
    ];
    for &sound in sound_files.iter() {
        let asset_path = format!("embedded://chess/assets/{}", sound);
        sounds
            .sounds
            .insert(sound.to_string(), asset_server.load(asset_path));
    }
}

pub fn spawn_sound(commands: &mut Commands, sound_effects: &SoundEffects, sound_name: &str) {
    if let Some(sound_handle) = sound_effects.sounds.get(sound_name) {
        commands
            .spawn(AudioBundle {
                source: sound_handle.clone(),
                settings: PlaybackSettings {
                    mode: PlaybackMode::Despawn,
                    ..Default::default()
                },
                ..Default::default()
            })
            .insert(SoundEffect);
    }
}

// pub fn manage_sounds(
//     mut commands: Commands,
//     q_audio: Query<(Entity, &AudioSink), With<SoundEffect>>,
// ) {
//     for (entity, audiosink) in q_audio.iter() {
//         if audiosink.empty() {
//             commands.entity(entity).despawn();
//         }
//     }
// }
