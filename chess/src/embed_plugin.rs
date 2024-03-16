use bevy::{
    app::{App, Plugin},
    asset::embedded_asset,
};

pub struct EmbeddedAssetPlugin;

impl Plugin for EmbeddedAssetPlugin {
    fn build(&self, app: &mut App) {
        // We get to choose some prefix relative to the workspace root which
        // will be ignored in "embedded://" asset paths.
        // Path to asset must be relative to this file, because that's how
        // include_bytes! works.
        let omit_prefix = "";
        embedded_asset!(app, omit_prefix, "assets/bb.png");
        embedded_asset!(app, omit_prefix, "assets/bk.png");
        embedded_asset!(app, omit_prefix, "assets/bn.png");
        embedded_asset!(app, omit_prefix, "assets/bp.png");
        embedded_asset!(app, omit_prefix, "assets/bq.png");
        embedded_asset!(app, omit_prefix, "assets/br.png");
        embedded_asset!(app, omit_prefix, "assets/wb.png");
        embedded_asset!(app, omit_prefix, "assets/wk.png");
        embedded_asset!(app, omit_prefix, "assets/wn.png");
        embedded_asset!(app, omit_prefix, "assets/wp.png");
        embedded_asset!(app, omit_prefix, "assets/wq.png");
        embedded_asset!(app, omit_prefix, "assets/wr.png");
        embedded_asset!(app, omit_prefix, "assets/board.png");
        embedded_asset!(app, omit_prefix, "assets/capture.ogg");
        embedded_asset!(app, omit_prefix, "assets/move-self.ogg");
        embedded_asset!(app, omit_prefix, "assets/move-check.ogg");
        embedded_asset!(app, omit_prefix, "assets/notify.ogg");
        embedded_asset!(app, omit_prefix, "assets/castle.ogg");
    }
}
