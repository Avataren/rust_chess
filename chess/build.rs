use std::env;
use std::path::Path;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let asset_dir = Path::new(&manifest_dir).join("assets");

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("assets");

    std::fs::create_dir_all(&dest_path).unwrap();
    
    for entry in std::fs::read_dir(asset_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() {
            std::fs::copy(path.clone(), dest_path.join(path.file_name().unwrap())).unwrap();
        }
    }
}


