/// pool_tool — fast JSONL replay-pool management for the self-play training loop.
///
/// Subcommands:
///   append  --pool <path> --new-data <path> --pool-size <n>
///             Appends new-data to pool, trims to the newest pool-size lines.
///
///   split   --pool <path> --train <path> --val <path>
///             [--val-fraction 0.1] [--min-val 1000]
///             Shuffles pool in memory, writes train / val splits.
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};

use rand::seq::SliceRandom;
use rand::SeedableRng;

// ── arg parsing ──────────────────────────────────────────────────────────────

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == name)
        .map(|w| w[1].clone())
}

fn flag_or(args: &[String], name: &str, default: &str) -> String {
    flag(args, name).unwrap_or_else(|| default.to_owned())
}

fn require(args: &[String], name: &str) -> String {
    flag(args, name).unwrap_or_else(|| {
        eprintln!("pool_tool: missing required flag {name}");
        std::process::exit(1);
    })
}

// ── subcommands ──────────────────────────────────────────────────────────────

fn cmd_append(args: &[String]) {
    let pool_path = require(args, "--pool");
    let new_data_path = require(args, "--new-data");
    let pool_size: usize = require(args, "--pool-size")
        .parse()
        .expect("--pool-size must be an integer");

    // Append new data to the pool file.
    {
        let new_data = fs::read(&new_data_path).expect("failed to read new-data file");
        let mut pool = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&pool_path)
            .expect("failed to open pool file for append");
        pool.write_all(&new_data).expect("failed to append to pool");
        // Ensure the pool ends with a newline so the next append doesn't merge lines.
        if new_data.last() != Some(&b'\n') {
            pool.write_all(b"\n").expect("failed to write trailing newline");
        }
    }

    // Count lines; trim if over pool_size.
    let pool_bytes = fs::read(&pool_path).expect("failed to read pool");
    let total_lines = pool_bytes.iter().filter(|&&b| b == b'\n').count();

    if total_lines <= pool_size {
        println!("Pool size: {total_lines} positions");
        return;
    }

    // Keep the last pool_size lines: skip the first (total - pool_size) newlines.
    let skip = total_lines - pool_size;
    let mut newlines_seen = 0usize;
    let mut offset = 0usize;
    for (i, &b) in pool_bytes.iter().enumerate() {
        if b == b'\n' {
            newlines_seen += 1;
            if newlines_seen == skip {
                offset = i + 1;
                break;
            }
        }
    }
    fs::write(&pool_path, &pool_bytes[offset..]).expect("failed to write trimmed pool");
    println!("Pool size: {pool_size} positions");
}

fn cmd_split(args: &[String]) {
    let pool_path = require(args, "--pool");
    let train_path = require(args, "--train");
    let val_path = require(args, "--val");
    let val_fraction: f64 = flag_or(args, "--val-fraction", "0.1")
        .parse()
        .expect("--val-fraction must be a float");
    let min_val: usize = flag_or(args, "--min-val", "1000")
        .parse()
        .expect("--min-val must be an integer");

    // Read the entire file into one buffer — one allocation instead of 750k Strings.
    let buf = fs::read(&pool_path).expect("failed to read pool");

    // Collect (start, end) byte spans for each non-empty line.
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for (i, &b) in buf.iter().enumerate() {
        if b == b'\n' {
            if i > start {
                spans.push((start, i + 1)); // include newline
            }
            start = i + 1;
        }
    }
    if start < buf.len() {
        spans.push((start, buf.len())); // last line without trailing newline
    }

    let total = spans.len();

    // Shuffle the spans (cheap: just 16-byte pairs).
    let mut rng = rand::rngs::SmallRng::from_entropy();
    spans.shuffle(&mut rng);

    let n_val = ((total as f64 * val_fraction) as usize).max(min_val).min(total);
    let n_train = total - n_val;

    write_spans(&train_path, &buf, &spans[..n_train]);
    write_spans(&val_path,   &buf, &spans[n_train..]);

    println!("Split: {n_train} train / {n_val} val");
}

fn write_spans(path: &str, buf: &[u8], spans: &[(usize, usize)]) {
    let f = File::create(path).unwrap_or_else(|e| panic!("failed to create {path}: {e}"));
    let mut w = BufWriter::with_capacity(1 << 20, f); // 1 MB write buffer
    for &(s, e) in spans {
        w.write_all(&buf[s..e]).expect("write error");
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: pool_tool <append|split> [--flag value ...]");
        std::process::exit(1);
    }
    match args[1].as_str() {
        "append" => cmd_append(&args[2..]),
        "split"  => cmd_split(&args[2..]),
        cmd => {
            eprintln!("pool_tool: unknown subcommand '{cmd}'");
            std::process::exit(1);
        }
    }
}
