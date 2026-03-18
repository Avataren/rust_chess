//! Fast JSONL → binary numpy encoder for NNUE training.
//!
//! Reads {fen, cp} JSONL and writes numpy-compatible .npy files:
//!
//! Dual mode (--dual):
//!   {out}.white_indices.npy  (N, 32) uint16
//!   {out}.black_indices.npy  (N, 32) uint16
//!   {out}.counts.npy         (N,)    uint8
//!   {out}.cp.npy             (N,)    float32   (white-absolute)
//!   {out}.piece_count.npy    (N,)    uint8
//!
//! Single mode (default):
//!   {out}.indices.npy        (N, 32) uint16
//!   {out}.counts.npy         (N,)    uint8
//!   {out}.cp.npy             (N,)    float32   (side-to-move)
//!   {out}.piece_count.npy    (N,)    uint8
//!
//! Uses streaming BufWriter — no mmap, no large RAM allocation.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use chess_board::{ChessBoard, FENParser};

// ── Architecture constants (must match chess_evaluation::neural_eval) ─────

const HALFKP_DIM: usize = 12 * 64 * 16; // 12,288
const SENTINEL: u16 = HALFKP_DIM as u16;
const MAX_ACTIVE: usize = 32;
const WRITE_BUF: usize = 8 * 1024 * 1024; // 8 MB per output file
const PROGRESS_EVERY: usize = 1_000_000;

// ── King bucket table ─────────────────────────────────────────────────────

const KING_BUCKET: [usize; 64] = {
    let mut t = [0usize; 64];
    let mut sq = 0usize;
    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;
        let fb = if file <= 3 { file } else { 7 - file };
        let rh = if rank <= 3 { 0 } else { 1 };
        t[sq] = rh * 4 + fb;
        sq += 1;
    }
    t
};

// ── NPY v1.0 header writer ────────────────────────────────────────────────

fn write_npy_header(w: &mut impl Write, dtype: &str, shape: &[usize]) -> io::Result<()> {
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
        )
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}\n",
        dtype, shape_str
    );
    // Total = 10-byte prefix + HEADER_LEN must be a multiple of 64.
    let prefix = 10usize;
    let total = prefix + dict.len();
    let padded = (total + 63) / 64 * 64;
    let padding = padded - total;
    let header_len = (dict.len() + padding) as u16;

    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1u8, 0u8])?;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&dict.as_bytes()[..dict.len() - 1])?; // without trailing \n
    for _ in 0..padding {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;
    Ok(())
}

// ── HalfKP encoding ───────────────────────────────────────────────────────

fn encode_dual(board: &ChessBoard) -> ([u16; MAX_ACTIVE], [u16; MAX_ACTIVE], u8) {
    let white_bb = board.get_white();
    let black_bb = board.get_black();

    let wk_sq = (white_bb & board.get_kings()).0.trailing_zeros() as usize;
    let wk_sq = wk_sq.min(63);
    let wk_bucket = KING_BUCKET[wk_sq];
    let mirror_w = (wk_sq % 8) >= 4;

    let bk_sq_raw = (black_bb & board.get_kings()).0.trailing_zeros() as usize;
    let bk_sq_raw = bk_sq_raw.min(63);
    let bk_flipped = bk_sq_raw ^ 56;
    let bk_bucket = KING_BUCKET[bk_flipped];
    let mirror_b = (bk_flipped % 8) >= 4;

    let mut w_idx = [SENTINEL; MAX_ACTIVE];
    let mut b_idx = [SENTINEL; MAX_ACTIVE];
    let mut wc = 0usize;
    let mut bc = 0usize;

    macro_rules! push_w {
        ($bb:expr, $slot:expr) => {{
            let mut bb = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                if wc < MAX_ACTIVE {
                    let m = if mirror_w { sq ^ 7 } else { sq };
                    w_idx[wc] = ($slot * 64 * 16 + m * 16 + wk_bucket) as u16;
                    wc += 1;
                }
            }
        }};
    }

    macro_rules! push_b {
        ($bb:expr, $slot:expr) => {{
            let mut bb = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                if bc < MAX_ACTIVE {
                    let r = sq ^ 56;
                    let m = if mirror_b { r ^ 7 } else { r };
                    b_idx[bc] = ($slot * 64 * 16 + m * 16 + bk_bucket) as u16;
                    bc += 1;
                }
            }
        }};
    }

    push_w!(white_bb & board.get_pawns(),    0);
    push_w!(white_bb & board.get_knights(),  1);
    push_w!(white_bb & board.get_bishops(),  2);
    push_w!(white_bb & board.get_rooks(),    3);
    push_w!(white_bb & board.get_queens(),   4);
    push_w!(white_bb & board.get_kings(),    5);
    push_w!(black_bb & board.get_pawns(),    6);
    push_w!(black_bb & board.get_knights(),  7);
    push_w!(black_bb & board.get_bishops(),  8);
    push_w!(black_bb & board.get_rooks(),    9);
    push_w!(black_bb & board.get_queens(),  10);
    push_w!(black_bb & board.get_kings(),   11);

    push_b!(black_bb & board.get_pawns(),    0);
    push_b!(black_bb & board.get_knights(),  1);
    push_b!(black_bb & board.get_bishops(),  2);
    push_b!(black_bb & board.get_rooks(),    3);
    push_b!(black_bb & board.get_queens(),   4);
    push_b!(black_bb & board.get_kings(),    5);
    push_b!(white_bb & board.get_pawns(),    6);
    push_b!(white_bb & board.get_knights(),  7);
    push_b!(white_bb & board.get_bishops(),  8);
    push_b!(white_bb & board.get_rooks(),    9);
    push_b!(white_bb & board.get_queens(),  10);
    push_b!(white_bb & board.get_kings(),   11);

    (w_idx, b_idx, wc as u8)
}

fn encode_single(board: &ChessBoard) -> ([u16; MAX_ACTIVE], u8) {
    let white_to_move = board.is_white_active();
    let flip = !white_to_move;
    let (ours, theirs) = if white_to_move {
        (board.get_white(), board.get_black())
    } else {
        (board.get_black(), board.get_white())
    };

    let king_raw = (ours & board.get_kings()).0.trailing_zeros() as usize;
    let king_sq = if flip { king_raw ^ 56 } else { king_raw }.min(63);
    let bucket = KING_BUCKET[king_sq];

    let mut indices = [SENTINEL; MAX_ACTIVE];
    let mut count = 0usize;

    macro_rules! push {
        ($bb:expr, $slot:expr) => {{
            let mut bb = $bb;
            while bb.0 != 0 {
                let sq = bb.0.trailing_zeros() as usize;
                bb.0 &= bb.0 - 1;
                if count < MAX_ACTIVE {
                    let m = if flip { sq ^ 56 } else { sq };
                    indices[count] = ($slot * 64 * 16 + m * 16 + bucket) as u16;
                    count += 1;
                }
            }
        }};
    }

    push!(ours   & board.get_pawns(),    0);
    push!(ours   & board.get_knights(),  1);
    push!(ours   & board.get_bishops(),  2);
    push!(ours   & board.get_rooks(),    3);
    push!(ours   & board.get_queens(),   4);
    push!(ours   & board.get_kings(),    5);
    push!(theirs & board.get_pawns(),    6);
    push!(theirs & board.get_knights(),  7);
    push!(theirs & board.get_bishops(),  8);
    push!(theirs & board.get_rooks(),    9);
    push!(theirs & board.get_queens(),  10);
    push!(theirs & board.get_kings(),   11);

    (indices, count as u8)
}

// ── Line counter ──────────────────────────────────────────────────────────

fn count_lines(path: &Path) -> usize {
    let f = File::open(path).expect("cannot open input for counting");
    BufReader::new(f).lines().count()
}

// ── Output path helper ────────────────────────────────────────────────────

fn out_path(prefix: &Path, ext: &str) -> PathBuf {
    let mut s = prefix.to_string_lossy().into_owned();
    s.push_str(ext);
    PathBuf::from(s)
}

fn open_out(prefix: &Path, ext: &str) -> BufWriter<File> {
    let p = out_path(prefix, ext);
    BufWriter::with_capacity(
        WRITE_BUF,
        File::create(&p).unwrap_or_else(|e| panic!("cannot create {}: {e}", p.display())),
    )
}

// ── Encode loop ───────────────────────────────────────────────────────────

fn run(input: &Path, out_prefix: &Path, dual: bool, max_cp: f64) {
    eprint!("Counting lines in {} … ", input.display());
    let n = count_lines(input);
    eprintln!("{n}");

    let mut board = ChessBoard::new();
    let reader = BufReader::with_capacity(WRITE_BUF, File::open(input).expect("cannot open input"));

    if dual {
        let mut w_f  = open_out(out_prefix, ".white_indices.npy");
        let mut b_f  = open_out(out_prefix, ".black_indices.npy");
        let mut c_f  = open_out(out_prefix, ".counts.npy");
        let mut cp_f = open_out(out_prefix, ".cp.npy");
        let mut pc_f = open_out(out_prefix, ".piece_count.npy");

        write_npy_header(&mut w_f,  "<u2", &[n, MAX_ACTIVE]).unwrap();
        write_npy_header(&mut b_f,  "<u2", &[n, MAX_ACTIVE]).unwrap();
        write_npy_header(&mut c_f,  "|u1", &[n]).unwrap();
        write_npy_header(&mut cp_f, "<f4", &[n]).unwrap();
        write_npy_header(&mut pc_f, "|u1", &[n]).unwrap();

        let t0 = Instant::now();
        let mut i = 0usize;
        for line in reader.lines() {
            let line = line.expect("read error");
            if line.trim().is_empty() { continue; }

            let v: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("JSON error on line {i}: {e}\n  {line}"));
            let fen = v["fen"].as_str().expect("missing fen");
            let mut cp = v["cp"].as_f64().expect("missing cp");
            cp = cp.clamp(-max_cp, max_cp);

            board.clear();
            FENParser::set_board_from_fen(&mut board, fen);

            if !board.is_white_active() { cp = -cp; }

            let (w_idx, b_idx, count) = encode_dual(&board);
            let piece_count = board.get_all_pieces().count_ones() as u8;

            for v in &w_idx  { w_f.write_all(&v.to_le_bytes()).unwrap(); }
            for v in &b_idx  { b_f.write_all(&v.to_le_bytes()).unwrap(); }
            c_f.write_all(&[count]).unwrap();
            cp_f.write_all(&(cp as f32).to_le_bytes()).unwrap();
            pc_f.write_all(&[piece_count]).unwrap();

            i += 1;
            if i % PROGRESS_EVERY == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = i as f64 / elapsed;
                eprintln!("  {i}/{n}  ({:.0}k pos/s)", rate / 1000.0);
            }
        }
        // Ensure all buffers flushed
        for f in [&mut w_f, &mut b_f, &mut c_f, &mut cp_f, &mut pc_f] {
            f.flush().unwrap();
        }
        eprintln!("Wrote dual files to {}.*", out_prefix.display());
    } else {
        let mut idx_f = open_out(out_prefix, ".indices.npy");
        let mut c_f   = open_out(out_prefix, ".counts.npy");
        let mut cp_f  = open_out(out_prefix, ".cp.npy");
        let mut pc_f  = open_out(out_prefix, ".piece_count.npy");

        write_npy_header(&mut idx_f, "<u2", &[n, MAX_ACTIVE]).unwrap();
        write_npy_header(&mut c_f,   "|u1", &[n]).unwrap();
        write_npy_header(&mut cp_f,  "<f4", &[n]).unwrap();
        write_npy_header(&mut pc_f,  "|u1", &[n]).unwrap();

        let t0 = Instant::now();
        let mut i = 0usize;
        for line in reader.lines() {
            let line = line.expect("read error");
            if line.trim().is_empty() { continue; }

            let v: serde_json::Value = serde_json::from_str(&line)
                .unwrap_or_else(|e| panic!("JSON error on line {i}: {e}\n  {line}"));
            let fen = v["fen"].as_str().expect("missing fen");
            let mut cp = v["cp"].as_f64().expect("missing cp");
            cp = cp.clamp(-max_cp, max_cp);

            board.clear();
            FENParser::set_board_from_fen(&mut board, fen);

            let (indices, count) = encode_single(&board);
            let piece_count = board.get_all_pieces().count_ones() as u8;

            for v in &indices { idx_f.write_all(&v.to_le_bytes()).unwrap(); }
            c_f.write_all(&[count]).unwrap();
            cp_f.write_all(&(cp as f32).to_le_bytes()).unwrap();
            pc_f.write_all(&[piece_count]).unwrap();

            i += 1;
            if i % PROGRESS_EVERY == 0 {
                let elapsed = t0.elapsed().as_secs_f64();
                let rate = i as f64 / elapsed;
                eprintln!("  {i}/{n}  ({:.0}k pos/s)", rate / 1000.0);
            }
        }
        for f in [&mut idx_f, &mut c_f, &mut cp_f, &mut pc_f] {
            f.flush().unwrap();
        }
        eprintln!("Wrote single files to {}.*", out_prefix.display());
    }
}

// ── CLI ───────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut input: Option<PathBuf>  = None;
    let mut output: Option<PathBuf> = None;
    let mut dual   = false;
    let mut max_cp = 1500.0f64;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input"      => { i += 1; input  = Some(PathBuf::from(&args[i])); }
            "--output"     => { i += 1; output = Some(PathBuf::from(&args[i])); }
            "--dual"       => { dual = true; }
            "--max-cp-abs" => { i += 1; max_cp = args[i].parse().expect("--max-cp-abs must be a number"); }
            "--help" | "-h" => {
                println!("Usage: nnue_preprocess --input <file.jsonl> --output <prefix> [--dual] [--max-cp-abs 1500]");
                std::process::exit(0);
            }
            other => { eprintln!("Unknown argument: {other}"); std::process::exit(1); }
        }
        i += 1;
    }

    let input  = input.expect("--input <file.jsonl> is required");
    let output = output.expect("--output <prefix> is required");

    let t0 = Instant::now();
    run(&input, &output, dual, max_cp);
    eprintln!("Total: {:.1}s", t0.elapsed().as_secs_f64());
}
