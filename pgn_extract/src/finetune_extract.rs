use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use pgn_reader::{BufferedReader, RawHeader, SanPlus, Skip, Visitor};
use rand::Rng;
use shakmaty::{Chess, Position, fen::Fen};

/// Extract FEN positions from a PGN file with piece-count filtering.
///
/// Designed for generating balanced finetune datasets. Run twice:
///
///   # 500k endgame positions (≤16 pieces, from late game)
///   finetune_extract --input games.pgn.zst --output endgame.fens \
///     --max-pieces 16 --sample-from-last 40 --max-positions 500000
///
///   # 500k middlegame positions (17+ pieces)
///   finetune_extract --input games.pgn.zst --output midgame.fens \
///     --min-pieces 17 --max-positions 500000
///
///   # Combine and shuffle
///   cat endgame.fens midgame.fens | shuf > finetune.fens
#[derive(Parser)]
#[command(about = "Extract FEN positions from PGN with piece-count filtering for finetune datasets")]
struct Args {
    /// Input PGN file (plain or .zst compressed)
    #[arg(short, long)]
    input: PathBuf,

    /// Output file (one FEN per line); defaults to stdout
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Maximum number of positions to extract
    #[arg(short = 'n', long, default_value_t = 1_000_000)]
    max_positions: usize,

    /// Minimum Elo for both players (0 = no filter)
    #[arg(long, default_value_t = 0)]
    min_elo: u32,

    /// Maximum Elo for both players (0 = no filter)
    #[arg(long, default_value_t = 0)]
    max_elo: u32,

    /// Minimum ply before sampling a position (skip opening moves)
    #[arg(long, default_value_t = 12)]
    min_ply: usize,

    /// Only consider the last N plies of each game for sampling.
    /// Useful for targeting endgame positions: set to 40-60.
    /// 0 = consider all plies after min_ply.
    #[arg(long, default_value_t = 0)]
    sample_from_last: usize,

    /// Only keep positions with at least this many pieces (0 = no filter)
    #[arg(long, default_value_t = 0)]
    min_pieces: u32,

    /// Only keep positions with at most this many pieces (32 = no filter)
    #[arg(long, default_value_t = 32)]
    max_pieces: u32,

    /// Skip the first N games in the PGN before collecting positions.
    /// Use to avoid overlap with a previously extracted dataset.
    #[arg(long, default_value_t = 0)]
    skip_games: usize,

    /// Positions to sample per game from the filtered candidate set
    #[arg(long, default_value_t = 1)]
    positions_per_game: usize,
}

struct Extractor {
    args: Args,

    // per-game state
    white_elo: u32,
    black_elo: u32,
    skip_game: bool,
    pos: Chess,
    ply: usize,
    // (fen, ply_index) — ply tracked so we can apply sample_from_last
    candidates: Vec<(String, usize)>,

    // output
    out: BufWriter<Box<dyn Write>>,
    found: usize,
    games_scanned: u64,

    rng: rand::rngs::ThreadRng,
}

impl Extractor {
    fn new(args: Args) -> Self {
        let writer: Box<dyn Write> = match &args.output {
            Some(path) => Box::new(File::create(path).expect("cannot create output file")),
            None => Box::new(std::io::stdout()),
        };
        Self {
            args,
            white_elo: 0,
            black_elo: 0,
            skip_game: false,
            pos: Chess::default(),
            ply: 0,
            candidates: Vec::new(),
            out: BufWriter::new(writer),
            found: 0,
            games_scanned: 0,
            rng: rand::thread_rng(),
        }
    }

    fn elo_ok(&self) -> bool {
        let min_ok = self.args.min_elo == 0
            || (self.white_elo >= self.args.min_elo && self.black_elo >= self.args.min_elo);
        let max_ok = self.args.max_elo == 0
            || (self.white_elo <= self.args.max_elo && self.black_elo <= self.args.max_elo);
        min_ok && max_ok
    }
}

impl Visitor for Extractor {
    type Result = bool; // true = keep going, false = stop

    fn begin_game(&mut self) {
        self.white_elo = 0;
        self.black_elo = 0;
        self.skip_game = false;
        self.pos = Chess::default();
        self.ply = 0;
        self.candidates.clear();
    }

    fn header(&mut self, key: &[u8], value: RawHeader<'_>) {
        match key {
            b"WhiteElo" => {
                self.white_elo = std::str::from_utf8(value.as_bytes())
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
            }
            b"BlackElo" => {
                self.black_elo = std::str::from_utf8(value.as_bytes())
                    .ok()
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
            }
            _ => {}
        }
    }

    fn end_headers(&mut self) -> Skip {
        self.skip_game = !self.elo_ok();
        Skip(self.skip_game || self.found >= self.args.max_positions)
    }

    fn san(&mut self, san_plus: SanPlus) {
        if self.skip_game {
            return;
        }
        let san = san_plus.san;
        if let Ok(mv) = san.to_move(&self.pos) {
            self.pos = self.pos.clone().play(&mv).unwrap();
            self.ply += 1;

            if self.ply < self.args.min_ply {
                return;
            }

            // Piece count filter — shakmaty gives us this directly
            let piece_count = self.pos.board().occupied().count() as u32;
            if piece_count < self.args.min_pieces || piece_count > self.args.max_pieces {
                return;
            }

            let fen = Fen::from_position(self.pos.clone(), shakmaty::EnPassantMode::Legal);
            self.candidates.push((fen.to_string(), self.ply));
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true)
    }

    fn end_game(&mut self) -> Self::Result {
        self.games_scanned += 1;

        if self.games_scanned <= self.args.skip_games as u64 {
            return self.found < self.args.max_positions;
        }

        if !self.skip_game && !self.candidates.is_empty() {
            // Apply sample_from_last: restrict to positions from the last N plies
            let pool: Vec<&str> = if self.args.sample_from_last > 0 {
                let last_ply = self.candidates.last().map(|(_, p)| *p).unwrap_or(0);
                let cutoff = last_ply.saturating_sub(self.args.sample_from_last);
                self.candidates
                    .iter()
                    .filter(|(_, p)| *p >= cutoff)
                    .map(|(fen, _)| fen.as_str())
                    .collect()
            } else {
                self.candidates.iter().map(|(fen, _)| fen.as_str()).collect()
            };

            if pool.is_empty() {
                return self.found < self.args.max_positions;
            }

            let n = self.args.positions_per_game.min(pool.len());

            // Reservoir-sample n positions from the pool
            let mut chosen: Vec<usize> = (0..pool.len().min(n)).collect();
            for i in n..pool.len() {
                let j = self.rng.gen_range(0..=i);
                if j < n {
                    chosen[j] = i;
                }
            }

            for idx in chosen {
                if self.found >= self.args.max_positions {
                    break;
                }
                writeln!(self.out, "{}", pool[idx]).unwrap();
                self.found += 1;
            }
        }

        if self.games_scanned % 100_000 == 0 {
            eprintln!(
                "  {} games scanned, {} / {} positions found",
                self.games_scanned, self.found, self.args.max_positions
            );
        }

        self.found < self.args.max_positions
    }
}

fn main() {
    let args = Args::parse();

    let min_pieces = args.min_pieces;
    let max_pieces = args.max_pieces;
    let sample_from_last = args.sample_from_last;
    let max_positions = args.max_positions;
    let input = args.input.clone();
    let output = args.output.clone();

    eprintln!(
        "finetune_extract: {} → {}",
        input.display(),
        output.as_ref().map(|p| p.display().to_string()).unwrap_or("stdout".into()),
    );
    eprintln!(
        "  pieces: {}–{}  |  sample_from_last: {}  |  target: {}",
        if min_pieces == 0 { "any".to_string() } else { min_pieces.to_string() },
        if max_pieces == 32 { "any".to_string() } else { max_pieces.to_string() },
        if sample_from_last == 0 { "all".to_string() } else { sample_from_last.to_string() },
        max_positions,
    );

    let file = File::open(&input).expect("cannot open input PGN");
    let mut reader = BufferedReader::new(file);
    let mut extractor = Extractor::new(args);

    loop {
        match reader.read_game(&mut extractor) {
            Ok(Some(keep_going)) => {
                if !keep_going {
                    break;
                }
            }
            Ok(None) => break,
            Err(e) => {
                eprintln!("parse error: {e}");
            }
        }
    }

    extractor.out.flush().unwrap();
    eprintln!(
        "Done: {} positions from {} games",
        extractor.found, extractor.games_scanned,
    );
}
