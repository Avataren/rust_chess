use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use pgn_reader::{BufferedReader, RawHeader, SanPlus, Skip, Visitor};
use rand::Rng;
use shakmaty::{Chess, Position, fen::Fen};

#[derive(Parser)]
#[command(about = "Extract random FEN positions from a PGN file")]
struct Args {
    /// Input PGN file
    #[arg(short, long)]
    input: PathBuf,

    /// Output file (one FEN per line); defaults to stdout
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Maximum number of positions to extract
    #[arg(short, long, default_value_t = 1_000_000)]
    max_positions: usize,

    /// Minimum Elo for both players (0 = no filter)
    #[arg(long, default_value_t = 0)]
    min_elo: u32,

    /// Maximum Elo for both players (0 = no filter)
    #[arg(long, default_value_t = 0)]
    max_elo: u32,

    /// Minimum ply before sampling a position
    #[arg(long, default_value_t = 12)]
    min_ply: usize,

    /// Sample at most this many positions per game (chosen uniformly at random)
    #[arg(long, default_value_t = 1)]
    positions_per_game: usize,
}

struct Extractor {
    min_elo: u32,
    max_elo: u32,
    min_ply: usize,
    positions_per_game: usize,
    max_positions: usize,

    // per-game state
    white_elo: u32,
    black_elo: u32,
    skip_game: bool,
    pos: Chess,
    ply: usize,
    candidate_fens: Vec<String>,

    // output
    out: BufWriter<Box<dyn Write>>,
    found: usize,

    rng: rand::rngs::ThreadRng,
}

impl Extractor {
    fn new(args: &Args) -> Self {
        let writer: Box<dyn Write> = match &args.output {
            Some(path) => Box::new(File::create(path).expect("cannot create output file")),
            None => Box::new(std::io::stdout()),
        };
        Self {
            min_elo: args.min_elo,
            max_elo: args.max_elo,
            min_ply: args.min_ply,
            positions_per_game: args.positions_per_game,
            max_positions: args.max_positions,
            white_elo: 0,
            black_elo: 0,
            skip_game: false,
            pos: Chess::default(),
            ply: 0,
            candidate_fens: Vec::new(),
            out: BufWriter::new(writer),
            found: 0,
            rng: rand::thread_rng(),
        }
    }

    fn elo_ok(&self) -> bool {
        let min_ok = self.min_elo == 0
            || (self.white_elo >= self.min_elo && self.black_elo >= self.min_elo);
        let max_ok = self.max_elo == 0
            || (self.white_elo <= self.max_elo && self.black_elo <= self.max_elo);
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
        self.candidate_fens.clear();
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
        Skip(self.skip_game || self.found >= self.max_positions)
    }

    fn san(&mut self, san_plus: SanPlus) {
        if self.skip_game {
            return;
        }
        let san = san_plus.san;
        if let Ok(mv) = san.to_move(&self.pos) {
            self.pos = self.pos.clone().play(&mv).unwrap();
            self.ply += 1;
            if self.ply >= self.min_ply {
                let fen = Fen::from_position(self.pos.clone(), shakmaty::EnPassantMode::Legal);
                self.candidate_fens.push(fen.to_string());
            }
        }
    }

    fn begin_variation(&mut self) -> Skip {
        Skip(true) // skip variations
    }

    fn end_game(&mut self) -> Self::Result {
        if !self.skip_game && !self.candidate_fens.is_empty() {
            let n = self.positions_per_game.min(self.candidate_fens.len());
            // reservoir-sample n positions
            let mut chosen: Vec<usize> = (0..self.candidate_fens.len().min(n)).collect();
            for i in n..self.candidate_fens.len() {
                let j = self.rng.gen_range(0..=i);
                if j < n {
                    chosen[j] = i;
                }
            }
            for idx in chosen {
                if self.found >= self.max_positions {
                    break;
                }
                writeln!(self.out, "{}", self.candidate_fens[idx]).unwrap();
                self.found += 1;
            }
        }
        self.found < self.max_positions
    }
}

fn main() {
    let args = Args::parse();

    let file = File::open(&args.input).expect("cannot open input PGN");
    let mut reader = BufferedReader::new(file);
    let mut extractor = Extractor::new(&args);

    eprintln!(
        "Scanning {} for up to {} positions (elo {}-{})...",
        args.input.display(),
        args.max_positions,
        if args.min_elo == 0 { "any".to_string() } else { args.min_elo.to_string() },
        if args.max_elo == 0 { "any".to_string() } else { args.max_elo.to_string() },
    );

    let mut games = 0u64;
    loop {
        match reader.read_game(&mut extractor) {
            Ok(Some(keep_going)) => {
                games += 1;
                if games % 100_000 == 0 {
                    eprintln!("  {} games scanned, {} positions found", games, extractor.found);
                }
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
        "Done: {} positions from {} games → {}",
        extractor.found,
        games,
        args.output.as_ref().map(|p| p.display().to_string()).unwrap_or("stdout".into()),
    );
}
