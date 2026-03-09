use bevy::prelude::*;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy_async_task::TaskRunner;
use bevy_tweening::{lens::TransformPositionLens, *};
use std::task::Poll;
use chess_board::ChessBoard;
use chess_evaluation::alpha_beta;
use chess_foundation::ChessMove;
use move_generator::{
    move_generator::get_all_legal_moves_for_color, piece_conductor::PieceConductor,
};
use std::time::Duration;

use crate::{
    board::BoardDimensions,
    game_events::{ChessAction, ChessEvent, RefreshPiecesFromBoardEvent},
    pieces::ChessPieceComponent,
    ChessBoardRes, PieceConductorRes,
};

// Function to get local position from board coordinates (col, row)
fn get_local_position_from_board_coords(
    col: u16,
    row: u16,
    board_dimensions: &BoardDimensions,
) -> Vec3 {
    Vec3::new(
        col as f32 * board_dimensions.square_size - board_dimensions.board_size.x / 2.0
            + board_dimensions.square_size / 2.0,
        row as f32 * board_dimensions.square_size - board_dimensions.board_size.y / 2.0
            + board_dimensions.square_size / 2.0,
        1.5,
    )
}

pub fn on_tween_completed(
    mut tween_completed_events: MessageReader<AnimCompletedEvent>,
    mut refresh_pieces_events: MessageWriter<RefreshPiecesFromBoardEvent>,
) {
    for _ in tween_completed_events.read() {
        println!("Tween completed");
        refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
        break;
    }
}

async fn alpha_beta_task(
    mut chess_board: &mut ChessBoard, // Note: Pass ownership, avoid &mut
    conductor: &PieceConductor, // Avoid &mut, ensure these types are Clone or otherwise cheap to move
    depth: i32,
    alpha: i32,
    beta: i32,
    is_white: bool,
) -> (i32, Option<ChessMove>) {
    alpha_beta(&mut chess_board, &conductor, depth, alpha, beta, is_white)
}

pub fn handle_async_moves(
    mut task_executor: TaskRunner<(i32, Option<ChessMove>)>,
    mut commands: Commands,
    mut chess_board: ResMut<ChessBoardRes>,
    board_dimensions: Res<BoardDimensions>,
    move_generator: ResMut<PieceConductorRes>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    mut chess_ew: MessageReader<ChessEvent>,
) {
    if task_executor.is_idle() {
        // Task is idle — check for new chess events to start a task
        for event in chess_ew.read() {
            if let ChessAction::MakeMove = event.action {
                let mut chess_board_clone = chess_board.chess_board.clone();
                let move_generator_clone = move_generator.magic.clone();
                task_executor.start(async move {
                    alpha_beta_task(
                        &mut chess_board_clone,
                        &move_generator_clone,
                        5,        // depth
                        i32::MIN, // alpha
                        i32::MAX, // beta
                        false,    // is_white
                    )
                    .await
                });
                println!("Alpha-beta task started!");
                break;
            }
        }
        return;
    }

    // Drain events while task is running to prevent buildup
    for _ in chess_ew.read() {}

    match task_executor.poll() {
        Poll::Pending => {
            // println!("Alpha-beta computation in progress...");
        }
        Poll::Ready((score, mut best_move)) => {
            println!("Received score {score}");
            if best_move.is_none() {
                println!("No move found");
                let all_moves = get_all_legal_moves_for_color(
                    &mut chess_board.chess_board,
                    &move_generator.magic,
                    false,
                );
                if all_moves.is_empty() {
                    println!("Checkmate");
                    return;
                } else {
                    best_move = Some(all_moves[0]);
                    println!("Random move");
                }
            }
            let mut engine_move = best_move.unwrap();

            // if let Some(engine_move) = moves.choose(&mut rand::thread_rng()) {
            if chess_board.chess_board.make_move(&mut engine_move) {
                // Inside the 'if let Some(engine_move) = moves.choose(&mut rand::thread_rng())' block
                let start_local_position = get_local_position_from_board_coords(
                    engine_move.start_square() % 8,
                    engine_move.start_square() / 8, // Adjusted calculation
                    &board_dimensions,
                );
                let end_local_position = get_local_position_from_board_coords(
                    engine_move.target_square() % 8,
                    engine_move.target_square() / 8, // Adjusted calculation
                    &board_dimensions,
                );

                if let Some((entity, _, _)) = piece_query.iter_mut().find(|(_, _, chess_piece)| {
                    chess_piece.col == (engine_move.start_square() % 8) as usize
                        && chess_piece.row == 7 - (engine_move.start_square() / 8) as usize
                }) {
                    let tween = Tween::new(
                        EaseFunction::CubicOut,
                        Duration::from_millis(250),
                        TransformPositionLens {
                            start: start_local_position,
                            end: end_local_position,
                        },
                    );

                    commands.entity(entity).insert(TweenAnim::new(tween));
                } else {
                    println!("No piece found at start square");
                }
            }
        }
    }
}

pub fn handle_chess_events(
    mut chess_ew: MessageReader<ChessEvent>,
    mut chess_board: ResMut<ChessBoardRes>,
    mut refresh_pieces_events: MessageWriter<RefreshPiecesFromBoardEvent>,
) {
    for event in chess_ew.read() {
        match event.action {
            ChessAction::Undo => {
                println!("Undoing move");
                chess_board.chess_board.undo_move();
                refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
            }
            //refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
            _ => {}
        }
    }
}
