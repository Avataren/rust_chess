use bevy::{
    input::{mouse::MouseButton, touch::TouchPhase},
    prelude::*,
};
use bevy::ecs::message::{MessageReader, MessageWriter};
use move_generator::move_generator::{get_all_legal_moves_for_color, get_legal_move_list_from_square};

use crate::{
    board::{BoardDimensions, ChessBoardTransform},
    board_accessories::{DebugSquare, EnableDebugMarkers},
    game_events::{
        ChessAction, ChessEvent, DragPieceEvent, DropPieceEvent, PickUpPieceEvent,
        RefreshPiecesFromBoardEvent,
    },
    game_resources::{GameOverState, GamePhase, PlayerColor, ValidMoves},
    pieces::{get_board_coords_from_cursor, ChessPieceComponent},
    sound::{spawn_sound, SoundEffects},
    ChessBoardRes, PieceConductorRes,
};

fn is_player_piece(piece_char: char, player_color: PlayerColor) -> bool {
    match player_color {
        PlayerColor::White => piece_char.is_uppercase(),
        PlayerColor::Black => piece_char.is_lowercase(),
    }
}
use chess_foundation::{board_helper::board_row_col_to_square_index, Bitboard};

#[derive(Resource)]
pub struct PieceIsPickedUp {
    pub piece_type: Option<char>,
    pub piece_entity: Option<Entity>,
    pub original_row_col: (usize, usize),
    pub current_position: Vec3,
    pub is_dragging: bool,
    pub is_tap_selected: bool,
}

impl Default for PieceIsPickedUp {
    fn default() -> Self {
        PieceIsPickedUp {
            piece_type: None,
            piece_entity: None,
            original_row_col: (0, 0),
            current_position: Vec3::new(0.0, 0.0, 0.0),
            is_dragging: false,
            is_tap_selected: false,
        }
    }
}

pub fn handle_touch_input(
    mut chess_pickup_ew: MessageWriter<PickUpPieceEvent>,
    mut chess_drag_ew: MessageWriter<DragPieceEvent>,
    mut chess_drop_ew: MessageWriter<DropPieceEvent>,
    piece_is_picked_up: Res<PieceIsPickedUp>,
    mut touch_er: MessageReader<TouchInput>,
) {
    for ev in touch_er.read() {
        match ev.phase {
            TouchPhase::Started => {
                if piece_is_picked_up.is_tap_selected && !piece_is_picked_up.is_dragging {
                    // Piece already selected via tap — treat this touch as the destination
                    chess_drop_ew.write(DropPieceEvent {
                        position: ev.position,
                    });
                } else {
                    // Nothing selected yet — pick up whatever is at this position
                    chess_pickup_ew.write(PickUpPieceEvent {
                        position: ev.position,
                    });
                }
            }
            TouchPhase::Moved => {
                chess_drag_ew.write(DragPieceEvent {
                    position: ev.position,
                });
            }
            TouchPhase::Ended | TouchPhase::Canceled => {
                if piece_is_picked_up.is_dragging {
                    chess_drop_ew.write(DropPieceEvent {
                        position: ev.position,
                    });
                }
                // If tap-selected (not dragging), keep the selection for the next tap
            }
        }
    }
}

pub fn handle_mouse_input(
    q_windows: Query<&Window>,
    mouse_button_input: ResMut<'_, ButtonInput<MouseButton>>,
    mut chess_pickup_ew: MessageWriter<PickUpPieceEvent>,
    mut chess_drag_ew: MessageWriter<DragPieceEvent>,
    mut chess_drop_ew: MessageWriter<DropPieceEvent>,
) {
    if let Some(window) = q_windows.iter().next() {
        if let Some(cursor_position) = window.cursor_position() {
            if mouse_button_input.just_pressed(MouseButton::Left) {
                chess_pickup_ew.write(PickUpPieceEvent {
                    position: cursor_position,
                });
            } else if mouse_button_input.pressed(MouseButton::Left) {
                chess_drag_ew.write(DragPieceEvent {
                    position: cursor_position,
                });
            }

            if mouse_button_input.just_released(MouseButton::Left) {
                chess_drop_ew.write(DropPieceEvent {
                    position: cursor_position,
                });
            }
        }
    }
}

fn board_coords_to_chess_coords(board_coords: Vec2, square_size: f32) -> Option<(usize, usize)> {
    if board_coords.x < 0.0 || board_coords.y < 0.0 {
        return None; // Early return for negative coordinates
    }

    let col_f = board_coords.x / square_size;
    let row_f = board_coords.y / square_size;

    if col_f >= 8.0 || row_f >= 8.0 {
        return None; // Coordinates outside the chess board
    }

    let col = col_f.floor() as usize;
    let row = (7.0 - row_f.floor()) as usize;

    Some((col, row))
}

pub fn pick_up_piece(
    q_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    board_transform: Res<ChessBoardTransform>,
    board_dimensions: Res<BoardDimensions>,
    mut chess_board: ResMut<ChessBoardRes>,
    mut piece_is_picked_up: ResMut<PieceIsPickedUp>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    magic_res: Res<PieceConductorRes>,
    mut commands: Commands,
    mut chess_input_er: MessageReader<PickUpPieceEvent>,
    mut valid_moves_res: ResMut<ValidMoves>,
    game_over_state: Res<GameOverState>,
    game_phase: Res<GamePhase>,
    player_color: Res<PlayerColor>,
) {
    let mut position = Option::None;
    for inp in chess_input_er.read() {
        position = Some(inp.position);
    }
    if position.is_none() {
        return;
    }
    if *game_phase != GamePhase::Playing || *game_over_state != GameOverState::Playing {
        return;
    }

    if let Ok((camera, camera_transform)) = q_camera.single() {
        let Some(board_coords) = get_board_coords_from_cursor(
            position.unwrap(),
            camera,
            camera_transform,
            &board_transform,
            &board_dimensions,
        ) else {
            return;
        };

        if let Some((col, row)) =
            board_coords_to_chess_coords(board_coords, board_dimensions.square_size)
        {
            // spawn debug pieces
            if let Some((entity, transform, chess_piece)) = piece_query
                .iter_mut()
                .find(|(_, _, chess_piece)| {
                    chess_piece.row == row
                        && chess_piece.col == col
                        && is_player_piece(chess_piece.piece_type, *player_color)
                })
            {
                piece_is_picked_up.piece_type = Some(chess_piece.piece_type);
                piece_is_picked_up.original_row_col = (row, col);
                piece_is_picked_up.current_position = transform.translation;
                piece_is_picked_up.piece_entity = Some(entity);
                piece_is_picked_up.is_dragging = false;
                piece_is_picked_up.is_tap_selected = true;

                println!("**************** PICKUP ****************");

                let valid_moves = get_legal_move_list_from_square(
                    board_row_col_to_square_index(row, col),
                    &mut chess_board.chess_board,
                    &magic_res.magic,
                );

                commands.spawn(EnableDebugMarkers::new(valid_moves.clone()));

                valid_moves_res.set_moves(valid_moves);

                println!("Picked up piece: {:?}", chess_piece.piece_type);
            } else {
                *piece_is_picked_up = PieceIsPickedUp::default();
            }
        }
    }
}

pub fn drag_piece(
    q_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    board_transform: Res<ChessBoardTransform>,
    board_dimensions: Res<BoardDimensions>,
    mut piece_is_picked_up: ResMut<PieceIsPickedUp>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    mut chess_input_er: MessageReader<DragPieceEvent>,
) {
    let mut position = Option::None;
    for inp in chess_input_er.read() {
        position = Some(inp.position);
    }
    if position.is_none() {
        return;
    }

    if piece_is_picked_up.piece_entity.is_none() {
        return;
    }

    if let Ok((camera, camera_transform)) = q_camera.single() {
        if let Some(piece_entity) = piece_is_picked_up.piece_entity {
            if let Ok((_, mut transform, _)) = piece_query.get_mut(piece_entity) {
                let Some(board_coords) = get_board_coords_from_cursor(
                    position.unwrap(),
                    camera,
                    camera_transform,
                    &board_transform,
                    &board_dimensions,
                ) else {
                    return;
                };

                piece_is_picked_up.is_dragging = true;
                transform.translation = Vec3::new(
                    board_coords.x - board_dimensions.board_size.x / 2.0,
                    board_coords.y - board_dimensions.board_size.y / 2.0,
                    1.0,
                );
            }
        }
    }
}

pub fn drop_piece(
    q_camera: Query<(&Camera, &GlobalTransform), With<Camera2d>>,
    board_transform: Res<ChessBoardTransform>,
    board_dimensions: Res<BoardDimensions>,
    mut piece_is_picked_up: ResMut<PieceIsPickedUp>,
    mut piece_query: Query<(Entity, &mut Transform, &mut ChessPieceComponent)>,
    mut chess_input_er: MessageReader<DropPieceEvent>,
    mut commands: Commands,
    sound_effects: Res<SoundEffects>,
    mut chess_board: ResMut<ChessBoardRes>,
    mut refresh_pieces_events: MessageWriter<RefreshPiecesFromBoardEvent>,
    mut valid_moves_res: ResMut<ValidMoves>,
    mut game_event_ew: MessageWriter<ChessEvent>,
    mut game_over_state: ResMut<GameOverState>,
    game_phase: Res<GamePhase>,
    player_color: Res<PlayerColor>,
    move_generator_res: Res<PieceConductorRes>,
) {
    let mut position = Option::None;
    for inp in chess_input_er.read() {
        position = Some(inp.position);
    }

    if position.is_none() {
        return;
    }
    if *game_phase != GamePhase::Playing || *game_over_state != GameOverState::Playing {
        *piece_is_picked_up = PieceIsPickedUp::default();
        return;
    }

    let has_piece = piece_is_picked_up.piece_entity.is_some();
    let can_drop = piece_is_picked_up.is_dragging || piece_is_picked_up.is_tap_selected;
    if !has_piece || !can_drop {
        *piece_is_picked_up = PieceIsPickedUp::default();
        refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
        return;
    }

    if let Ok((camera, camera_transform)) = q_camera.single() {

        if let Some(piece_entity) = piece_is_picked_up.piece_entity {
            if let Ok((_, _transform, mut chess_piece)) = piece_query.get_mut(piece_entity) {
                let Some(board_coords) = get_board_coords_from_cursor(
                    position.unwrap(),
                    camera,
                    camera_transform,
                    &board_transform,
                    &board_dimensions,
                ) else {
                    *piece_is_picked_up = PieceIsPickedUp::default();
                    refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
                    return;
                };

                if let Some((col, row)) =
                    board_coords_to_chess_coords(board_coords, board_dimensions.square_size)
                {
                    //println!("board_coords: {:?}", board_coords);

                    let square = board_row_col_to_square_index(row, col);
                    let valid_move = valid_moves_res
                        .moves
                        .iter()
                        .find(|chess_move| chess_move.target_square() == square);

                    if valid_move.is_none()
                        || (board_coords.x < 0.0)
                        || (board_coords.x > board_dimensions.board_size.x)
                        || (board_coords.y < 0.0)
                        || (board_coords.y > board_dimensions.board_size.y)
                        || (piece_is_picked_up.original_row_col.0 == row
                            && piece_is_picked_up.original_row_col.1 == col)
                    {
                        //trying to drop piece outside board, or same position as picked up
                        println!("invalid move!");
                        *piece_is_picked_up = PieceIsPickedUp::default();
                        refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
                        return;
                    }

                    chess_piece.row = row;
                    chess_piece.col = col;

                    let is_capture = !(chess_board.chess_board.get_all_pieces()
                        & Bitboard::from_square_index(valid_move.unwrap().target_square()))
                    .is_empty();
                    // println!("************************************");
                    // println!("Making move from {} to {}", valid_move.unwrap().start_square(), valid_move.unwrap().target_square());
                    // println!("************************************");

                    if chess_board
                        .chess_board
                        .make_move(&mut valid_move.unwrap().clone())
                    {
                        valid_moves_res.moves.clear();
                        if is_capture {
                            spawn_sound(&mut commands, &sound_effects, "capture.ogg");
                        } else {
                            spawn_sound(&mut commands, &sound_effects, "move-self.ogg");
                        }
                        println!("Released piece at row: {}, col: {}", row, col);

                        if chess_board.chess_board.is_repetition(3) {
                            println!("Draw by repetition!");
                            *game_over_state = GameOverState::Draw;
                            refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
                            *piece_is_picked_up = PieceIsPickedUp::default();
                            return;
                        }

                        // Check if the AI has any legal moves before starting the search.
                        let ai_is_white = *player_color == PlayerColor::Black;
                        let ai_moves = get_all_legal_moves_for_color(
                            &mut chess_board.chess_board,
                            &move_generator_res.magic,
                            ai_is_white,
                        );
                        if ai_moves.is_empty() {
                            if move_generator_res.magic.is_king_in_check(&chess_board.chess_board, ai_is_white) {
                                println!("Checkmate — player wins!");
                                *game_over_state = GameOverState::PlayerWins;
                            } else {
                                println!("Stalemate!");
                                *game_over_state = GameOverState::Stalemate;
                            }
                            refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
                            *piece_is_picked_up = PieceIsPickedUp::default();
                            return;
                        }
                    }
                    refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
                    game_event_ew.write(ChessEvent::new(ChessAction::MakeMove));
                } else {
                    *piece_is_picked_up = PieceIsPickedUp::default();
                    refresh_pieces_events.write(RefreshPiecesFromBoardEvent);
                }
            }
        }
    }
}

pub fn hide_debug_on_drop(
    mut chess_input_er: MessageReader<DropPieceEvent>,
    mut debug_squares_query: Query<(Entity, &DebugSquare)>,
    mut commands: Commands,
) {
    for _ in chess_input_er.read() {
        for (entity, _) in debug_squares_query.iter_mut() {
            commands.entity(entity).insert(Visibility::Hidden);
        }
        break;
    }
}
