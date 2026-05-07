use enumset::EnumSet;
use futures::prelude::*;
use std::convert::Infallible;
use std::sync::Arc;

use crate::bot::Bot;
use crate::data::*;
use crate::sharing::SharedState;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub enum IncomingMessage {
    Start {
        hold: Option<Piece>,
        queue: Vec<Piece>,
        combo: u32,
        b2b: u32,
        board: Board,
    },
    Stop,
    Suggest,
    Advance {
        mv: Placement,
    },
    AddPiece {
        piece: Piece,
    },
    Exit,
}

#[derive(Serialize, Deserialize)]
pub enum OutgoingMessage {
    Ready,
    Suggestions(Vec<Placement>),
    State(GameState),
}

impl From<[[Option<char>; 10]; 40]> for Board {
    fn from(_: [[Option<char>; 10]; 40]) -> Self {
        todo!()
    }
}

pub async fn run(
    mut incoming: impl Stream<Item = IncomingMessage> + Unpin,
    mut outgoing: impl Sink<OutgoingMessage, Error = Infallible> + Unpin,
) {
    outgoing.send(OutgoingMessage::Ready).await.unwrap();

    let bot = Arc::new(SharedState::<Bot>::new());
    spawn_workers(&bot);

    let mut waiting_on_piece = None;
    while let Some(msg) = incoming.next().await {
        match msg {
            IncomingMessage::Start {
                hold,
                queue,
                combo,
                b2b,
                board,
            } => {
                let mut queue = queue.into_iter().map(Into::into);
                let combo = combo.min(20) as u8;
                match hold.or_else(|| queue.next()) {
                    Some(hold) => bot.start(Bot::new(
                        GameState {
                            hold,
                            combo,
                            b2b,
                            board,
                            bag: EnumSet::all() - hold,
                        },
                        queue,
                    )),
                    None => {
                        bot.stop();
                        waiting_on_piece = Some((board, combo, b2b, queue))
                    }
                }
            }
            IncomingMessage::Stop => {
                bot.stop();
                waiting_on_piece = None;
            }
            IncomingMessage::Suggest => {
                if let Some(suggestions) = bot.read_op_if_exists(|bot| bot.suggest()) {
                    outgoing
                        .send(OutgoingMessage::Suggestions(suggestions))
                        .await
                        .unwrap();
                }
            }
            IncomingMessage::Advance { mv } => {
                if let Some(state) = bot.write_op_if_exists(|bot| {
                    bot.advance(mv);
                    bot.state()
                }) {
                    outgoing
                        .send(OutgoingMessage::State(state))
                        .await
                        .unwrap();
                }
            }
            IncomingMessage::AddPiece { piece } => {
                if let Some((board, combo, b2b, mut queue)) = waiting_on_piece.take() {
                    bot.start(Bot::new(
                        GameState {
                            hold: piece,
                            combo,
                            b2b,
                            board,
                            bag: EnumSet::all() - piece,
                        },
                        std::iter::empty(),
                    ));
                } else {
                    bot.write_op_if_exists(|bot| bot.add_piece(piece));
                }
            }
            IncomingMessage::Exit => {
                bot.stop();
                break;
            }
        }
    }
}

fn spawn_workers(bot: &Arc<SharedState<Bot>>) {
    let bot = bot.clone();
    std::thread::spawn(move || {
        loop {
            bot.read_op(|bot| bot.expand());
        }
    });
}
