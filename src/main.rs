use enumset::EnumSet;

use crate::data::*;
#[macro_use]
mod data;
mod movegen;
use pentafluoride::interface::run;

fn main() {
    let incoming = futures::stream::repeat_with(|| {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).unwrap();
        serde_json::from_str(&line).unwrap()
    });

    let outgoing = futures::sink::unfold((), |_, msg| {
        serde_json::to_writer(std::io::stdout(), &msg).unwrap();
        println!();
        use std::io::Write;
        std::io::stdout().flush().unwrap();
        async { Ok(()) }
    });

    futures::pin_mut!(incoming);
    futures::pin_mut!(outgoing);

    futures::executor::block_on(run(incoming, outgoing));
}
