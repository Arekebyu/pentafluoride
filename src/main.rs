use std::{fs::File, io::BufReader, path::PathBuf, sync::Arc};

use enumset::EnumSet;

use crate::data::*;
#[macro_use]
mod data;
mod movegen;
use pentafluoride::{evals, interface::run};

fn main() {
    puffin::set_scopes_on(true);
    let _puffin_server =
        puffin_http::Server::new(&format!("0.0.0.0:{}", puffin_http::DEFAULT_PORT));

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
    let path = PathBuf::from("../src/default.json");
    let weights = {
        let f = BufReader::new(File::open(path).unwrap());
        Arc::new(serde_json::from_reader(f).unwrap())
    };

    futures::pin_mut!(incoming);
    futures::pin_mut!(outgoing);

    futures::executor::block_on(run(incoming, outgoing, weights));
}
