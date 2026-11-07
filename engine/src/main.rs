use std::error::Error;
use std::io;

use engine::{Backend, Model};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let backend = match args.next().as_deref() {
        Some("cpu") => Backend::Cpu,
        Some("metal") => Backend::Metal,
        _ => return Err("usage: engine <cpu|metal> <model> <new tokens> < prompt.json".into()),
    };
    let name = args.next().ok_or("missing model")?;
    let count = args.next().ok_or("missing token count")?.parse()?;
    if args.next().is_some() {
        return Err("unexpected argument".into());
    }
    let prompt: Vec<usize> = serde_json::from_reader(io::stdin().lock())?;
    if prompt.is_empty() {
        return Err("prompt is empty".into());
    }

    let model = Model::load(&name, backend)?;
    let tokens = model.generate(&prompt, count);
    println!("{}", serde_json::to_string(&tokens[prompt.len()..])?);
    Ok(())
}
