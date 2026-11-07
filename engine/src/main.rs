use std::error::Error;
use std::io;

use engine::{Backend, Model, generate_speculative};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let backend = match args.next().as_deref() {
        Some("cpu") => Backend::Cpu,
        Some("metal") => Backend::Metal,
        _ => return Err("usage: engine <cpu|metal> <model> <new tokens> [--draft <model>] [--gamma <count>] [--mxfp4] < prompt.json".into()),
    };
    let mut name = args.next().ok_or("missing model")?;
    let count = args.next().ok_or("missing token count")?.parse()?;
    let mut draft_name = None;
    let mut gamma = 5;
    let mut mxfp4 = false;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--mxfp4" => mxfp4 = true,
            "--draft" => draft_name = Some(args.next().ok_or("missing draft model")?),
            "--gamma" => gamma = args.next().ok_or("missing gamma")?.parse()?,
            _ => return Err(format!("unexpected argument: {flag}").into()),
        }
    }
    if gamma == 0 {
        return Err("gamma must be positive".into());
    }
    let prompt: Vec<usize> = serde_json::from_reader(io::stdin().lock())?;
    if prompt.is_empty() {
        return Err("prompt is empty".into());
    }

    if mxfp4 {
        name = format!("{name}-MXFP4");
        draft_name = draft_name.map(|name| format!("{name}-MXFP4"));
    }
    let model = Model::load(&name, backend)?;
    let tokens = if let Some(draft_name) = draft_name {
        let draft = Model::load(&draft_name, backend)?;
        let result = generate_speculative(&model, &draft, &prompt, count, gamma);
        eprintln!(
            "accepted {} / {} draft tokens in {} rounds",
            result.accepted, result.drafted, result.rounds
        );
        result.tokens
    } else {
        model.generate(&prompt, count)
    };
    println!("{}", serde_json::to_string(&tokens[prompt.len()..])?);
    Ok(())
}
