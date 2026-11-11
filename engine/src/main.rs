use std::error::Error;
use std::io;

use engine::{Backend, DraftSchedule, Model, generate_scheduled};
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let backend = match args.next().as_deref() {
        Some("cpu") => Backend::Cpu,
        Some("metal") => Backend::Metal,
        Some("gpu") => Backend::Gpu,
        _ => return Err("usage: engine <cpu|metal|gpu> <model> <new tokens> [--draft <model>] [--gamma <count>] [--mxfp4] [--adaptive] [--max-gamma <count>] [--prefill-backend <cpu|metal|gpu>] [--draft-backend <cpu|metal|gpu>] [--stats] < prompt.json".into()),
    };
    let mut name = args.next().ok_or("missing model")?;
    let count = args.next().ok_or("missing token count")?.parse()?;
    let mut draft_name = None;
    let mut gamma = 5;
    let mut mxfp4 = false;
    let mut adaptive = false;
    let mut max_gamma = 8;
    let mut prefill_backend: Option<Backend> = None;
    let mut draft_backend: Option<Backend> = None;
    let mut stats = false;
    let mut time_rows: Option<String> = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--adaptive" => adaptive = true,
            "--max-gamma" => max_gamma = args.next().ok_or("missing max gamma")?.parse()?,
            "--prefill-backend" => {
                prefill_backend = Some(args.next().ok_or("missing prefill backend")?.parse()?)
            }
            "--draft-backend" => {
                draft_backend = Some(args.next().ok_or("missing draft backend")?.parse()?)
            }
            "--stats" => stats = true,
            "--time-rows" => time_rows = Some(args.next().ok_or("missing row counts")?),
            "--mxfp4" => mxfp4 = true,
            "--draft" => draft_name = Some(args.next().ok_or("missing draft model")?),
            "--gamma" => gamma = args.next().ok_or("missing gamma")?.parse()?,
            _ => return Err(format!("unexpected argument: {flag}").into()),
        }
    }
    if gamma == 0 {
        return Err("gamma must be positive".into());
    }
    if (adaptive || draft_backend.is_some() || stats) && draft_name.is_none() {
        return Err("--adaptive, --draft-backend and --stats require --draft".into());
    }
    let schedule = if adaptive {
        DraftSchedule::adaptive(gamma, max_gamma)?
    } else {
        DraftSchedule::fixed(gamma)
    };
    let prompt: Vec<usize> = serde_json::from_reader(io::stdin().lock())?;
    if prompt.is_empty() {
        return Err("prompt is empty".into());
    }

    if mxfp4 {
        name = format!("{name}-MXFP4");
        draft_name = draft_name.map(|name| format!("{name}-MXFP4"));
    }
    let model = Model::load_with_prefill(&name, backend, prefill_backend.unwrap_or(backend))?;
    #[cfg(target_os = "macos")]
    if let Some(list) = time_rows {
        // Profiling: how long one pass takes for each number of tokens.
        for rows in list.split(',') {
            let rows: usize = rows.parse()?;
            let (wall_ms, gpu_ms) = model.time_pass(&prompt, rows, 15);
            println!("{}", serde_json::json!({"rows": rows, "wall_ms": wall_ms, "gpu_ms": gpu_ms}));
        }
        return Ok(());
    }
    let tokens = if let Some(draft_name) = draft_name {
        let draft_backend = draft_backend.unwrap_or(backend);
        let draft = Model::load_with_prefill(
            &draft_name,
            draft_backend,
            prefill_backend.unwrap_or(draft_backend),
        )?;
        let started = Instant::now();
        let result = generate_scheduled(&model, &draft, &prompt, count, schedule);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        if stats {
            let stats = serde_json::json!({
                "adaptive": adaptive,
                "elapsed_ms": elapsed_ms,
                "generated": result.tokens.len() - prompt.len(),
                "accepted": result.accepted,
                "drafted": result.drafted,
                "rounds": result.rounds,
                "target_prefill_backend": prefill_backend.unwrap_or(backend),
                "draft_prefill_backend": prefill_backend.unwrap_or(draft_backend),
                "draft_backend": draft_backend,
                "verify_backend": backend,
                "prefill": result.prefill,
                "draft": result.draft,
                "verify": result.verify,
                "round_stats": result.round_stats,
            });
            eprintln!("{stats}");
        }
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
