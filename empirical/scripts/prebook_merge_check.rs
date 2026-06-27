use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
struct LineRef {
    src_idx: usize,
    offset: u64,
    len: usize,
}

#[derive(Debug, Default)]
struct SideStats {
    rows: u64,
    bad_rows: u64,
    dups_ignored: u64,
}

#[derive(Debug, Clone)]
struct Args {
    target: PathBuf,
    outdir: PathBuf,
    runs: Vec<PathBuf>,
    merge: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut target: Option<PathBuf> = None;
    let mut outdir: Option<PathBuf> = None;
    let mut runs: Vec<PathBuf> = Vec::new();
    let mut merge = true;

    let mut it = env::args().skip(1).peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--target" => {
                let v = it.next().ok_or("missing value for --target")?;
                target = Some(PathBuf::from(v));
            }
            "--outdir" => {
                let v = it.next().ok_or("missing value for --outdir")?;
                outdir = Some(PathBuf::from(v));
            }
            "--run" => {
                let v = it.next().ok_or("missing value for --run")?;
                runs.push(PathBuf::from(v));
            }
            "--check-only" => {
                merge = false;
            }
            "--help" | "-h" => {
                return Err(
                    "Usage:\n  prebook_merge_check --target <prebook_ledgers_full.txt> --outdir <final_dir> --run <run_dir> [--run <run_dir> ...] [--check-only]"
                        .to_string(),
                );
            }
            _ => return Err(format!("unknown arg: {a}")),
        }
    }

    let target = target.ok_or("missing --target")?;
    let outdir = outdir.ok_or("missing --outdir")?;
    if runs.is_empty() {
        return Err("missing at least one --run".to_string());
    }
    Ok(Args {
        target,
        outdir,
        runs,
        merge,
    })
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' | b'\r' | b'\n' => i += 1,
            _ => break,
        }
    }
    i
}

fn parse_ledger_index(line: &[u8]) -> Option<u64> {
    let key = br#""ledger_index""#;
    let pos = find_subslice(line, key)?;
    let mut i = pos + key.len();
    i = skip_ws(line, i);
    if i >= line.len() || line[i] != b':' {
        return None;
    }
    i += 1;
    i = skip_ws(line, i);

    let mut j = i;
    while j < line.len() && line[j].is_ascii_digit() {
        j += 1;
    }
    if j == i {
        return None;
    }
    std::str::from_utf8(&line[i..j]).ok()?.parse::<u64>().ok()
}

fn has_offers_array(line: &[u8]) -> bool {
    let key = br#""offers""#;
    let Some(pos) = find_subslice(line, key) else {
        return false;
    };
    let mut i = pos + key.len();
    i = skip_ws(line, i);
    if i >= line.len() || line[i] != b':' {
        return false;
    }
    i += 1;
    i = skip_ws(line, i);
    i < line.len() && line[i] == b'['
}

fn scan_side(
    files: &[PathBuf],
    filename: &str,
) -> io::Result<(HashMap<u64, LineRef>, SideStats, Vec<PathBuf>)> {
    let mut idx_map: HashMap<u64, LineRef> = HashMap::new();
    let mut stats = SideStats::default();
    let mut used_files: Vec<PathBuf> = Vec::new();

    for dir in files {
        let fp = dir.join(filename);
        if !fp.exists() {
            continue;
        }
        let src_idx = used_files.len();
        used_files.push(fp.clone());

        let f = File::open(&fp)?;
        let mut r = BufReader::new(f);
        let mut buf = Vec::<u8>::new();
        let mut off: u64 = 0;

        loop {
            buf.clear();
            let n = r.read_until(b'\n', &mut buf)?;
            if n == 0 {
                break;
            }
            stats.rows += 1;
            let valid = has_offers_array(&buf);
            let li = parse_ledger_index(&buf);
            match (valid, li) {
                (true, Some(ledger)) => {
                    if idx_map.contains_key(&ledger) {
                        stats.dups_ignored += 1;
                    } else {
                        idx_map.insert(
                            ledger,
                            LineRef {
                                src_idx,
                                offset: off,
                                len: n,
                            },
                        );
                    }
                }
                _ => stats.bad_rows += 1,
            }
            off += n as u64;
        }
    }

    Ok((idx_map, stats, used_files))
}

fn read_target(path: &Path) -> io::Result<HashSet<u64>> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut s = HashSet::new();
    for line in r.lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(v) = t.parse::<u64>() {
            s.insert(v);
        }
    }
    Ok(s)
}

fn write_merged_side(
    out_path: &Path,
    ledgers: &[u64],
    idx_map: &HashMap<u64, LineRef>,
    src_files: &[PathBuf],
) -> io::Result<()> {
    let mut in_files: Vec<File> = src_files
        .iter()
        .map(File::open)
        .collect::<io::Result<Vec<File>>>()?;
    let mut out = File::create(out_path)?;

    for li in ledgers {
        let Some(rf) = idx_map.get(li) else {
            continue;
        };
        let f = &mut in_files[rf.src_idx];
        f.seek(SeekFrom::Start(rf.offset))?;
        let mut buf = vec![0u8; rf.len];
        f.read_exact(&mut buf)?;
        out.write_all(&buf)?;
    }
    out.flush()?;
    Ok(())
}

fn write_u64_list(path: &Path, vals: &[u64]) -> io::Result<()> {
    let mut f = File::create(path)?;
    for v in vals {
        writeln!(f, "{v}")?;
    }
    f.flush()?;
    Ok(())
}

fn main() -> io::Result<()> {
    let args = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    let mut source_dirs: Vec<PathBuf> = Vec::new();
    let subdirs = [
        "prebook_qn_pool",
        "prebook_s1",
        "prebook_s2",
        "prebook_rusty",
        "prebook_cluster",
        "prebook_qn_rebalance",
        "prebook_rusty_rebalance",
        "prebook_rusty_takeover_35_44",
    ];
    for run in &args.runs {
        for s in &subdirs {
            let d = run.join(s);
            if d.exists() {
                source_dirs.push(d);
            }
        }
    }

    let target = read_target(&args.target)?;
    let mut target_sorted: Vec<u64> = target.iter().copied().collect();
    target_sorted.sort_unstable();

    let (x_map, x_stats, x_files) = scan_side(&source_dirs, "book_rusd_xrp_getsXRP.ndjson")?;
    let (r_map, r_stats, r_files) = scan_side(&source_dirs, "book_rusd_xrp_getsrUSD.ndjson")?;

    let x_keys: HashSet<u64> = x_map.keys().copied().collect();
    let r_keys: HashSet<u64> = r_map.keys().copied().collect();

    let mut done: Vec<u64> = x_keys.intersection(&r_keys).copied().collect();
    done.sort_unstable();
    let done_set: HashSet<u64> = done.iter().copied().collect();

    let mut remaining: Vec<u64> = target
        .difference(&done_set)
        .copied()
        .collect();
    remaining.sort_unstable();

    let mut extra: Vec<u64> = done_set
        .difference(&target)
        .copied()
        .collect();
    extra.sort_unstable();

    fs::create_dir_all(&args.outdir)?;

    let out_x = args.outdir.join("book_rusd_xrp_getsXRP.ndjson");
    let out_r = args.outdir.join("book_rusd_xrp_getsrUSD.ndjson");
    let out_summary = args.outdir.join("prebook_final_summary.json");
    let out_remaining = args.outdir.join("remaining_ledgers.txt");
    let out_extra = args.outdir.join("extra_ledgers.txt");

    if args.merge {
        write_merged_side(&out_x, &done, &x_map, &x_files)?;
        write_merged_side(&out_r, &done, &r_map, &r_files)?;
    }
    write_u64_list(&out_remaining, &remaining)?;
    write_u64_list(&out_extra, &extra)?;

    let mut summary = String::new();
    summary.push_str("{\n");
    summary.push_str(&format!("  \"target_total\": {},\n", target.len()));
    summary.push_str(&format!("  \"done_total\": {},\n", done.len()));
    summary.push_str(&format!("  \"remaining_total\": {},\n", remaining.len()));
    summary.push_str(&format!("  \"extra_total\": {},\n", extra.len()));
    summary.push_str("  \"getsXRP\": {\n");
    summary.push_str(&format!("    \"rows\": {},\n", x_stats.rows));
    summary.push_str(&format!("    \"bad_rows\": {},\n", x_stats.bad_rows));
    summary.push_str(&format!("    \"dups_ignored\": {},\n", x_stats.dups_ignored));
    summary.push_str(&format!("    \"unique_ledgers\": {}\n", x_map.len()));
    summary.push_str("  },\n");
    summary.push_str("  \"getsrUSD\": {\n");
    summary.push_str(&format!("    \"rows\": {},\n", r_stats.rows));
    summary.push_str(&format!("    \"bad_rows\": {},\n", r_stats.bad_rows));
    summary.push_str(&format!("    \"dups_ignored\": {},\n", r_stats.dups_ignored));
    summary.push_str(&format!("    \"unique_ledgers\": {}\n", r_map.len()));
    summary.push_str("  },\n");
    summary.push_str(&format!("  \"merge_enabled\": {},\n", if args.merge { "true" } else { "false" }));
    summary.push_str("  \"source_dirs\": [\n");
    for (i, d) in source_dirs.iter().enumerate() {
        let comma = if i + 1 == source_dirs.len() { "" } else { "," };
        summary.push_str(&format!("    \"{}\"{}\n", d.display(), comma));
    }
    summary.push_str("  ],\n");
    summary.push_str("  \"outputs\": {\n");
    summary.push_str(&format!("    \"summary\": \"{}\",\n", out_summary.display()));
    summary.push_str(&format!("    \"remaining\": \"{}\",\n", out_remaining.display()));
    summary.push_str(&format!("    \"extra\": \"{}\"", out_extra.display()));
    if args.merge {
        summary.push_str(&format!(
            ",\n    \"getsXRP\": \"{}\",\n    \"getsrUSD\": \"{}\"\n",
            out_x.display(),
            out_r.display()
        ));
    } else {
        summary.push('\n');
    }
    summary.push_str("  }\n");
    summary.push_str("}\n");
    fs::write(&out_summary, &summary)?;

    println!("target_total={}", target.len());
    println!("done_total={}", done.len());
    println!("remaining_total={}", remaining.len());
    println!("extra_total={}", extra.len());
    println!("bad_rows_total={}", x_stats.bad_rows + r_stats.bad_rows);
    println!("summary_file={}", out_summary.display());
    if args.merge {
        println!("merged_getsXRP={}", out_x.display());
        println!("merged_getsrUSD={}", out_r.display());
    }

    Ok(())
}
