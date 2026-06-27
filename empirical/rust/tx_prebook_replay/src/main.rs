use anyhow::{anyhow, Context, Result};
use clap::Parser;
use indexmap::IndexMap;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::Decimal;
use serde_json::{json, Map, Value};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const XRP: &str = "XRP";
const FUNDING_FIELDS: [&str; 3] = ["owner_funds", "taker_gets_funded", "taker_pays_funded"];
const XRPL_OWNER_FUNDS_RESERVE_BASE_DROPS: i64 = 1_000_000;
const XRPL_OWNER_FUNDS_RESERVE_INC_DROPS: i64 = 200_000;
const QUALITY_MANTISSA_MASK: u64 = 0x00FF_FFFF_FFFF_FFFF;

#[derive(Parser, Debug, Clone)]
#[command(
    about = "Replay tx-level prebook from ledger snapshots + metadata and emit NDJSON snapshots"
)]
struct Args {
    #[arg(long)]
    ledger_start: i64,
    #[arg(long)]
    ledger_end: i64,
    #[arg(long)]
    book_gets_xrp: String,
    #[arg(long)]
    book_gets_rusd: String,
    #[arg(long)]
    prebook_shards_dir: Option<String>,
    #[arg(long)]
    metadata_ndjson: String,
    #[arg(long)]
    target_tx_file: Option<String>,
    #[arg(long)]
    account_lines_snapshots: Option<String>,
    #[arg(long)]
    amm_swaps: Option<String>,
    #[arg(long)]
    clob_legs: Option<String>,
    #[arg(long)]
    output_dir: String,
    #[arg(long, default_value = "tx_prebook_replay_snapshots.ndjson")]
    snapshots_name: String,
    #[arg(long, default_value = "tx_prebook_replay_events.ndjson")]
    events_name: String,
    #[arg(long, default_value_t = false)]
    no_events: bool,
    #[arg(long, default_value_t = 0)]
    max_offers_per_side: usize,
    #[arg(long, default_value_t = 2000)]
    progress_every: usize,
    #[arg(long, default_value_t = false)]
    allow_filtered_ledger_metadata: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    GetsXrp,
    GetsRusd,
}

impl Side {
    fn as_str(self) -> &'static str {
        match self {
            Side::GetsXrp => "getsXRP",
            Side::GetsRusd => "getsrUSD",
        }
    }
}

#[derive(Debug, Clone)]
struct Change {
    step_idx: i64,
    node_kind: String,
    offer_id: String,
    pre_offer: Option<Value>,
    post_offer: Option<Value>,
    side_pre: Option<Side>,
    side_post: Option<Side>,
}

#[derive(Debug, Clone)]
struct ShardInfo {
    sid: i64,
    ledger_min: i64,
    ledger_max: i64,
    path: PathBuf,
}

#[derive(Debug)]
struct BookCursor {
    side: String,
    paths: Vec<PathBuf>,
    path_idx: usize,
    reader: Option<BufReader<File>>,
    current_path: Option<PathBuf>,
    current_line_no: usize,
    next_snapshot: Option<(i64, Vec<Value>)>,
}

#[derive(Debug, Clone)]
struct OfferRow {
    quality_sort_key: Option<u64>,
    row: Value,
}

#[derive(Debug, Clone)]
struct AccountLineSnapshot {
    owner_account: String,
    currency: String,
    issuer: String,
    balance: Decimal,
}

fn xrp_quantum() -> Decimal {
    Decimal::new(1, 6)
}

fn print_progress(
    line_no: usize,
    matched: usize,
    emitted: usize,
    start: Instant,
    final_line: bool,
) {
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let rate = (line_no as f64) / elapsed;
    let text = format!(
        "[replay] lines={} matched_tx={} emitted={} elapsed={:.1}s rate={:.1} line/s",
        line_no, matched, emitted, elapsed, rate
    );
    if final_line {
        println!("\r{}", text);
    } else {
        print!("\r{}", text);
        let _ = std::io::stdout().flush();
    }
}

fn as_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

fn as_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn decimal_to_string(d: Decimal) -> String {
    d.normalize().to_string()
}

fn parse_scientific_decimal_str(s: &str) -> Option<Decimal> {
    let e_pos = s.find('e').or_else(|| s.find('E'))?;
    let mantissa_raw = s.get(..e_pos)?.trim();
    let exp_raw = s.get(e_pos + 1..)?.trim();
    if mantissa_raw.is_empty() || exp_raw.is_empty() {
        return None;
    }
    let exp = exp_raw.parse::<i32>().ok()?;

    let mut sign = "";
    let mut mantissa = mantissa_raw;
    if let Some(rest) = mantissa.strip_prefix('+') {
        mantissa = rest;
    } else if let Some(rest) = mantissa.strip_prefix('-') {
        sign = "-";
        mantissa = rest;
    }
    if mantissa.is_empty() {
        return None;
    }

    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((l, r)) => (l, r),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    if !(int_part.chars().all(|c| c.is_ascii_digit())
        && frac_part.chars().all(|c| c.is_ascii_digit()))
    {
        return None;
    }

    let mut digits = String::with_capacity(int_part.len() + frac_part.len());
    digits.push_str(int_part);
    digits.push_str(frac_part);
    if digits.is_empty() {
        return None;
    }

    let digits_trimmed = digits.trim_start_matches('0');
    if digits_trimmed.is_empty() {
        return Some(Decimal::ZERO);
    }
    digits = digits_trimmed.to_string();

    let int_len = int_part.len() as i32;
    let new_pos = int_len + exp;
    let len = digits.len() as i32;
    let mut plain = String::new();
    plain.push_str(sign);
    if new_pos <= 0 {
        plain.push_str("0.");
        for _ in 0..(-new_pos) {
            plain.push('0');
        }
        plain.push_str(&digits);
    } else if new_pos >= len {
        plain.push_str(&digits);
        for _ in 0..(new_pos - len) {
            plain.push('0');
        }
    } else {
        let split = new_pos as usize;
        plain.push_str(&digits[..split]);
        plain.push('.');
        plain.push_str(&digits[split..]);
    }

    if let Some(dot) = plain.find('.') {
        while plain.ends_with('0') {
            plain.pop();
        }
        if plain.ends_with('.') {
            plain.pop();
        }
        if plain == "-" {
            plain.push('0');
        } else if dot == 0 {
            plain.insert(0, '0');
        } else if plain.starts_with("-.") {
            plain.insert(1, '0');
        }
    }

    plain.parse::<Decimal>().ok()
}

fn parse_decimal_str(s: &str) -> Option<Decimal> {
    let t = s.trim();
    if t.is_empty() {
        return None;
    }
    t.parse::<Decimal>()
        .ok()
        .or_else(|| parse_scientific_decimal_str(t))
}

fn amt_currency(a: &Value) -> Option<String> {
    match a {
        Value::String(_) => Some(XRP.to_string()),
        Value::Object(m) => m.get("currency").and_then(as_string),
        _ => None,
    }
}

fn amt_decimal(a: &Value) -> Option<Decimal> {
    match a {
        Value::String(s) => parse_decimal_str(s).map(|d| d * xrp_quantum()),
        Value::Object(m) => m
            .get("value")
            .and_then(as_string)
            .and_then(|s| parse_decimal_str(&s)),
        _ => None,
    }
}

fn is_iou_amt(a: &Value, iou_currency: &str, iou_issuer: Option<&str>) -> bool {
    let Some(m) = a.as_object() else {
        return false;
    };
    let Some(cur) = m.get("currency").and_then(as_string) else {
        return false;
    };
    if cur != iou_currency {
        return false;
    }
    if let Some(issuer) = iou_issuer {
        let got = m.get("issuer").and_then(as_string);
        if got.as_deref() != Some(issuer) {
            return false;
        }
    }
    m.get("value").is_some()
}

fn book_offer_id(o: &Value) -> String {
    if let Some(idx) = o.get("index").and_then(as_string) {
        return idx;
    }
    let acc = o.get("Account").and_then(as_string);
    let seq = o.get("Sequence").and_then(as_string);
    if let (Some(acc), Some(seq)) = (acc, seq) {
        return format!("{}:{}", acc, seq);
    }
    "unknown-offer-id".to_string()
}

fn offer_side(o: &Value, iou_currency: &str, iou_issuer: Option<&str>) -> Option<Side> {
    let gets = o.get("TakerGets")?;
    let pays = o.get("TakerPays")?;
    if amt_currency(gets).as_deref() == Some(XRP) && is_iou_amt(pays, iou_currency, iou_issuer) {
        return Some(Side::GetsXrp);
    }
    if is_iou_amt(gets, iou_currency, iou_issuer) && amt_currency(pays).as_deref() == Some(XRP) {
        return Some(Side::GetsRusd);
    }
    None
}

fn quality_out_over_in(o: &Value, _side: Side) -> Option<Decimal> {
    let gets = o.get("TakerGets")?;
    let pays = o.get("TakerPays")?;
    let out = amt_decimal(gets)?;
    let inp = amt_decimal(pays)?;
    if out <= Decimal::ZERO || inp <= Decimal::ZERO {
        return None;
    }
    out.checked_div(inp)
}

fn quality_in_per_out(o: &Value) -> Option<Decimal> {
    let gets = o.get("TakerGets")?;
    let pays = o.get("TakerPays")?;
    let out = amt_decimal(gets)?;
    let inp = amt_decimal(pays)?;
    if out <= Decimal::ZERO || inp <= Decimal::ZERO {
        return None;
    }
    inp.checked_div(out)
}

fn fixed_decimal_string_from_mantissa_exponent(mantissa: u64, exponent: i32) -> String {
    let digits = mantissa.to_string();
    if exponent >= 0 {
        return format!("{}{}", digits, "0".repeat(exponent as usize));
    }

    let split = (digits.len() as i32) + exponent;
    if split > 0 {
        let split = split as usize;
        return format!("{}.{}", &digits[..split], &digits[split..]);
    }

    format!("0.{}{}", "0".repeat((-split) as usize), digits)
}

fn quality_from_book_directory_value(raw: &Value) -> Option<String> {
    let rate = quality_raw_from_book_directory_value(raw)?;
    let mantissa = rate & QUALITY_MANTISSA_MASK;
    if mantissa == 0 {
        return None;
    }
    let exponent = ((rate >> 56) as i32) - 100;
    Some(fixed_decimal_string_from_mantissa_exponent(
        mantissa, exponent,
    ))
}

fn quality_raw_from_book_directory_value(raw: &Value) -> Option<u64> {
    let text = as_string(raw)?;
    if text.len() < 16 {
        return None;
    }
    u64::from_str_radix(&text[text.len() - 16..], 16).ok()
}

fn normalize_offer(raw_offer: &Value, body: &Map<String, Value>) -> Option<Value> {
    let mut o = raw_offer.as_object()?.clone();
    if !(o.contains_key("TakerGets") && o.contains_key("TakerPays")) {
        return None;
    }

    if !o.contains_key("index") {
        if let Some(v) = body.get("LedgerIndex") {
            o.insert("index".to_string(), v.clone());
        }
    }
    if !o.contains_key("Account") {
        if let Some(v) = body.get("Account") {
            o.insert("Account".to_string(), v.clone());
        }
    }
    if !o.contains_key("Sequence") {
        if let Some(v) = body.get("Sequence") {
            o.insert("Sequence".to_string(), v.clone());
        }
    }
    for k in [
        "Flags",
        "BookDirectory",
        "BookNode",
        "OwnerNode",
        "Expiration",
    ] {
        if !o.contains_key(k) {
            if let Some(v) = body.get(k) {
                o.insert(k.to_string(), v.clone());
            }
        }
    }
    if !o.contains_key("quality") {
        let q = o.get("BookDirectory").and_then(quality_from_book_directory_value);
        if let Some(q) = q {
            o.insert("quality".to_string(), Value::String(q));
        }
    }
    Some(Value::Object(o))
}

fn merge_pre_from_modified(
    final_fields: &Map<String, Value>,
    prev_fields: &Map<String, Value>,
) -> Value {
    let mut pre = final_fields.clone();
    for (k, v) in prev_fields {
        pre.insert(k.clone(), v.clone());
    }
    Value::Object(pre)
}

fn build_offer_pre(node_kind: &str, body: &Map<String, Value>) -> Option<Value> {
    let final_fields = body
        .get("FinalFields")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let prev_fields = body
        .get("PreviousFields")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    match node_kind {
        "ModifiedNode" => {
            let pre = merge_pre_from_modified(&final_fields, &prev_fields);
            normalize_offer(&pre, body)
        }
        "DeletedNode" => {
            // DeletedNode pre-state must be reconstructed as pre-image:
            // FinalFields is the last state before deletion, and
            // PreviousFields carries tx-start values for changed fields.
            let pre = merge_pre_from_modified(&final_fields, &prev_fields);
            normalize_offer(&pre, body)
        }
        _ => None,
    }
}

fn build_offer_post(node_kind: &str, body: &Map<String, Value>) -> Option<Value> {
    let final_fields = body
        .get("FinalFields")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let new_fields = body
        .get("NewFields")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    match node_kind {
        "DeletedNode" => None,
        "ModifiedNode" => normalize_offer(&Value::Object(final_fields), body),
        "CreatedNode" => normalize_offer(&Value::Object(new_fields), body),
        _ => None,
    }
}

fn iter_offer_changes(
    tx_obj: &Value,
    iou_currency: &str,
    iou_issuer: Option<&str>,
) -> Result<Vec<Change>> {
    let mut out = Vec::<Change>::new();
    let Some(res) = tx_obj.get("result").and_then(|v| v.as_object()) else {
        return Ok(out);
    };
    let Some(meta) = res.get("meta").and_then(|v| v.as_object()) else {
        return Ok(out);
    };
    let Some(affected) = meta.get("AffectedNodes").and_then(|v| v.as_array()) else {
        return Ok(out);
    };
    for (i, node) in affected.iter().enumerate() {
        let Some(nobj) = node.as_object() else {
            continue;
        };
        let node_kind = ["CreatedNode", "ModifiedNode", "DeletedNode"]
            .iter()
            .find(|k| nobj.contains_key(**k))
            .map(|s| (*s).to_string());
        let Some(node_kind) = node_kind else {
            continue;
        };
        let Some(body) = nobj.get(&node_kind).and_then(|v| v.as_object()) else {
            continue;
        };
        if body.get("LedgerEntryType").and_then(as_string).as_deref() != Some("Offer") {
            continue;
        }
        let Some(offer_id) = body.get("LedgerIndex").and_then(as_string) else {
            continue;
        };
        let pre_offer = build_offer_pre(&node_kind, body);
        let post_offer = build_offer_post(&node_kind, body);
        let side_pre = pre_offer
            .as_ref()
            .and_then(|o| offer_side(o, iou_currency, iou_issuer));
        let side_post = post_offer
            .as_ref()
            .and_then(|o| offer_side(o, iou_currency, iou_issuer));
        if side_pre.is_none() && side_post.is_none() {
            continue;
        }
        out.push(Change {
            step_idx: i as i64,
            node_kind,
            offer_id,
            pre_offer,
            post_offer,
            side_pre,
            side_post,
        });
    }
    Ok(out)
}

fn extract_tx_fields(obj: &Value) -> Option<(i64, i64, String)> {
    let res = obj.get("result")?;
    let meta = res.get("meta")?;
    let li = obj
        .get("ledger_index")
        .and_then(as_i64)
        .or_else(|| res.get("ledger_index").and_then(as_i64))
        .or_else(|| res.get("inLedger").and_then(as_i64))?;
    let txi = meta.get("TransactionIndex").and_then(as_i64)?;
    let txh = obj
        .get("tx_hash")
        .and_then(as_string)
        .or_else(|| res.get("hash").and_then(as_string))?;
    Some((li, txi, txh))
}

impl BookCursor {
    fn new(paths: Vec<PathBuf>, side: &str) -> Result<Self> {
        let mut out = Self {
            side: side.to_string(),
            paths,
            path_idx: 0,
            reader: None,
            current_path: None,
            current_line_no: 0,
            next_snapshot: None,
        };
        out.next_snapshot = out.read_next_snapshot()?;
        Ok(out)
    }

    fn open_next_reader(&mut self) -> Result<bool> {
        while self.path_idx < self.paths.len() {
            let path = self.paths[self.path_idx].clone();
            self.path_idx += 1;
            let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
            self.reader = Some(BufReader::new(f));
            self.current_path = Some(path);
            self.current_line_no = 0;
            return Ok(true);
        }
        self.reader = None;
        self.current_path = None;
        Ok(false)
    }

    fn read_next_snapshot(&mut self) -> Result<Option<(i64, Vec<Value>)>> {
        let mut line = String::new();
        loop {
            if self.reader.is_none() && !self.open_next_reader()? {
                return Ok(None);
            }
            let Some(reader) = self.reader.as_mut() else {
                continue;
            };
            line.clear();
            let n = reader.read_line(&mut line)?;
            if n == 0 {
                self.reader = None;
                self.current_path = None;
                continue;
            }
            self.current_line_no += 1;
            let s = line.trim();
            if s.is_empty() {
                continue;
            }
            let path = self
                .current_path
                .as_ref()
                .ok_or_else(|| anyhow!("internal: missing current_path for {}", self.side))?;
            let obj: Value = serde_json::from_str(s).with_context(|| {
                format!("parse json {}:{}", path.display(), self.current_line_no)
            })?;
            let li = obj
                .get("ledger_index")
                .and_then(as_i64)
                .or_else(|| obj.get("ledger").and_then(as_i64))
                .or_else(|| obj.get("ledger_current_index").and_then(as_i64))
                .ok_or_else(|| {
                    anyhow!(
                        "missing ledger_index at {}:{}",
                        path.display(),
                        self.current_line_no
                    )
                })?;
            let offers = obj
                .get("offers")
                .and_then(|v| v.as_array().cloned())
                .or_else(|| {
                    obj.get("result")
                        .and_then(|v| v.get("offers"))
                        .and_then(|v| v.as_array().cloned())
                })
                .unwrap_or_default();
            return Ok(Some((li, offers)));
        }
    }

    fn take_exact(&mut self, ledger_index: i64) -> Result<Vec<Value>> {
        loop {
            let Some((li, _)) = self.next_snapshot.as_ref() else {
                break;
            };
            if *li < ledger_index {
                self.next_snapshot = self.read_next_snapshot()?;
                continue;
            }
            if *li == ledger_index {
                let (_, offers) = self
                    .next_snapshot
                    .take()
                    .ok_or_else(|| anyhow!("internal: missing snapshot for {}", self.side))?;
                self.next_snapshot = self.read_next_snapshot()?;
                return Ok(offers);
            }
            break;
        }
        Err(anyhow!(
            "missing exact ledger-boundary prebook snapshot: need ledger={} side={}",
            ledger_index,
            self.side
        ))
    }
}

fn pick_shard_files(
    shards_dir: &Path,
    side: &str,
    ledger_min: i64,
    ledger_max: i64,
) -> Result<Vec<PathBuf>> {
    let summary_path = shards_dir.join("summary.json");
    let summary_val: Value = serde_json::from_reader(
        File::open(&summary_path).with_context(|| format!("open {}", summary_path.display()))?,
    )
    .with_context(|| format!("parse {}", summary_path.display()))?;
    let ranges = summary_val
        .get(side)
        .and_then(|v| v.get("ranges"))
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            anyhow!(
                "missing ranges for side={} in {}",
                side,
                summary_path.display()
            )
        })?;

    let mut infos = Vec::<ShardInfo>::new();
    for r in ranges {
        let robj = match r.as_object() {
            Some(v) => v,
            None => continue,
        };
        let sid = robj
            .get("sid")
            .and_then(as_i64)
            .ok_or_else(|| anyhow!("range.sid missing"))?;
        let rmin = robj
            .get("ledger_min")
            .and_then(as_i64)
            .ok_or_else(|| anyhow!("range.ledger_min missing"))?;
        let rmax = robj
            .get("ledger_max")
            .and_then(as_i64)
            .ok_or_else(|| anyhow!("range.ledger_max missing"))?;
        let p = robj
            .get("path")
            .and_then(as_string)
            .ok_or_else(|| anyhow!("range.path missing"))?;
        let pb = {
            let c = PathBuf::from(p);
            if c.is_absolute() {
                c
            } else {
                shards_dir.join(c)
            }
        };
        infos.push(ShardInfo {
            sid,
            ledger_min: rmin,
            ledger_max: rmax,
            path: pb,
        });
    }

    let mut overlaps: Vec<ShardInfo> = infos
        .iter()
        .filter(|r| !(r.ledger_max < ledger_min || r.ledger_min > ledger_max))
        .cloned()
        .collect();
    let preds: Vec<ShardInfo> = infos
        .iter()
        .filter(|r| r.ledger_max < ledger_min)
        .cloned()
        .collect();

    if let Some(best_pred) = preds.iter().max_by_key(|r| r.ledger_max).cloned() {
        if !overlaps.iter().any(|x| x.sid == best_pred.sid) {
            overlaps.push(best_pred);
        }
    }
    if overlaps.is_empty() {
        return Err(anyhow!(
            "no prebook shards selected for [{}..{}] side={}",
            ledger_min,
            ledger_max,
            side
        ));
    }
    overlaps.sort_by_key(|r| r.sid);
    for r in &overlaps {
        if !r.path.exists() {
            return Err(anyhow!("missing prebook shard file {}", r.path.display()));
        }
    }
    Ok(overlaps.into_iter().map(|r| r.path).collect())
}

fn build_book_paths(
    full_path: &str,
    prebook_shards_dir: Option<&str>,
    side: &str,
    ledger_min: i64,
    ledger_max: i64,
) -> Result<Vec<PathBuf>> {
    if let Some(sd) = prebook_shards_dir {
        pick_shard_files(Path::new(sd), side, ledger_min, ledger_max)
    } else {
        Ok(vec![PathBuf::from(full_path)])
    }
}

fn find_iou_from_book_paths(paths: &[PathBuf]) -> Result<Option<(String, Option<String>)>> {
    for path in paths {
        let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut rdr = BufReader::new(f);
        let mut line = String::new();
        let mut line_no: usize = 0;
        loop {
            line.clear();
            let n = rdr.read_line(&mut line)?;
            if n == 0 {
                break;
            }
            line_no += 1;
            let s = line.trim();
            if s.is_empty() {
                continue;
            }
            let obj: Value = serde_json::from_str(s)
                .with_context(|| format!("parse json {}:{}", path.display(), line_no))?;
            let offers = obj.get("offers").and_then(|v| v.as_array()).or_else(|| {
                obj.get("result")
                    .and_then(|v| v.get("offers"))
                    .and_then(|v| v.as_array())
            });
            let Some(offers) = offers else {
                continue;
            };
            for o in offers {
                if let Some(gets) = o.get("TakerGets") {
                    if let Some(cur) = amt_currency(gets) {
                        if cur != XRP {
                            let issuer = gets
                                .as_object()
                                .and_then(|x| x.get("issuer"))
                                .and_then(as_string);
                            return Ok(Some((cur, issuer)));
                        }
                    }
                }
                if let Some(pays) = o.get("TakerPays") {
                    if let Some(cur) = amt_currency(pays) {
                        if cur != XRP {
                            let issuer = pays
                                .as_object()
                                .and_then(|x| x.get("issuer"))
                                .and_then(as_string);
                            return Ok(Some((cur, issuer)));
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

fn state_from_offers(
    offers: &[Value],
    iou_currency: &str,
    iou_issuer: Option<&str>,
) -> (IndexMap<String, Value>, IndexMap<String, Value>) {
    let mut gets_xrp = IndexMap::<String, Value>::new();
    let mut gets_rusd = IndexMap::<String, Value>::new();
    for o in offers {
        let Some(side) = offer_side(o, iou_currency, iou_issuer) else {
            continue;
        };
        let oid = book_offer_id(o);
        match side {
            Side::GetsXrp => {
                gets_xrp.insert(oid, o.clone());
            }
            Side::GetsRusd => {
                gets_rusd.insert(oid, o.clone());
            }
        }
    }
    (gets_xrp, gets_rusd)
}

fn serialize_offer_row(o: &Value, side: Side) -> OfferRow {
    let oid = book_offer_id(o);
    let quality_sort_key = o
        .get("BookDirectory")
        .and_then(quality_raw_from_book_directory_value);
    let q_out_over_in = quality_out_over_in(o, side);
    let q_in_per_out = quality_in_per_out(o).map(decimal_to_string);
    let quality_v = o.get("quality").and_then(as_string);
    let mut row = json!({
        "offer_id": oid,
        "account": o.get("Account"),
        "sequence": o.get("Sequence"),
        "book_directory": o.get("BookDirectory"),
        "quality": quality_v,
        "quality_out_over_in": q_out_over_in.map(decimal_to_string),
        "quality_in_per_out": q_in_per_out,
        "taker_gets": o.get("TakerGets"),
        "taker_pays": o.get("TakerPays"),
        "owner_funds": o.get("owner_funds"),
    });
    if let Some(obj) = row.as_object_mut() {
        if let Some(v) = o.get("taker_gets_funded") {
            if !v.is_null() {
                obj.insert("taker_gets_funded".to_string(), v.clone());
            }
        }
        if let Some(v) = o.get("taker_pays_funded") {
            if !v.is_null() {
                obj.insert("taker_pays_funded".to_string(), v.clone());
            }
        }
    }
    OfferRow {
        quality_sort_key,
        row,
    }
}

fn merge_offer_preserving_funding(preferred: &Value, fallback: Option<&Value>) -> Value {
    let Some(mut out_obj) = preferred.as_object().cloned() else {
        return preferred.clone();
    };
    let Some(fallback_obj) = fallback.and_then(|v| v.as_object()) else {
        return Value::Object(out_obj);
    };
    if out_obj.get("TakerGets") != fallback_obj.get("TakerGets")
        || out_obj.get("TakerPays") != fallback_obj.get("TakerPays")
    {
        return Value::Object(out_obj);
    }

    for key in FUNDING_FIELDS {
        let missing_or_null = out_obj.get(key).map_or(true, Value::is_null);
        if !missing_or_null {
            continue;
        }
        if let Some(v) = fallback_obj.get(key) {
            if !v.is_null() {
                out_obj.insert(key.to_string(), v.clone());
            }
        }
    }
    Value::Object(out_obj)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OwnerFundsDomain {
    XrpDrops(i128),
    IouDecimal(Decimal),
}

type XrpOwnerFundsByAccount = HashMap<String, i128>;
type IouOwnerFundsKey = (String, String, String);
type IouOwnerFundsByKey = HashMap<IouOwnerFundsKey, Decimal>;

#[derive(Debug, Clone, PartialEq, Eq)]
enum RunningOwnerFunds {
    Unknown,
    XrpDrops(i128),
    IouDecimal(Decimal),
}

fn parse_xrp_owner_funds_to_drops(raw: &str) -> Result<i128> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(anyhow!("invalid XRP owner_funds: {:?}", raw));
    }
    if !trimmed.contains('.') && !trimmed.contains('e') && !trimmed.contains('E') {
        let drops = trimmed
            .parse::<i128>()
            .with_context(|| format!("invalid XRP owner_funds: {:?}", raw))?;
        if drops < 0 {
            return Err(anyhow!(
                "invalid XRP owner_funds: negative amount: {:?}",
                raw
            ));
        }
        return Ok(drops);
    }
    let value =
        parse_decimal_str(trimmed).ok_or_else(|| anyhow!("invalid XRP owner_funds: {:?}", raw))?;
    let drops = value
        .checked_mul(Decimal::from(1_000_000i64))
        .ok_or_else(|| anyhow!("overflow while scaling XRP owner_funds: {:?}", raw))?;
    if drops.fract() != Decimal::ZERO {
        return Err(anyhow!(
            "invalid XRP owner_funds: not drop-aligned: {:?}",
            raw
        ));
    }
    let drops_i128 = drops
        .trunc()
        .to_i128()
        .ok_or_else(|| anyhow!("invalid XRP owner_funds: out of range: {:?}", raw))?;
    if drops_i128 < 0 {
        return Err(anyhow!(
            "invalid XRP owner_funds: negative amount: {:?}",
            raw
        ));
    }
    Ok(drops_i128)
}

fn parse_owner_funds_domain(obj: &Map<String, Value>) -> Result<Option<OwnerFundsDomain>> {
    let Some(owner_funds_raw) = obj.get("owner_funds").and_then(as_string) else {
        return Ok(None);
    };
    let Some(taker_gets) = obj.get("TakerGets") else {
        return Err(anyhow!("owner_funds present but TakerGets missing"));
    };
    if taker_gets.is_string() {
        return Ok(Some(OwnerFundsDomain::XrpDrops(
            parse_xrp_owner_funds_to_drops(&owner_funds_raw)?,
        )));
    }
    let Some(_gets_obj) = taker_gets.as_object() else {
        return Err(anyhow!("owner_funds present but TakerGets domain invalid"));
    };
    let owner_funds = parse_decimal_str(&owner_funds_raw)
        .ok_or_else(|| anyhow!("invalid IOU owner_funds: {:?}", owner_funds_raw))?;
    if owner_funds < Decimal::ZERO {
        return Err(anyhow!(
            "invalid IOU owner_funds: negative amount: {:?}",
            owner_funds_raw
        ));
    }
    Ok(Some(OwnerFundsDomain::IouDecimal(owner_funds)))
}

fn normalize_account_owner_funds(
    offers_by_id: &IndexMap<String, Value>,
) -> Result<IndexMap<String, Value>> {
    let mut account_owner_funds = HashMap::<String, OwnerFundsDomain>::new();

    for offer in offers_by_id.values() {
        let Some(obj) = offer.as_object() else {
            continue;
        };
        let Some(account) = obj.get("Account").and_then(as_string) else {
            continue;
        };
        let Some(owner_funds) = parse_owner_funds_domain(obj)
            .with_context(|| format!("invalid owner_funds for account {}", account))?
        else {
            continue;
        };
        match &owner_funds {
            OwnerFundsDomain::XrpDrops(v) if *v <= 0 => continue,
            OwnerFundsDomain::IouDecimal(v) if *v <= Decimal::ZERO => continue,
            _ => {}
        }
        match account_owner_funds.get(&account) {
            None => {
                account_owner_funds.insert(account, owner_funds);
            }
            Some(OwnerFundsDomain::XrpDrops(prev)) => match owner_funds {
                OwnerFundsDomain::XrpDrops(curr) if curr < *prev => {
                    account_owner_funds.insert(account, OwnerFundsDomain::XrpDrops(curr));
                }
                OwnerFundsDomain::XrpDrops(_) => {}
                OwnerFundsDomain::IouDecimal(_) => {
                    return Err(anyhow!("mixed owner_funds domains for account {}", account));
                }
            },
            Some(OwnerFundsDomain::IouDecimal(prev)) => match owner_funds {
                OwnerFundsDomain::IouDecimal(curr) if curr < *prev => {
                    account_owner_funds.insert(account, OwnerFundsDomain::IouDecimal(curr));
                }
                OwnerFundsDomain::IouDecimal(_) => {}
                OwnerFundsDomain::XrpDrops(_) => {
                    return Err(anyhow!("mixed owner_funds domains for account {}", account));
                }
            },
        }
    }

    let mut out = IndexMap::<String, Value>::new();
    for (oid, offer) in offers_by_id {
        let Some(obj) = offer.as_object() else {
            out.insert(oid.clone(), offer.clone());
            continue;
        };
        let Some(account) = obj.get("Account").and_then(as_string) else {
            out.insert(oid.clone(), offer.clone());
            continue;
        };
        let Some(shared_owner_funds) = account_owner_funds.get(&account) else {
            out.insert(oid.clone(), offer.clone());
            continue;
        };
        let mut patched = obj.clone();
        let owner_funds_text = match shared_owner_funds {
            OwnerFundsDomain::XrpDrops(v) => v.to_string(),
            OwnerFundsDomain::IouDecimal(v) => decimal_to_string(*v),
        };
        patched.insert("owner_funds".to_string(), Value::String(owner_funds_text));
        out.insert(oid.clone(), Value::Object(patched));
    }
    Ok(out)
}

fn seed_running_owner_funds_from_state(
    state_side: &IndexMap<String, Value>,
    xrp_owner_funds_by_account: &mut XrpOwnerFundsByAccount,
    iou_owner_funds_by_key: &mut IouOwnerFundsByKey,
) -> Result<()> {
    for offer in state_side.values() {
        let Some(obj) = offer.as_object() else {
            continue;
        };
        let Some(account) = obj.get("Account").and_then(as_string) else {
            continue;
        };
        let Some(owner_funds) = parse_owner_funds_domain(obj).with_context(|| {
            format!(
                "invalid owner_funds while seeding running state for account {}",
                account
            )
        })?
        else {
            continue;
        };
        match owner_funds {
            OwnerFundsDomain::XrpDrops(v) => {
                let entry = xrp_owner_funds_by_account.entry(account).or_insert(v);
                if v < *entry {
                    *entry = v;
                }
            }
            OwnerFundsDomain::IouDecimal(v) => {
                let Some(gets_obj) = obj.get("TakerGets").and_then(|v| v.as_object()) else {
                    return Err(anyhow!(
                        "owner_funds present but TakerGets is not IOU object"
                    ));
                };
                let currency = gets_obj
                    .get("currency")
                    .and_then(as_string)
                    .ok_or_else(|| anyhow!("IOU owner_funds seed missing TakerGets.currency"))?;
                let issuer = gets_obj
                    .get("issuer")
                    .and_then(as_string)
                    .ok_or_else(|| anyhow!("IOU owner_funds seed missing TakerGets.issuer"))?;
                let entry = iou_owner_funds_by_key
                    .entry((account, currency, issuer))
                    .or_insert(v);
                if v < *entry {
                    *entry = v;
                }
            }
        }
    }
    Ok(())
}

fn seed_running_owner_funds_from_account_lines(
    ledger_index: i64,
    snapshots_by_ledger: &HashMap<i64, Vec<AccountLineSnapshot>>,
    iou_owner_funds_by_key: &mut IouOwnerFundsByKey,
) {
    let Some(rows) = snapshots_by_ledger.get(&ledger_index) else {
        return;
    };
    for row in rows {
        let entry = iou_owner_funds_by_key
            .entry((
                row.owner_account.clone(),
                row.currency.clone(),
                row.issuer.clone(),
            ))
            .or_insert(row.balance);
        if row.balance < *entry {
            *entry = row.balance;
        }
    }
}

fn running_owner_funds_for_offer(
    offer_obj: &Map<String, Value>,
    xrp_owner_funds_by_account: &XrpOwnerFundsByAccount,
    iou_owner_funds_by_key: &IouOwnerFundsByKey,
) -> Result<RunningOwnerFunds> {
    let Some(account) = offer_obj.get("Account").and_then(as_string) else {
        return Ok(RunningOwnerFunds::Unknown);
    };
    let Some(taker_gets) = offer_obj.get("TakerGets") else {
        return Ok(RunningOwnerFunds::Unknown);
    };
    if taker_gets.is_string() {
        return Ok(match xrp_owner_funds_by_account.get(&account) {
            Some(v) => RunningOwnerFunds::XrpDrops(*v),
            None => RunningOwnerFunds::Unknown,
        });
    }
    let Some(gets_obj) = taker_gets.as_object() else {
        return Err(anyhow!(
            "offer TakerGets domain invalid for account {}",
            account
        ));
    };
    let currency = gets_obj
        .get("currency")
        .and_then(as_string)
        .ok_or_else(|| anyhow!("IOU TakerGets missing currency for account {}", account))?;
    let issuer = gets_obj
        .get("issuer")
        .and_then(as_string)
        .ok_or_else(|| anyhow!("IOU TakerGets missing issuer for account {}", account))?;
    Ok(
        match iou_owner_funds_by_key.get(&(account, currency, issuer)) {
            Some(v) => RunningOwnerFunds::IouDecimal(*v),
            None => RunningOwnerFunds::Unknown,
        },
    )
}

fn apply_running_owner_funds_to_state(
    state_side: &IndexMap<String, Value>,
    xrp_owner_funds_by_account: &XrpOwnerFundsByAccount,
    iou_owner_funds_by_key: &IouOwnerFundsByKey,
) -> Result<(IndexMap<String, Value>, Vec<String>)> {
    let mut out = IndexMap::<String, Value>::new();
    let mut patched_offer_ids = Vec::<String>::new();

    for (offer_id, offer) in state_side {
        let Some(obj) = offer.as_object() else {
            out.insert(offer_id.clone(), offer.clone());
            continue;
        };
        let running_owner_funds =
            running_owner_funds_for_offer(obj, xrp_owner_funds_by_account, iou_owner_funds_by_key)?;
        let mut patched = obj.clone();
        let mut changed = false;
        match running_owner_funds {
            RunningOwnerFunds::Unknown => {}
            RunningOwnerFunds::XrpDrops(v) => {
                for key in FUNDING_FIELDS {
                    if patched.remove(key).is_some() {
                        changed = true;
                    }
                }
                if v > 0 {
                    patched.insert("owner_funds".to_string(), Value::String(v.to_string()));
                    changed = true;
                }
            }
            RunningOwnerFunds::IouDecimal(v) => {
                for key in FUNDING_FIELDS {
                    if patched.remove(key).is_some() {
                        changed = true;
                    }
                }
                if v > Decimal::ZERO {
                    patched.insert(
                        "owner_funds".to_string(),
                        Value::String(decimal_to_string(v)),
                    );
                    changed = true;
                }
            }
        }
        if changed {
            patched_offer_ids.push(offer_id.clone());
        }
        out.insert(offer_id.clone(), Value::Object(patched));
    }

    Ok((out, patched_offer_ids))
}

fn metadata_pre_state_field<'a>(node: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    if let Some(prev) = node.get("PreviousFields").and_then(|v| v.as_object()) {
        if let Some(v) = prev.get(key) {
            return Some(v);
        }
    }
    if let Some(fin) = node.get("FinalFields").and_then(|v| v.as_object()) {
        if let Some(v) = fin.get(key) {
            return Some(v);
        }
    }
    node.get("NewFields")
        .and_then(|v| v.as_object())
        .and_then(|m| m.get(key))
}

fn metadata_post_state_field<'a>(
    node_kind: &str,
    node: &'a Map<String, Value>,
    key: &str,
) -> Option<&'a Value> {
    match node_kind {
        "ModifiedNode" | "DeletedNode" => node
            .get("FinalFields")
            .and_then(|v| v.as_object())
            .and_then(|m| m.get(key)),
        "CreatedNode" => node
            .get("NewFields")
            .and_then(|v| v.as_object())
            .and_then(|m| m.get(key)),
        _ => None,
    }
}

fn extract_overlay_owner_funds_from_prebalance(tx_obj: &Value) -> HashMap<String, Value> {
    let mut out = HashMap::<String, Value>::new();
    let Some(res) = tx_obj.get("result").and_then(|v| v.as_object()) else {
        return out;
    };
    let Some(meta) = res.get("meta").and_then(|v| v.as_object()) else {
        return out;
    };
    let Some(affected) = meta.get("AffectedNodes").and_then(|v| v.as_array()) else {
        return out;
    };

    let mut account_root_pre_by_account = HashMap::<String, (i64, i64)>::new();
    let mut ripple_pre_by_key = HashMap::<(String, String, String), Decimal>::new();

    for wrapped in affected {
        let Some(nobj) = wrapped.as_object() else {
            continue;
        };
        let Some(body) = ["ModifiedNode", "DeletedNode", "CreatedNode"]
            .iter()
            .find_map(|k| nobj.get(*k).and_then(|v| v.as_object()))
        else {
            continue;
        };
        match body.get("LedgerEntryType").and_then(as_string).as_deref() {
            Some("AccountRoot") => {
                let Some(account) = metadata_pre_state_field(body, "Account").and_then(as_string)
                else {
                    continue;
                };
                let Some(bal_drops) = metadata_pre_state_field(body, "Balance").and_then(as_i64)
                else {
                    continue;
                };
                let owner_count = metadata_pre_state_field(body, "OwnerCount")
                    .and_then(as_i64)
                    .unwrap_or(0);
                account_root_pre_by_account.insert(account, (bal_drops, owner_count));
            }
            Some("RippleState") => {
                let Some(balance) =
                    metadata_pre_state_field(body, "Balance").and_then(|v| v.as_object())
                else {
                    continue;
                };
                let Some(low_limit) =
                    metadata_pre_state_field(body, "LowLimit").and_then(|v| v.as_object())
                else {
                    continue;
                };
                let Some(high_limit) =
                    metadata_pre_state_field(body, "HighLimit").and_then(|v| v.as_object())
                else {
                    continue;
                };
                let Some(currency) = balance
                    .get("currency")
                    .and_then(as_string)
                    .or_else(|| low_limit.get("currency").and_then(as_string))
                    .or_else(|| high_limit.get("currency").and_then(as_string))
                else {
                    continue;
                };
                let Some(low_acc) = low_limit.get("issuer").and_then(as_string) else {
                    continue;
                };
                let Some(high_acc) = high_limit.get("issuer").and_then(as_string) else {
                    continue;
                };
                let Some(bal_val) = balance
                    .get("value")
                    .and_then(as_string)
                    .and_then(|s| parse_decimal_str(&s))
                else {
                    continue;
                };
                if bal_val > Decimal::ZERO {
                    ripple_pre_by_key.insert((low_acc, currency, high_acc), bal_val);
                } else if bal_val < Decimal::ZERO {
                    ripple_pre_by_key.insert((high_acc, currency, low_acc), -bal_val);
                }
            }
            _ => {}
        }
    }

    for wrapped in affected {
        let Some(nobj) = wrapped.as_object() else {
            continue;
        };
        let node_kind = ["ModifiedNode", "DeletedNode"]
            .iter()
            .find(|k| nobj.contains_key(**k))
            .map(|s| *s);
        let Some(node_kind) = node_kind else {
            continue;
        };
        let Some(body) = nobj.get(node_kind).and_then(|v| v.as_object()) else {
            continue;
        };
        if body.get("LedgerEntryType").and_then(as_string).as_deref() != Some("Offer") {
            continue;
        }
        let Some(offer_id) = body.get("LedgerIndex").and_then(as_string) else {
            continue;
        };
        let Some(pre_offer) = build_offer_pre(node_kind, body) else {
            continue;
        };
        let Some(account) = pre_offer.get("Account").and_then(as_string) else {
            continue;
        };
        let Some(gets_raw) = pre_offer.get("TakerGets") else {
            continue;
        };
        let owner_funds = if gets_raw.is_string() {
            account_root_pre_by_account
                .get(&account)
                .and_then(|(bal_drops, owner_count)| {
                    let reserve_drops = XRPL_OWNER_FUNDS_RESERVE_BASE_DROPS
                        + (*owner_count).max(0) * XRPL_OWNER_FUNDS_RESERVE_INC_DROPS;
                    let spendable_drops = *bal_drops - reserve_drops;
                    if spendable_drops > 0 {
                        Some(Value::String(spendable_drops.to_string()))
                    } else {
                        None
                    }
                })
        } else {
            let Some(gets_obj) = gets_raw.as_object() else {
                continue;
            };
            let Some(currency) = gets_obj.get("currency").and_then(as_string) else {
                continue;
            };
            let Some(issuer) = gets_obj.get("issuer").and_then(as_string) else {
                continue;
            };
            ripple_pre_by_key
                .get(&(account, currency, issuer))
                .map(|bal| Value::String(decimal_to_string(*bal)))
        };
        if let Some(v) = owner_funds {
            out.insert(offer_id, v);
        }
    }
    out
}

fn apply_post_funding_state_from_metadata(
    tx_obj: &Value,
    xrp_owner_funds_by_account: &mut XrpOwnerFundsByAccount,
    iou_owner_funds_by_key: &mut IouOwnerFundsByKey,
) {
    let Some(res) = tx_obj.get("result").and_then(|v| v.as_object()) else {
        return;
    };
    let Some(meta) = res.get("meta").and_then(|v| v.as_object()) else {
        return;
    };
    let Some(affected) = meta.get("AffectedNodes").and_then(|v| v.as_array()) else {
        return;
    };

    for wrapped in affected {
        let Some(nobj) = wrapped.as_object() else {
            continue;
        };
        let Some((node_kind, body)) = ["ModifiedNode", "DeletedNode", "CreatedNode"]
            .iter()
            .find_map(|k| {
                nobj.get(*k)
                    .and_then(|v| v.as_object())
                    .map(|body| (*k, body))
            })
        else {
            continue;
        };

        match body.get("LedgerEntryType").and_then(as_string).as_deref() {
            Some("AccountRoot") => {
                let account = if node_kind == "DeletedNode" {
                    metadata_pre_state_field(body, "Account").and_then(as_string)
                } else {
                    metadata_post_state_field(node_kind, body, "Account").and_then(as_string)
                };
                let Some(account) = account else {
                    continue;
                };
                if node_kind == "DeletedNode" {
                    xrp_owner_funds_by_account.insert(account, 0);
                    continue;
                }
                let Some(balance_drops) =
                    metadata_post_state_field(node_kind, body, "Balance").and_then(as_i64)
                else {
                    continue;
                };
                let owner_count = metadata_post_state_field(node_kind, body, "OwnerCount")
                    .and_then(as_i64)
                    .unwrap_or(0)
                    .max(0);
                let reserve_drops = XRPL_OWNER_FUNDS_RESERVE_BASE_DROPS
                    + owner_count * XRPL_OWNER_FUNDS_RESERVE_INC_DROPS;
                let spendable_drops = (balance_drops - reserve_drops).max(0) as i128;
                xrp_owner_funds_by_account.insert(account, spendable_drops);
            }
            Some("RippleState") => {
                let low_limit_v = if node_kind == "DeletedNode" {
                    metadata_pre_state_field(body, "LowLimit")
                } else {
                    metadata_post_state_field(node_kind, body, "LowLimit")
                };
                let Some(low_limit) = low_limit_v.and_then(|v| v.as_object()) else {
                    continue;
                };
                let high_limit_v = if node_kind == "DeletedNode" {
                    metadata_pre_state_field(body, "HighLimit")
                } else {
                    metadata_post_state_field(node_kind, body, "HighLimit")
                };
                let Some(high_limit) = high_limit_v.and_then(|v| v.as_object()) else {
                    continue;
                };
                let balance_v = if node_kind == "DeletedNode" {
                    metadata_pre_state_field(body, "Balance")
                } else {
                    metadata_post_state_field(node_kind, body, "Balance")
                };
                let currency = balance_v
                    .and_then(|v| v.as_object())
                    .and_then(|m| m.get("currency"))
                    .and_then(as_string)
                    .or_else(|| low_limit.get("currency").and_then(as_string))
                    .or_else(|| high_limit.get("currency").and_then(as_string));
                let Some(currency) = currency else {
                    continue;
                };
                let Some(low_acc) = low_limit.get("issuer").and_then(as_string) else {
                    continue;
                };
                let Some(high_acc) = high_limit.get("issuer").and_then(as_string) else {
                    continue;
                };
                let low_key = (low_acc.clone(), currency.clone(), high_acc.clone());
                let high_key = (high_acc.clone(), currency.clone(), low_acc.clone());
                iou_owner_funds_by_key.insert(low_key.clone(), Decimal::ZERO);
                iou_owner_funds_by_key.insert(high_key.clone(), Decimal::ZERO);
                if node_kind == "DeletedNode" {
                    continue;
                }
                let Some(balance_value) = metadata_post_state_field(node_kind, body, "Balance")
                    .and_then(|v| v.as_object())
                    .and_then(|m| m.get("value"))
                    .and_then(as_string)
                    .and_then(|s| parse_decimal_str(&s))
                else {
                    continue;
                };
                if balance_value > Decimal::ZERO {
                    iou_owner_funds_by_key.insert(low_key, balance_value);
                } else if balance_value < Decimal::ZERO {
                    iou_owner_funds_by_key.insert(high_key, -balance_value);
                }
            }
            _ => {}
        }
    }
}

fn load_parquet_hash_column(path: &Path, column_name: &str) -> Result<HashSet<String>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = SerializedFileReader::new(file)
        .with_context(|| format!("open parquet {}", path.display()))?;
    let schema = reader.metadata().file_metadata().schema_descr_ptr();
    let mut col_idx: Option<usize> = None;
    for (idx, col) in schema.columns().iter().enumerate() {
        if col.name() == column_name {
            col_idx = Some(idx);
            break;
        }
    }
    let col_idx = col_idx.ok_or_else(|| {
        anyhow!(
            "parquet column not found: path={} column={}",
            path.display(),
            column_name
        )
    })?;

    let mut out = HashSet::<String>::new();
    let iter = reader
        .get_row_iter(None)
        .with_context(|| format!("row iter {}", path.display()))?;
    for rec in iter {
        let row = rec.with_context(|| format!("read row from {}", path.display()))?;
        let raw = if let Ok(s) = row.get_string(col_idx) {
            Some(s.to_string())
        } else if let Ok(b) = row.get_bytes(col_idx) {
            Some(String::from_utf8_lossy(b.data()).to_string())
        } else {
            None
        };
        if let Some(raw) = raw {
            let s = raw.trim();
            if !s.is_empty() {
                out.insert(s.to_string());
            }
        }
    }
    Ok(out)
}

fn load_parquet_hash_columns(path: &Path, column_names: &[&str]) -> Result<HashSet<String>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = SerializedFileReader::new(file)
        .with_context(|| format!("open parquet {}", path.display()))?;
    let schema = reader.metadata().file_metadata().schema_descr_ptr();
    let mut col_idx: Option<usize> = None;
    let mut chosen_name: Option<&str> = None;
    for wanted in column_names {
        for (idx, col) in schema.columns().iter().enumerate() {
            if col.name() == *wanted {
                col_idx = Some(idx);
                chosen_name = Some(*wanted);
                break;
            }
        }
        if col_idx.is_some() {
            break;
        }
    }
    let col_idx = col_idx.ok_or_else(|| {
        anyhow!(
            "parquet hash column not found: path={} candidates={:?}",
            path.display(),
            column_names
        )
    })?;
    let _chosen_name = chosen_name.expect("column name must exist when col_idx exists");

    let mut out = HashSet::<String>::new();
    let iter = reader
        .get_row_iter(None)
        .with_context(|| format!("row iter {}", path.display()))?;
    for rec in iter {
        let row = rec.with_context(|| format!("read row from {}", path.display()))?;
        let raw = if let Ok(s) = row.get_string(col_idx) {
            Some(s.to_string())
        } else if let Ok(b) = row.get_bytes(col_idx) {
            Some(String::from_utf8_lossy(b.data()).to_string())
        } else {
            None
        };
        if let Some(raw) = raw {
            let s = raw.trim();
            if !s.is_empty() {
                out.insert(s.to_uppercase());
            }
        }
    }
    if out.is_empty() {
        return Err(anyhow!(
            "parquet hash column is empty: path={} candidates={:?}",
            path.display(),
            column_names
        ));
    }
    Ok(out)
}

fn load_hashes_from_explicit_target_file(path: &Path) -> Result<HashSet<String>> {
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "txt" => {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("read {}", path.display()))?;
            let out: HashSet<String> = content
                .lines()
                .map(|line| line.trim().to_uppercase())
                .filter(|line| !line.is_empty())
                .collect();
            if out.is_empty() {
                return Err(anyhow!("target tx file is empty: {}", path.display()));
            }
            Ok(out)
        }
        "csv" => {
            let mut rdr = csv::Reader::from_path(path)
                .with_context(|| format!("open csv {}", path.display()))?;
            let headers = rdr
                .headers()
                .with_context(|| format!("read csv headers {}", path.display()))?
                .clone();
            let mut idx: Option<usize> = None;
            for wanted in ["transaction_hash", "tx_hash"] {
                if let Some(found) = headers.iter().position(|h| h == wanted) {
                    idx = Some(found);
                    break;
                }
            }
            let idx = idx.ok_or_else(|| {
                anyhow!(
                    "csv hash column not found: path={} candidates={:?}",
                    path.display(),
                    ["transaction_hash", "tx_hash"]
                )
            })?;
            let mut out = HashSet::<String>::new();
            for rec in rdr.records() {
                let rec = rec.with_context(|| format!("read csv row {}", path.display()))?;
                if let Some(raw) = rec.get(idx) {
                    let s = raw.trim();
                    if !s.is_empty() {
                        out.insert(s.to_uppercase());
                    }
                }
            }
            if out.is_empty() {
                return Err(anyhow!("csv hash column is empty: {}", path.display()));
            }
            Ok(out)
        }
        "parquet" => load_parquet_hash_columns(path, &["transaction_hash", "tx_hash"]),
        _ => Err(anyhow!(
            "unsupported target tx file format: {} (expected .csv, .parquet, or .txt)",
            path.display()
        )),
    }
}

fn load_target_tx_set(
    target_tx_file: Option<&str>,
    amm_swaps: Option<&str>,
    clob_legs: Option<&str>,
) -> Result<Option<HashSet<String>>> {
    if let Some(path) = target_tx_file {
        return load_hashes_from_explicit_target_file(Path::new(path)).map(Some);
    }
    let (Some(amm_path), Some(clob_path)) = (amm_swaps, clob_legs) else {
        return Ok(None);
    };
    let mut out = load_parquet_hash_column(Path::new(amm_path), "transaction_hash")?;
    out.extend(load_parquet_hash_column(Path::new(clob_path), "tx_hash")?);
    Ok(Some(out))
}

fn load_account_line_snapshots(
    path: Option<&str>,
) -> Result<HashMap<i64, Vec<AccountLineSnapshot>>> {
    let Some(path) = path else {
        return Ok(HashMap::new());
    };
    let f = File::open(path).with_context(|| format!("open {}", path))?;
    let mut out = HashMap::<i64, Vec<AccountLineSnapshot>>::new();
    for (line_no, line) in BufReader::new(f).lines().enumerate() {
        let line = line.with_context(|| format!("read {} line {}", path, line_no + 1))?;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let obj: Value = serde_json::from_str(s)
            .with_context(|| format!("parse {} line {}", path, line_no + 1))?;
        let owner_account = obj
            .get("owner_account")
            .and_then(as_string)
            .ok_or_else(|| anyhow!("missing owner_account in {} line {}", path, line_no + 1))?;
        let currency = obj
            .get("currency")
            .and_then(as_string)
            .ok_or_else(|| anyhow!("missing currency in {} line {}", path, line_no + 1))?;
        let issuer = obj
            .get("issuer")
            .and_then(as_string)
            .ok_or_else(|| anyhow!("missing issuer in {} line {}", path, line_no + 1))?;
        let ledger_index = obj
            .get("ledger_index")
            .and_then(as_i64)
            .ok_or_else(|| anyhow!("missing ledger_index in {} line {}", path, line_no + 1))?;
        let balance_raw = obj
            .get("balance")
            .and_then(as_string)
            .ok_or_else(|| anyhow!("missing balance in {} line {}", path, line_no + 1))?;
        let balance = parse_decimal_str(&balance_raw)
            .ok_or_else(|| anyhow!("invalid balance in {} line {}", path, line_no + 1))?;
        out.entry(ledger_index)
            .or_default()
            .push(AccountLineSnapshot {
                owner_account,
                currency,
                issuer,
                balance,
            });
    }
    Ok(out)
}

fn write_json_line<W: Write>(w: &mut W, v: &Value) -> Result<()> {
    serde_json::to_writer(&mut *w, v)?;
    w.write_all(b"\n")?;
    Ok(())
}

fn emit_snapshot<W: Write>(
    out_w: &mut W,
    state_side: &IndexMap<String, Value>,
    current_tx_pre_offers: &IndexMap<String, Value>,
    xrp_owner_funds_by_account: &XrpOwnerFundsByAccount,
    iou_owner_funds_by_key: &IouOwnerFundsByKey,
    metadata_owner_funds_by_offer_id: &HashMap<String, Value>,
    ledger_index: i64,
    tx_index: i64,
    tx_hash: &str,
    side: Side,
    used_prebook_ledger: i64,
    max_offers: usize,
) -> Result<usize> {
    let mut snapshot_state = state_side.clone();
    let mut current_tx_pre_offer_overlay_ids = Vec::<String>::new();
    for (offer_id, pre_offer) in current_tx_pre_offers {
        let existing_offer = snapshot_state.get(offer_id);
        let merged_offer = merge_offer_preserving_funding(pre_offer, existing_offer);
        snapshot_state.insert(offer_id.clone(), merged_offer);
        current_tx_pre_offer_overlay_ids.push(offer_id.clone());
    }
    let (mut merged, mut pre_overlay_offer_ids) = apply_running_owner_funds_to_state(
        &snapshot_state,
        xrp_owner_funds_by_account,
        iou_owner_funds_by_key,
    )?;
    let mut metadata_owner_funds_offer_ids = Vec::<String>::new();
    for (offer_id, owner_funds) in metadata_owner_funds_by_offer_id {
        let Some(existing_offer) = merged.get(offer_id) else {
            continue;
        };
        let Some(existing_obj) = existing_offer.as_object() else {
            continue;
        };
        let mut patched = existing_obj.clone();
        patched.insert("owner_funds".to_string(), owner_funds.clone());
        merged.insert(offer_id.clone(), Value::Object(patched));
        metadata_owner_funds_offer_ids.push(offer_id.clone());
    }
    let merged = normalize_account_owner_funds(&merged)
        .with_context(|| format!("normalize_account_owner_funds failed for tx={}", tx_hash))?;

    for offer in merged.values() {
        let offer_id = book_offer_id(offer);
        if offer
            .get("BookDirectory")
            .and_then(quality_raw_from_book_directory_value)
            .is_none()
        {
            return Err(anyhow!(
                "tx_prebook replay offer missing/invalid BookDirectory: tx={} side={} offer_id={}",
                tx_hash,
                side.as_str(),
                offer_id
            ));
        }
    }

    let mut rows: Vec<OfferRow> = merged
        .values()
        .map(|o| serialize_offer_row(o, side))
        .collect();
    // Rippled's BookTip walks each quality directory in sfIndexes order. Do not
    // tie-break equal BookDirectory qualities by offer id; that changes the
    // execution order for same-quality offers.
    rows.sort_by(|a, b| a.quality_sort_key.cmp(&b.quality_sort_key));
    if max_offers > 0 && rows.len() > max_offers {
        rows.truncate(max_offers);
    }
    let offers_json: Vec<Value> = rows.into_iter().map(|r| r.row).collect();
    pre_overlay_offer_ids.sort();
    current_tx_pre_offer_overlay_ids.sort();
    metadata_owner_funds_offer_ids.sort();

    let rec = json!({
        "scope": "tx_prebook_snapshot",
        "ledger_index": ledger_index,
        "transaction_index": tx_index,
        "transaction_hash": tx_hash,
        "side": side.as_str(),
        "used_prebook_ledger": used_prebook_ledger,
        "offers_count_visible": merged.len(),
        "offers_count_emitted": offers_json.len(),
        "pre_overlay_offer_count": pre_overlay_offer_ids.len(),
        "pre_overlay_offer_ids": pre_overlay_offer_ids,
        "current_tx_pre_offer_overlay_count": current_tx_pre_offer_overlay_ids.len(),
        "current_tx_pre_offer_overlay_ids": current_tx_pre_offer_overlay_ids,
        "metadata_owner_funds_offer_count": metadata_owner_funds_offer_ids.len(),
        "metadata_owner_funds_offer_ids": metadata_owner_funds_offer_ids,
        "offers": offers_json,
    });
    write_json_line(out_w, &rec)?;
    Ok(rec
        .get("offers_count_emitted")
        .and_then(as_i64)
        .unwrap_or(0) as usize)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let out_dir = PathBuf::from(&args.output_dir);
    std::fs::create_dir_all(&out_dir).with_context(|| format!("mkdir -p {}", out_dir.display()))?;

    let book_ledger_min = (args.ledger_start - 1).max(1);
    let book_xrp_paths = build_book_paths(
        &args.book_gets_xrp,
        args.prebook_shards_dir.as_deref(),
        "getsXRP",
        book_ledger_min,
        args.ledger_end,
    )?;
    let book_rusd_paths = build_book_paths(
        &args.book_gets_rusd,
        args.prebook_shards_dir.as_deref(),
        "getsrUSD",
        book_ledger_min,
        args.ledger_end,
    )?;

    let (iou_currency, iou_issuer) = find_iou_from_book_paths(&book_xrp_paths)?
        .or_else(|| find_iou_from_book_paths(&book_rusd_paths).ok().flatten())
        .ok_or_else(|| anyhow!("failed to detect IOU currency/issuer from prebook snapshots"))?;
    println!(
        "[cfg] IOU={} issuer={}",
        iou_currency,
        iou_issuer.clone().unwrap_or_else(|| "-".to_string())
    );
    let target_tx_set = load_target_tx_set(
        args.target_tx_file.as_deref(),
        args.amm_swaps.as_deref(),
        args.clob_legs.as_deref(),
    )?;
    let account_line_snapshots_by_ledger =
        load_account_line_snapshots(args.account_lines_snapshots.as_deref())?;
    if let Some(targets) = &target_tx_set {
        println!(
            "[cfg] emit snapshots: target window tx only | n={}",
            targets.len()
        );
    } else {
        println!("[cfg] emit snapshots: all metadata tx in range");
    }
    if !account_line_snapshots_by_ledger.is_empty() {
        println!(
            "[cfg] account line snapshots: ledgers={}",
            account_line_snapshots_by_ledger.len()
        );
    }

    let metadata_f = File::open(&args.metadata_ndjson)
        .with_context(|| format!("open {}", args.metadata_ndjson))?;
    let mut metadata_r = BufReader::new(metadata_f);

    let snapshots_path = out_dir.join(&args.snapshots_name);
    let snapshots_f = File::create(&snapshots_path)
        .with_context(|| format!("create {}", snapshots_path.display()))?;
    let mut snapshots_w = BufWriter::new(snapshots_f);

    let mut events_w: Option<BufWriter<File>> = if args.no_events {
        None
    } else {
        let ep = out_dir.join(&args.events_name);
        let f = File::create(&ep).with_context(|| format!("create {}", ep.display()))?;
        Some(BufWriter::new(f))
    };

    let mut line_no: usize = 0;
    let mut matched_tx: usize = 0;
    let mut emitted_tx: usize = 0;
    let mut emitted_rows: usize = 0;
    let mut current_ledger: Option<i64> = None;
    let mut used_prebook_ledger: Option<i64> = None;
    let mut prev_order_key: Option<(i64, i64)> = None;
    let mut expected_tx_index: Option<i64> = None;
    let mut state_getsxrp = IndexMap::<String, Value>::new();
    let mut state_getsrusd = IndexMap::<String, Value>::new();
    let mut xrp_owner_funds_by_account = XrpOwnerFundsByAccount::new();
    let mut iou_owner_funds_by_key = IouOwnerFundsByKey::new();
    let mut book_xrp_cursor = BookCursor::new(book_xrp_paths, "getsXRP")?;
    let mut book_rusd_cursor = BookCursor::new(book_rusd_paths, "getsrUSD")?;

    let start = Instant::now();
    let mut line = String::new();
    loop {
        line.clear();
        let n = metadata_r.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        line_no += 1;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let obj: Value = match serde_json::from_str(s) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some((li, txi, txh)) = extract_tx_fields(&obj) else {
            continue;
        };
        if li < args.ledger_start || li > args.ledger_end {
            continue;
        }

        let key = (li, txi);
        if let Some(prev) = prev_order_key {
            if key < prev {
                return Err(anyhow!(
                    "metadata order violation: expected non-decreasing (ledger_index, tx_index), prev={:?} curr={:?}",
                    prev,
                    key
                ));
            }
        }
        prev_order_key = Some(key);

        if current_ledger != Some(li) {
            expected_tx_index = Some(0);
        }
        if !args.allow_filtered_ledger_metadata {
            if let Some(expected) = expected_tx_index {
                if txi != expected {
                    return Err(anyhow!(
                        "metadata coverage gap within ledger: ledger={} expected_tx_index={} got={}. \
tx-level prebook replay requires full ledger metadata so earlier in-ledger offer changes are applied \
before each target tx. Pass --allow-filtered-ledger-metadata only if you explicitly accept stale pre-state risk.",
                        li,
                        expected,
                        txi
                    ));
                }
            }
        }
        expected_tx_index = Some(txi + 1);

        matched_tx += 1;
        if current_ledger != Some(li) {
            let used = li - 1;
            let ox = book_xrp_cursor.take_exact(used)?;
            let oru = book_rusd_cursor.take_exact(used)?;
            used_prebook_ledger = Some(used);
            let (sx, _) = state_from_offers(&ox, &iou_currency, iou_issuer.as_deref());
            let (_, sr) = state_from_offers(&oru, &iou_currency, iou_issuer.as_deref());
            state_getsxrp = sx;
            state_getsrusd = sr;
            xrp_owner_funds_by_account.clear();
            iou_owner_funds_by_key.clear();
            seed_running_owner_funds_from_state(
                &state_getsxrp,
                &mut xrp_owner_funds_by_account,
                &mut iou_owner_funds_by_key,
            )?;
            seed_running_owner_funds_from_state(
                &state_getsrusd,
                &mut xrp_owner_funds_by_account,
                &mut iou_owner_funds_by_key,
            )?;
            seed_running_owner_funds_from_account_lines(
                used,
                &account_line_snapshots_by_ledger,
                &mut iou_owner_funds_by_key,
            );
            current_ledger = Some(li);
        }

        let changes =
            iter_offer_changes(&obj, &iou_currency, iou_issuer.as_deref()).with_context(|| {
                format!(
                    "iter_offer_changes failed for tx={} ledger={} tx_index={}",
                    txh, li, txi
                )
            })?;
        let derived_owner_funds_by_offer_id = extract_overlay_owner_funds_from_prebalance(&obj);

        let should_emit = target_tx_set
            .as_ref()
            .map(|targets| targets.contains(&txh))
            .unwrap_or(true);
        if should_emit {
            let mut current_tx_pre_getsxrp = IndexMap::<String, Value>::new();
            let mut current_tx_pre_getsrusd = IndexMap::<String, Value>::new();
            for ch in &changes {
                let Some(pre_offer) = &ch.pre_offer else {
                    continue;
                };
                match ch.side_pre {
                    Some(Side::GetsXrp) => {
                        current_tx_pre_getsxrp.insert(ch.offer_id.clone(), pre_offer.clone());
                    }
                    Some(Side::GetsRusd) => {
                        current_tx_pre_getsrusd.insert(ch.offer_id.clone(), pre_offer.clone());
                    }
                    None => {}
                }
            }
            let used = used_prebook_ledger
                .ok_or_else(|| anyhow!("internal: used_prebook_ledger missing"))?;
            emitted_rows += emit_snapshot(
                &mut snapshots_w,
                &state_getsxrp,
                &current_tx_pre_getsxrp,
                &xrp_owner_funds_by_account,
                &iou_owner_funds_by_key,
                &derived_owner_funds_by_offer_id,
                li,
                txi,
                &txh,
                Side::GetsXrp,
                used,
                args.max_offers_per_side,
            )?;
            emitted_rows += emit_snapshot(
                &mut snapshots_w,
                &state_getsrusd,
                &current_tx_pre_getsrusd,
                &xrp_owner_funds_by_account,
                &iou_owner_funds_by_key,
                &derived_owner_funds_by_offer_id,
                li,
                txi,
                &txh,
                Side::GetsRusd,
                used,
                args.max_offers_per_side,
            )?;
            emitted_tx += 1;
        }

        apply_post_funding_state_from_metadata(
            &obj,
            &mut xrp_owner_funds_by_account,
            &mut iou_owner_funds_by_key,
        );

        for ch in &changes {
            let funding_fallback = state_getsxrp
                .get(&ch.offer_id)
                .or_else(|| state_getsrusd.get(&ch.offer_id))
                .cloned();

            if ch.side_pre == Some(Side::GetsXrp) && ch.side_post != Some(Side::GetsXrp) {
                state_getsxrp.shift_remove(&ch.offer_id);
            }
            if ch.side_pre == Some(Side::GetsRusd) && ch.side_post != Some(Side::GetsRusd) {
                state_getsrusd.shift_remove(&ch.offer_id);
            }

            if ch.side_post == Some(Side::GetsXrp) {
                if let Some(post) = &ch.post_offer {
                    let merged_post =
                        merge_offer_preserving_funding(post, funding_fallback.as_ref());
                    state_getsxrp.insert(ch.offer_id.clone(), merged_post);
                }
            } else if ch.side_post == Some(Side::GetsRusd) {
                if let Some(post) = &ch.post_offer {
                    let merged_post =
                        merge_offer_preserving_funding(post, funding_fallback.as_ref());
                    state_getsrusd.insert(ch.offer_id.clone(), merged_post);
                }
            } else if ch.side_pre.is_none() {
                state_getsxrp.shift_remove(&ch.offer_id);
                state_getsrusd.shift_remove(&ch.offer_id);
            }

            if let Some(ref mut ew) = events_w {
                let action = if ch.post_offer.is_some()
                    && matches!(ch.side_post, Some(Side::GetsXrp) | Some(Side::GetsRusd))
                {
                    "upsert"
                } else {
                    "delete"
                };
                let rec = json!({
                    "scope": "tx_prebook_event",
                    "phase": "post_apply",
                    "ledger_index": li,
                    "transaction_index": txi,
                    "transaction_hash": txh,
                    "offer_id": ch.offer_id,
                    "node_kind": ch.node_kind,
                    "side_pre": ch.side_pre.map(|s| s.as_str()),
                    "side_post": ch.side_post.map(|s| s.as_str()),
                    "action": action,
                    "step_idx": ch.step_idx,
                });
                write_json_line(ew, &rec)?;
            }
        }

        if args.progress_every > 0 && (line_no % args.progress_every == 0) {
            print_progress(line_no, matched_tx, emitted_tx, start, false);
        }
    }

    snapshots_w.flush()?;
    if let Some(ref mut ew) = events_w {
        ew.flush()?;
    }

    print_progress(line_no, matched_tx, emitted_tx, start, true);
    println!("[done] snapshots={}", snapshots_path.display());
    if !args.no_events {
        println!(
            "[done] events={}",
            out_dir.join(&args.events_name).display()
        );
    }
    println!(
        "[done] matched_tx={} emitted_tx={} emitted_rows={} lines={}",
        matched_tx, emitted_tx, emitted_rows, line_no
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_owner_funds_preserves_xrp_drop_strings() {
        let offers = IndexMap::from([
            (
                "offer-a".to_string(),
                json!({"Account":"ra","TakerGets":"1000","owner_funds":"100"}),
            ),
            (
                "offer-b".to_string(),
                json!({"Account":"ra","TakerGets":"2000"}),
            ),
        ]);

        let normalized = normalize_account_owner_funds(&offers).unwrap();
        assert_eq!(
            normalized["offer-a"]["owner_funds"],
            Value::String("100".to_string())
        );
        assert_eq!(
            normalized["offer-b"]["owner_funds"],
            Value::String("100".to_string())
        );
    }

    #[test]
    fn normalize_owner_funds_canonicalizes_xrp_decimal_strings() {
        let offers = IndexMap::from([
            (
                "offer-a".to_string(),
                json!({"Account":"ra","TakerGets":"1000","owner_funds":"1885.797642"}),
            ),
            (
                "offer-b".to_string(),
                json!({"Account":"ra","TakerGets":"2000"}),
            ),
        ]);

        let normalized = normalize_account_owner_funds(&offers).unwrap();
        assert_eq!(
            normalized["offer-a"]["owner_funds"],
            Value::String("1885797642".to_string())
        );
        assert_eq!(
            normalized["offer-b"]["owner_funds"],
            Value::String("1885797642".to_string())
        );
    }

    #[test]
    fn normalize_owner_funds_rejects_mixed_domains() {
        let offers = IndexMap::from([
            (
                "offer-a".to_string(),
                json!({"Account":"ra","TakerGets":"1000","owner_funds":"100"}),
            ),
            (
                "offer-b".to_string(),
                json!({"Account":"ra","TakerGets":{"currency":"USD","issuer":"rIssuer","value":"5"},"owner_funds":"3.5"}),
            ),
        ]);

        let err = normalize_account_owner_funds(&offers).unwrap_err();
        assert!(err.to_string().contains("mixed owner_funds domains"));
    }

    #[test]
    fn normalize_offer_prefers_book_directory_quality() {
        let offer = normalize_offer(
            &json!({
                "Account": "rhi2fe7TsQ91WpNQ83VdsfqvH6qo7Vo8e5",
                "Sequence": 96305910,
                "BookDirectory": "54123B870896F4964184D7042DB75D78913D3E377BFC4ECA4F06E16D1F80A22E",
                "TakerGets": "1290849",
                "TakerPays": {
                    "currency": "524C555344000000000000000000000000000000",
                    "issuer": "rMxCKbEDwqr76QuheSUMdEGf4B9xJ8m5De",
                    "value": "2.499998432499028"
                }
            }),
            &Map::new(),
        )
        .unwrap();
        let row = serialize_offer_row(&offer, Side::GetsXrp).row;

        assert_eq!(
            offer.get("quality").and_then(as_string).as_deref(),
            Some("0.000001936708656472622")
        );
        assert_eq!(
            row.get("book_directory").and_then(as_string).as_deref(),
            Some("54123B870896F4964184D7042DB75D78913D3E377BFC4ECA4F06E16D1F80A22E")
        );
        assert_eq!(
            row.get("quality").and_then(as_string).as_deref(),
            Some("0.000001936708656472622")
        );
        assert_eq!(
            row.get("quality_in_per_out").and_then(as_string).as_deref(),
            Some("1.9367086564726222819245318391")
        );
    }

    #[test]
    fn emit_snapshot_only_patches_existing_state_rows() {
        let mut state_side = IndexMap::new();
        state_side.insert(
            "offer-a".to_string(),
            json!({
                "index": "offer-a",
                "Account": "ra",
                "BookDirectory": "0000000000000000000000000000000000000000000000000000000000000001",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "10"
                },
                "TakerPays": "20000000"
            }),
        );
        let metadata_owner_funds_by_offer_id = HashMap::from([
            ("offer-a".to_string(), Value::String("7".to_string())),
            ("offer-missing".to_string(), Value::String("9".to_string())),
        ]);
        let mut out = Vec::<u8>::new();

        let emitted = emit_snapshot(
            &mut out,
            &state_side,
            &IndexMap::new(),
            &XrpOwnerFundsByAccount::new(),
            &IouOwnerFundsByKey::new(),
            &metadata_owner_funds_by_offer_id,
            100,
            3,
            "TX",
            Side::GetsRusd,
            99,
            0,
        )
        .unwrap();

        assert_eq!(emitted, 1);
        let rec: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(rec.get("pre_overlay_offer_count").and_then(as_i64), Some(0));
        assert_eq!(
            rec.get("pre_overlay_offer_ids")
                .and_then(|v| v.as_array())
                .map(|v| v.len()),
            Some(0)
        );
        assert_eq!(
            rec.get("metadata_owner_funds_offer_count").and_then(as_i64),
            Some(1)
        );
        assert_eq!(
            rec.get("metadata_owner_funds_offer_ids")
                .and_then(|v| v.as_array())
                .and_then(|v| v.first())
                .and_then(as_string)
                .as_deref(),
            Some("offer-a")
        );
        let offers = rec.get("offers").and_then(|v| v.as_array()).unwrap();
        assert_eq!(offers.len(), 1);
        assert_eq!(
            offers[0].get("offer_id").and_then(as_string).as_deref(),
            Some("offer-a")
        );
        assert_eq!(
            offers[0].get("owner_funds").and_then(as_string).as_deref(),
            Some("7")
        );
    }

    #[test]
    fn emit_snapshot_orders_by_book_directory_quality_before_truncating() {
        let mut state_side = IndexMap::new();
        state_side.insert(
            "offer-first".to_string(),
            json!({
                "index": "offer-first",
                "Account": "ra",
                "BookDirectory": "0000000000000000000000000000000000000000000000000000000000000002",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "10"
                },
                "TakerPays": "20000000"
            }),
        );
        state_side.insert(
            "offer-second".to_string(),
            json!({
                "index": "offer-second",
                "Account": "rb",
                "BookDirectory": "0000000000000000000000000000000000000000000000000000000000000001",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "9"
                },
                "TakerPays": "10000000"
            }),
        );
        let mut out = Vec::<u8>::new();

        emit_snapshot(
            &mut out,
            &state_side,
            &IndexMap::new(),
            &XrpOwnerFundsByAccount::new(),
            &IouOwnerFundsByKey::new(),
            &HashMap::new(),
            100,
            4,
            "TX2",
            Side::GetsRusd,
            99,
            0,
        )
        .unwrap();

        let rec: Value = serde_json::from_slice(&out).unwrap();
        let offers = rec.get("offers").and_then(|v| v.as_array()).unwrap();
        assert_eq!(
            offers
                .iter()
                .filter_map(|v| v.get("offer_id").and_then(as_string))
                .collect::<Vec<_>>(),
            vec!["offer-second".to_string(), "offer-first".to_string()]
        );
    }

    #[test]
    fn emit_snapshot_preserves_same_quality_directory_order() {
        let mut state_side = IndexMap::new();
        state_side.insert(
            "offer-z".to_string(),
            json!({
                "index": "offer-z",
                "Account": "ra",
                "BookDirectory": "0000000000000000000000000000000000000000000000000000000000000001",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "10"
                },
                "TakerPays": "20000000"
            }),
        );
        state_side.insert(
            "offer-a".to_string(),
            json!({
                "index": "offer-a",
                "Account": "rb",
                "BookDirectory": "0000000000000000000000000000000000000000000000000000000000000001",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "9"
                },
                "TakerPays": "18000000"
            }),
        );
        let mut out = Vec::<u8>::new();

        emit_snapshot(
            &mut out,
            &state_side,
            &IndexMap::new(),
            &XrpOwnerFundsByAccount::new(),
            &IouOwnerFundsByKey::new(),
            &HashMap::new(),
            100,
            4,
            "TX2",
            Side::GetsRusd,
            99,
            0,
        )
        .unwrap();

        let rec: Value = serde_json::from_slice(&out).unwrap();
        let offers = rec.get("offers").and_then(|v| v.as_array()).unwrap();
        assert_eq!(
            offers
                .iter()
                .filter_map(|v| v.get("offer_id").and_then(as_string))
                .collect::<Vec<_>>(),
            vec!["offer-z".to_string(), "offer-a".to_string()]
        );
    }

    #[test]
    fn emit_snapshot_overlays_current_tx_pre_offer_before_truncating() {
        let mut state_side = IndexMap::new();
        state_side.insert(
            "deep-state-offer".to_string(),
            json!({
                "index": "deep-state-offer",
                "Account": "ra",
                "BookDirectory": "0000000000000000000000000000000000000000000000000000000000000009",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "10"
                },
                "TakerPays": "20000000"
            }),
        );
        let mut current_tx_pre_offers = IndexMap::new();
        current_tx_pre_offers.insert(
            "real-filled-offer".to_string(),
            json!({
                "index": "real-filled-offer",
                "Account": "rb",
                "BookDirectory": "0000000000000000000000000000000000000000000000000000000000000001",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "9"
                },
                "TakerPays": "10000000"
            }),
        );
        let mut out = Vec::<u8>::new();

        emit_snapshot(
            &mut out,
            &state_side,
            &current_tx_pre_offers,
            &XrpOwnerFundsByAccount::new(),
            &IouOwnerFundsByKey::new(),
            &HashMap::new(),
            100,
            4,
            "TX4",
            Side::GetsRusd,
            99,
            1,
        )
        .unwrap();

        let rec: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            rec.get("current_tx_pre_offer_overlay_count")
                .and_then(as_i64),
            Some(1)
        );
        let offers = rec.get("offers").and_then(|v| v.as_array()).unwrap();
        assert_eq!(offers.len(), 1);
        assert_eq!(
            offers[0].get("offer_id").and_then(as_string).as_deref(),
            Some("real-filled-offer")
        );
    }

    #[test]
    fn emit_snapshot_applies_running_owner_funds_and_clears_stale_funded_fields() {
        let mut state_side = IndexMap::new();
        state_side.insert(
            "offer-a".to_string(),
            json!({
                "index": "offer-a",
                "Account": "ra",
                "BookDirectory": "0000000000000000000000000000000000000000000000000000000000000001",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "10"
                },
                "TakerPays": "20000000",
                "owner_funds": "99",
                "taker_gets_funded": {"currency": "USD", "issuer": "rIssuer", "value": "9"},
                "taker_pays_funded": "18000000"
            }),
        );
        let mut iou_owner_funds_by_key = IouOwnerFundsByKey::new();
        iou_owner_funds_by_key.insert(
            ("ra".to_string(), "USD".to_string(), "rIssuer".to_string()),
            Decimal::ZERO,
        );
        let mut out = Vec::<u8>::new();

        emit_snapshot(
            &mut out,
            &state_side,
            &IndexMap::new(),
            &XrpOwnerFundsByAccount::new(),
            &iou_owner_funds_by_key,
            &HashMap::new(),
            100,
            5,
            "TX3",
            Side::GetsRusd,
            99,
            0,
        )
        .unwrap();

        let rec: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(rec.get("pre_overlay_offer_count").and_then(as_i64), Some(1));
        let offers = rec.get("offers").and_then(|v| v.as_array()).unwrap();
        assert!(
            offers[0].get("owner_funds").is_none()
                || offers[0].get("owner_funds").is_some_and(Value::is_null)
        );
        assert!(offers[0]
            .get("taker_gets_funded")
            .and_then(|v| v.as_object())
            .is_none());
        assert!(offers[0].get("taker_pays_funded").is_none());
    }

    #[test]
    fn post_funding_state_updates_iou_prefix_balance_for_next_tx() {
        let tx_obj = json!({
            "result": {
                "meta": {
                    "AffectedNodes": [
                        {
                            "ModifiedNode": {
                                "LedgerEntryType": "RippleState",
                                "FinalFields": {
                                    "Balance": {
                                        "currency": "USD",
                                        "issuer": "rrrrrrrrrrrrrrrrrrrrBZbvji",
                                        "value": "0"
                                    },
                                    "LowLimit": {
                                        "currency": "USD",
                                        "issuer": "ra",
                                        "value": "1000"
                                    },
                                    "HighLimit": {
                                        "currency": "USD",
                                        "issuer": "rIssuer",
                                        "value": "0"
                                    }
                                },
                                "PreviousFields": {
                                    "Balance": {
                                        "currency": "USD",
                                        "issuer": "rrrrrrrrrrrrrrrrrrrrBZbvji",
                                        "value": "5"
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        });

        let mut xrp_owner_funds_by_account = XrpOwnerFundsByAccount::new();
        let mut iou_owner_funds_by_key = IouOwnerFundsByKey::new();
        iou_owner_funds_by_key.insert(
            ("ra".to_string(), "USD".to_string(), "rIssuer".to_string()),
            Decimal::from(5),
        );

        apply_post_funding_state_from_metadata(
            &tx_obj,
            &mut xrp_owner_funds_by_account,
            &mut iou_owner_funds_by_key,
        );

        let state_side = IndexMap::from([(
            "offer-a".to_string(),
            json!({
                "index": "offer-a",
                "Account": "ra",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "4"
                },
                "TakerPays": "8000000",
                "owner_funds": "5"
            }),
        )]);

        let (patched, patched_ids) = apply_running_owner_funds_to_state(
            &state_side,
            &xrp_owner_funds_by_account,
            &iou_owner_funds_by_key,
        )
        .unwrap();
        assert_eq!(patched_ids, vec!["offer-a".to_string()]);
        assert!(patched["offer-a"].get("owner_funds").is_none());
    }

    #[test]
    fn boundary_account_lines_seed_iou_owner_funds_for_later_created_offer() {
        let snapshots = HashMap::from([(
            99i64,
            vec![AccountLineSnapshot {
                owner_account: "ra".to_string(),
                currency: "USD".to_string(),
                issuer: "rIssuer".to_string(),
                balance: Decimal::new(25, 1),
            }],
        )]);
        let mut iou_owner_funds_by_key = IouOwnerFundsByKey::new();
        seed_running_owner_funds_from_account_lines(99, &snapshots, &mut iou_owner_funds_by_key);

        let state_side = IndexMap::from([(
            "offer-a".to_string(),
            json!({
                "index": "offer-a",
                "Account": "ra",
                "TakerGets": {
                    "currency": "USD",
                    "issuer": "rIssuer",
                    "value": "10"
                },
                "TakerPays": "20000000"
            }),
        )]);

        let (patched, patched_ids) = apply_running_owner_funds_to_state(
            &state_side,
            &XrpOwnerFundsByAccount::new(),
            &iou_owner_funds_by_key,
        )
        .unwrap();
        assert_eq!(patched_ids, vec!["offer-a".to_string()]);
        assert_eq!(
            patched["offer-a"]
                .get("owner_funds")
                .and_then(as_string)
                .as_deref(),
            Some("2.5")
        );
    }
}
