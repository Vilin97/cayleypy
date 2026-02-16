#%%
"""Fetch all wandb summaries via API and save everything to CSV.

Caches processed run data to .wandb_cache.json so only new runs are fetched
on subsequent invocations.
"""
import json, csv, re, os, tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import wandb

CACHE_PATH = Path(__file__).with_name(".wandb_cache.json")
PROJECTS = ["vilin97-uw/cayley_consecutive_k_cycles", "CayleyPy/cycles"]

PARAM_KEYS = ["k", "n", "generator_family", "device", "coset", "central_mode", "inverse_closed", "hardware_name", "num_gpus"]
RESULT_KEYS = ["diameter", "num_layers", "layer_sizes", "runtime_sec", "peak_memory_gib", "peak_memory_bytes", "peak_memory_total_bytes", "peak_memory_total_gib", "peak_per_gpu_gib", "last_layer_str"]


def normalize_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, separators=(",", ":"))
    except TypeError:
        return repr(v)


def parse_hardware_from_log(r):
    """Try to extract hardware name from the run's output.log."""
    try:
        tmpdir = tempfile.mkdtemp()
        r.file("output.log").download(replace=True, root=tmpdir)
        with open(os.path.join(tmpdir, "output.log")) as f:
            for line in f:
                m = re.search(r"Hardware: (?:GPU|CPU) = (.+)", line)
                if m:
                    return m.group(1).strip()
    except Exception:
        pass
    return None


def process_run(r):
    row = {}

    # Parameters from config (canonical source)
    cfg = dict(r.config)
    for key in PARAM_KEYS:
        if key in cfg:
            row[key] = normalize_value(cfg[key])

    # Results from summary
    summ = dict(r.summary)
    for key in RESULT_KEYS:
        if key in summ:
            row[key] = normalize_value(summ[key])

    # Fall back to summary for params missing from config (old runs)
    for key in PARAM_KEYS:
        if key not in row and key in summ:
            row[key] = normalize_value(summ[key])

    # Parse hardware name from output.log if not in config
    if "hardware_name" not in row:
        hw = parse_hardware_from_log(r)
        if hw:
            row["hardware_name"] = hw

    # Provenance
    row["project"] = r.project
    row["entity"] = r.entity
    row["run_id"] = r.id
    row["run_name"] = r.name
    row["created_at"] = str(r.created_at)

    return row


def load_cache():
    if CACHE_PATH.exists():
        with CACHE_PATH.open() as f:
            data = json.load(f)
        data.setdefault("skipped_run_ids", [])
        data.setdefault("backfilled_run_ids", [])
        return data
    return {"rows": [], "seen_run_ids": [], "skipped_run_ids": [], "backfilled_run_ids": []}


def save_cache(rows, seen_run_ids, skipped_run_ids, backfilled_run_ids):
    tmp = CACHE_PATH.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump({
            "rows": rows,
            "seen_run_ids": sorted(seen_run_ids),
            "skipped_run_ids": sorted(skipped_run_ids),
            "backfilled_run_ids": sorted(backfilled_run_ids),
        }, f)
    tmp.rename(CACHE_PATH)


def fetch_rows():
    cache = load_cache()
    rows = list(cache["rows"])
    seen = set(cache["seen_run_ids"])
    skipped_ids = set(cache["skipped_run_ids"])
    backfilled_ids = set(cache["backfilled_run_ids"])
    dirty = False

    api = wandb.Api(timeout=30)
    for project in PROJECTS:
        all_runs = api.runs(project)
        new_runs = [r for r in all_runs if r.id not in seen and r.id not in skipped_ids]
        if not new_runs:
            print(f"{project}: all {len(all_runs)} runs cached")
            continue
        print(f"{project}: fetching {len(new_runs)} new runs ({len(all_runs) - len(new_runs)} cached)")
        n_skipped = 0
        for r in tqdm(new_runs):
            row = process_run(r)
            if row.get("diameter") is None:
                n_skipped += 1
                skipped_ids.add(r.id)
                continue  # failed or still running — don't cache
            rows.append(row)
            seen.add(r.id)
        if n_skipped:
            print(f"  skipped {n_skipped} runs without diameter (failed/running)")
        dirty = True

    # Backfill hardware_name, num_gpus, peak_per_gpu_gib for cached rows not yet attempted
    need_backfill = [r for r in rows if r["run_id"] not in backfilled_ids]
    if need_backfill:
        print(f"Backfilling {len(need_backfill)} rows...")
        run_id_to_row = {r["run_id"]: r for r in need_backfill}
        for project in PROJECTS:
            for r in tqdm(api.runs(project), desc=f"Backfilling {project}"):
                if r.id not in run_id_to_row:
                    continue
                cached = run_id_to_row[r.id]
                if not cached.get("hardware_name"):
                    hw = parse_hardware_from_log(r)
                    if hw:
                        cached["hardware_name"] = hw
                summ = dict(r.summary)
                if cached.get("num_gpus") is None and "num_gpus" in summ:
                    cached["num_gpus"] = summ["num_gpus"]
                if cached.get("peak_per_gpu_gib") is None and "peak_per_gpu_gib" in summ:
                    cached["peak_per_gpu_gib"] = summ["peak_per_gpu_gib"]
                if cached.get("peak_memory_total_bytes") is None and "peak_memory_total_bytes" in summ:
                    cached["peak_memory_total_bytes"] = summ["peak_memory_total_bytes"]
                backfilled_ids.add(r.id)
        dirty = True

    if dirty:
        save_cache(rows, seen, skipped_ids, backfilled_ids)
    return rows


def format_runtime(seconds):
    """Format seconds as DD:HH:MM:SS."""
    if seconds is None:
        return ""
    s = int(seconds)
    days, s = divmod(s, 86400)
    hours, s = divmod(s, 3600)
    minutes, s = divmod(s, 60)
    return f"{days:02d}:{hours:02d}:{minutes:02d}:{s:02d}"


def format_timestamp(ts):
    """Format timestamp as YYYY-MM-DD HH:MM:SS."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return ts


def format_memory_gb(peak_bytes):
    """Format bytes as GB string with 3 decimals."""
    if peak_bytes is None:
        return ""
    return f"{peak_bytes / 1e9:.3f}"


def sort_key(r):
    return (
        str(r.get("coset") or ""),
        str(r.get("generator_family") or ""),
        str(r.get("central_mode") or ""),
        str(r.get("inverse_closed") or ""),
        r.get("k") or 0,
        r.get("n") or 0,
    )


def format_row(r):
    """Build a row dict with the desired columns and formatting."""
    out = {}
    out["coset"] = r.get("coset", "")
    out["generator_family"] = r.get("generator_family", "")
    out["central_mode"] = r.get("central_mode", "")
    out["inverse_closed"] = r.get("inverse_closed", "")
    out["k"] = r.get("k", "")
    out["n"] = r.get("n", "")
    out["diameter"] = r.get("diameter", "")
    out["device"] = r.get("device", "")
    out["hardware_name"] = r.get("hardware_name", "")
    out["runtime"] = format_runtime(r.get("runtime_sec"))
    out["peak_memory_gb"] = format_memory_gb(r.get("peak_memory_bytes"))
    out["peak_memory_total_gb"] = format_memory_gb(r.get("peak_memory_total_bytes"))
    out["timestamp"] = format_timestamp(r.get("created_at"))
    out["layer_sizes"] = r.get("layer_sizes", "")
    out["last_layer_str"] = r.get("last_layer_str", "")

    # Everything else
    skip = {"coset", "generator_family", "central_mode", "inverse_closed", "k", "n",
            "diameter", "device", "hardware_name", "runtime_sec", "peak_memory_bytes",
            "peak_memory_gib", "peak_memory_total_bytes", "peak_memory_total_gib",
            "peak_per_gpu_gib", "created_at", "layer_sizes", "last_layer_str"}
    for key, val in r.items():
        if key not in skip:
            out[key] = val
    return out


rows = fetch_rows()

#%%
"make a csv"

rows = [r for r in rows if 'n' in r and 'k' in r]
rows.sort(key=sort_key)

formatted = [format_row(r) for r in rows]

priority = [
    "coset",
    "generator_family",
    "central_mode",
    "inverse_closed",
    "k",
    "n",
    "diameter",
    "device",
    "hardware_name",
    "runtime",
    "peak_memory_gb",
    "peak_memory_total_gb",
    "timestamp",
    "layer_sizes",
    "last_layer_str",
]

all_keys = set().union(*(r.keys() for r in formatted))
fieldnames = [p for p in priority if p in all_keys] + sorted(
    k for k in all_keys if k not in priority
)

out = Path("diameters_layers.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in formatted:
        w.writerow(r)

print(f"Wrote {len(formatted)} rows to {out}")

# Detect conflicting duplicates
by_key = defaultdict(list)
for r in rows:
    key = (
        r.get("generator_family"),
        r.get("coset"),
        r.get("central_mode"),
        r.get("inverse_closed"),
        r.get("k"),
        r.get("n"),
    )
    by_key[key].append(r)

conflicts = {k: v for k, v in by_key.items() if len({x.get("diameter") for x in v}) > 1}
if conflicts:
    print(f"\nWARNING: {len(conflicts)} parameter combinations have conflicting diameters:")
    for key, vals in list(conflicts.items())[:20]:
        print(f"  {key}")
        for v in vals:
            print(f"    diameter={v.get('diameter')}, run_id={v.get('run_id')}, created_at={v.get('created_at')}")
else:
    print("\nNo conflicting diameters found.")

# %%
