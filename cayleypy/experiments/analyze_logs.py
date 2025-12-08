#%%
"""Download only files needed for analysis."""
import wandb
from tqdm import tqdm

api = wandb.Api()
runs = api.runs("vilin97-uw/cayley_consecutive_k_cycles")

NEEDED = {
    "wandb-metadata.json",
    "wandb-summary.json",
    "output.log",
}

for r in tqdm(runs):
    for f in r.files():
        if f.name in NEEDED:
            f.download(root=f"wandb_logs/{r.name}", replace=True)

#%%
"""Extract diameters and layer sizes from logs and save to CSV."""
import json, csv
from pathlib import Path

# run this from cayleypy/cayleypy/experiments
root = Path.cwd() / "wandb_logs"

def get_arg(args, flag, cast=str, default=None):
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            try:
                return cast(args[i + 1])
            except Exception:
                return default
    return default

rows_by_key = {}

for d in tqdm(root.iterdir()):
    if not d.is_dir():
        print(f"Skipping non-directory {d}")
        continue

    meta_path = d / "wandb-metadata.json"
    summary_path = d / "wandb-summary.json"
    if not (meta_path.exists() and summary_path.exists()):
        print(f"Skipping {d} because metadata or summary is missing")
        continue

    meta = json.loads(meta_path.read_text())
    args = meta.get("args", [])

    k = get_arg(args, "--k", int)
    n = get_arg(args, "--n", int)
    mode = get_arg(args, "--mode", str, default="consecutive")

    if k is None or n is None or n <= 30:
        print(f"Skipping {d} because k={k} is missing, and n={n} is missing or too small")
        continue

    data = json.loads(summary_path.read_text())
    diameter = data.get("diameter")
    layer_sizes = data.get("layer_sizes")
    peak_mem_gib = data.get("peak_memory_gib")
    runtime_sec = data.get("runtime_sec")

    # check for CUDA OOM in output.log
    cuda_oom = False
    log_path = d / "output.log"
    if log_path.exists():
        try:
            log_txt = log_path.read_text(errors="ignore")
            if "CUDA out of memory" in log_txt:
                cuda_oom = True
        except Exception:
            pass

    row = [k, n, mode, diameter, peak_mem_gib, runtime_sec, cuda_oom, layer_sizes]
    key = (k, n, mode)

    existing = rows_by_key.get(key)
    # prefer non-OOM; if both OOM or both non-OOM, keep the first
    if existing is None or (existing[6] and not cuda_oom):
        rows_by_key[key] = row

rows = list(rows_by_key.values())
rows.sort(key=lambda r: (r[1], r[0]))  # sort by n, then k

out = Path("diameters_layers.csv")
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k", "n", "mode", "diameter", "peak_memory_gib", "runtime_sec", "cuda_oom", "layer_sizes"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {out}")

# %%
