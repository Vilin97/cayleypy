#%%
"""Fetch all wandb summaries via API and save everything to CSV."""
import json, csv
from pathlib import Path
from tqdm import tqdm
import wandb

api = wandb.Api()
runs = api.runs("vilin97-uw/cayley_consecutive_k_cycles")
runs_new = api.runs("CayleyPy/cycles")

def normalize_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, separators=(",", ":"))
    except TypeError:
        return repr(v)


rows = []

def process_run(r):
    data = dict(r.summary)
    row = {}
    for k2, v2 in data.items():
        if k2 in {"_wandb", "_step", "last_layer_list"}:
            continue
        row[k2] = normalize_value(v2)
    return row

print(f"Fetching {len(runs)} old runs and {len(runs_new)} new runs...")
for r in tqdm(runs):
    rows.append(process_run(r))

for r in tqdm(runs_new):
    rows.append(process_run(r))

#%%
"make a csv"

rows = [r for r in rows if 'n' in r and 'k' in r]
rows.sort(
    key=lambda r: (
        r.get("generator_family"),
        r.get("mode") or r.get("central_mode"),
        r.get("k"),
        r.get("n"),
    )
)

all_keys = set().union(*(r.keys() for r in rows))

priority = [
    "generator_family",
    "mode",
    "central_mode",
    "k",
    "n",
    "diameter",
    "num_layers",
    "last_layer_str",
    "device",
    "runtime_sec",
    "peak_memory_gib",
    "peak_memory_bytes",
]

fieldnames = [p for p in priority if p in all_keys] + sorted(
    k for k in all_keys if k not in priority
)

out = Path("diameters_layers.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows to {out}")

# %%
