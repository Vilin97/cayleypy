# %%
"""Fetch all wandb summaries via API and save everything to CSV."""
import json, csv
from pathlib import Path
from tqdm import tqdm
import wandb

api = wandb.Api()
runs_old = api.runs("vilin97-uw/cayley_consecutive_k_cycles")
runs_new = api.runs("CayleyPy/cycles")


def normalize_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, separators=(",", ":"))
    except TypeError:
        return repr(v)


def process_run(r):
    data = dict(r.summary)
    row = {}
    for k, v in data.items():
        if k in {"_wandb", "_step", "last_layer_list"}:
            continue
        row[k] = normalize_value(v)
    return row


rows = []

print(f"Fetching {len(runs_old)} old runs and {len(runs_new)} new runs...")
for r in tqdm(runs_old):
    rows.append(process_run(r))

for r in tqdm(runs_new):
    rows.append(process_run(r))

# %%
# keep only valid experiment rows
rows = [r for r in rows if "k" in r and "n" in r]

# sort by the new experiment structure
rows.sort(
    key=lambda r: (
        r.get("generator_family"),
        r.get("k"),
        r.get("n"),
        r.get("n_coincide", 0),
    )
)

all_keys = set().union(*(r.keys() for r in rows))

priority = [
    "generator_family",
    "k",
    "n",
    "n_coincide",
    "inverse_closed",
    "diameter",
    "num_layers",
    "last_layer_str",
    "device",
    "runtime_sec",
    "peak_memory_gib",
    "peak_memory_bytes",
]

fieldnames = [p for p in priority if p in all_keys] + sorted(k for k in all_keys if k not in priority)

out = Path("diameters_layers.csv")
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"Wrote {len(rows)} rows to {out}")
