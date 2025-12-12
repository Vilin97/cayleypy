#%%
"""Fetch all wandb summaries via API and save everything to CSV."""
import json, csv
from pathlib import Path
from tqdm import tqdm
import wandb

api = wandb.Api()
runs = api.runs("vilin97-uw/cayley_consecutive_k_cycles")


def normalize_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return json.dumps(v, separators=(",", ":"))
    except TypeError:
        return repr(v)


rows = []

for r in tqdm(runs):
    data = dict(r.summary)

    n = data.get("n")
    k = data.get("k")

    if k is None or n is None or n <= 30:
        continue

    row = {}
    for k2, v2 in data.items():
        if k2 in {"_wandb", "_step", "last_layer_list"}:
            continue
        row[k2] = normalize_value(v2)

    rows.append(row)

#%%
# Sort by n, k, generator_family, mode
rows.sort(
    key=lambda r: (
        r.get("k"),
        r.get("n"),
        r.get("generator_family"),
        r.get("mode") or r.get("central_mode"),
    )
)

all_keys = set().union(*(r.keys() for r in rows))

priority = [
    "k",
    "n",
    "generator_family",
    "mode",
    "central_mode",
    "device",
    "diameter",
    "num_layers",
    "runtime_sec",
    "peak_memory_gib",
    "peak_memory_bytes",
    "last_layer_str",
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
