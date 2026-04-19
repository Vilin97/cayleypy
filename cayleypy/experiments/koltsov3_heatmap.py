"""Pull koltsov3 runs from wandb and plot heatmaps of last_layer_count and diameter."""

import numpy as np
import matplotlib.pyplot as plt
import wandb


def fetch_koltsov3_data():
    api = wandb.Api()
    runs = api.runs("CayleyPy/cycles", filters={"config.generator_family": "koltsov3"})
    print(f"Fetched {len(runs)} runs")

    records = []
    for r in runs:
        s = r.summary
        if "n" in s and "L" in s and "diameter" in s and "last_layer_count" in s:
            records.append({
                "n": s["n"],
                "L": s["L"],
                "diameter": s["diameter"],
                "last_layer_count": s["last_layer_count"],
            })
    return records


def build_heatmap_array(records, value_key):
    ns = sorted(set(r["n"] for r in records))
    ls = sorted(set(r["L"] for r in records))
    n_to_idx = {n: i for i, n in enumerate(ns)}
    l_to_idx = {l: i for i, l in enumerate(ls)}

    data = np.full((len(ls), len(ns)), np.nan)
    for r in records:
        data[l_to_idx[r["L"]], n_to_idx[r["n"]]] = r[value_key]

    return data, ns, ls


def plot_heatmap(data, ns, ls, value_key, title, filename):
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_key, fontsize=12)

    # Tick labels
    ax.set_xticks(range(len(ns)))
    ax.set_xticklabels(ns, fontsize=8)
    ax.set_yticks(range(len(ls)))
    ax.set_yticklabels(ls, fontsize=8)
    ax.set_xlabel("n", fontsize=14)
    ax.set_ylabel("L", fontsize=14)
    ax.set_title(title, fontsize=16)

    # Annotate cells with values
    for i in range(len(ls)):
        for j in range(len(ns)):
            val = data[i, j]
            if not np.isnan(val):
                text = str(int(val))
                color = "white" if val < (np.nanmax(data) + np.nanmin(data)) / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", fontsize=5, color=color)

    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    print(f"Saved {filename}")
    plt.close(fig)


def main():
    records = fetch_koltsov3_data()
    print(f"Valid records: {len(records)}")

    for value_key, title, filename in [
        ("last_layer_count", "Koltsov3: Last layer count (k=1, perm_type=2)", "koltsov3_last_layer_count.png"),
        ("diameter", "Koltsov3: Diameter (k=1, perm_type=2)", "koltsov3_diameter.png"),
    ]:
        data, ns, ls = build_heatmap_array(records, value_key)
        plot_heatmap(data, ns, ls, value_key, title, filename)


if __name__ == "__main__":
    main()
