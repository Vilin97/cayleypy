#%%
import networkx as nx
from cayleypy import CayleyGraph, PermutationGroups
from tqdm import tqdm
import torch
#%%
k = 7
max_diam = 30 # 33
diams = []
last_layers = []
memory_stats = []
for n in tqdm(range(k,max_diam + 1), desc="Computing diameters"):
    torch.cuda.reset_peak_memory_stats()
    central = [0]*(n//2) + [1]*(n - n//2)
    defn = PermutationGroups.consecutive_k_cycles(n, k).with_central_state(central)
    graph = CayleyGraph(defn)
    result = graph.bfs(return_all_edges=False, return_all_hashes=False)
    diams.append(result.diameter())
    last_layers.append(result.last_layer())
    print(f"n={n}, diameter:{result.diameter()}, layer sizes:{result.layer_sizes}\n")

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"Peak GPU memory: {peak_bytes / 1024**3:.3f} GiB")
    memory_stats.append(peak_bytes)

#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.plot(range(k, max_diam + 1), memory_stats, marker='o', linewidth=2)
plt.title(f"Peak GPU Memory vs n for consecutive_k_cycles (k={k}) with c.s.", fontsize=14, pad=12)
plt.yscale("log")
plt.xlabel("n", fontsize=12)
plt.ylabel("Peak GPU Memory (bytes)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

#%%
for n, layer in enumerate(last_layers):
    print(f"n={n+k}:\n{layer}\n")

#%%
import matplotlib.pyplot as plt

start_n = 5
ns = list(range(start_n, start_n + len(diams)))

plt.figure(figsize=(8,5))
plt.plot(ns, diams, marker='o', linewidth=2)
plt.title(f"Diameter vs n for consecutive_k_cycles (k={k}) with c.s.", fontsize=14, pad=12)
plt.xlabel("n", fontsize=12)
plt.ylabel("Diameter", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(ns)

for x, y in zip(ns, diams):
    plt.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=10)

plt.tight_layout()
plt.show()

#%%
diams
# %%
