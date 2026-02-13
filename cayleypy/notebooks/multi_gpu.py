#%%
# ---------------------------
# GPU монитор
# ---------------------------

def start_gpu_monitor():
    try:
        nvmlInit()
        print("GPU мониторинг запущен (каждые 3 сек)")
        print("─" * 90)

        def monitor():
            out = sys.__stdout__   # прямой вывод мимо Jupyter
            while True:
                try:
                    s = ""
                    for i in range(torch.cuda.device_count()):
                        h = nvmlDeviceGetHandleByIndex(i)
                        u = nvmlDeviceGetUtilizationRates(h).gpu
                        m = nvmlDeviceGetMemoryInfo(h)
                        s += f" T4-{i}: {u:3d}% | {m.used//1048576:5d}/{m.total//1048576:5d} MB "
                    
                    out.write("\r" + s)
                    out.flush()
                    time.sleep(3)
                except:
                    break

        threading.Thread(target=monitor, daemon=True).start()

    except:
        print("NVML недоступен")


start_gpu_monitor()

#%%
import numpy as np, time, torch, torch.nn as nn
from torch.cuda.amp import autocast
import subprocess
from collections import defaultdict
import torch.nn.functional as F

# ────────────────────── ЯДРО ──────────────────────
def get_unique_states_2(states, hasher):
    h = torch.sum(states * hasher, dim=1)
    hs, idx = torch.sort(h)
    mask = torch.cat((torch.tensor([True], device=states.device), hs[1:] != hs[:-1]))
    return states[idx][mask]

def beam_search_profile(state_start, generators, scorer, beam_width, target, hasher):
    gen = torch.tensor(generators, device='cuda', dtype=torch.long)
    beam = torch.tensor(state_start, device='cuda', dtype=torch.uint8).unsqueeze(0)
   
    phase_times = {"gen_dedup": [], "forward": [], "topk": [], "misc": []}
   
    for step in range(20):
        t_total = time.time()
       
        # ФАЗА 1: Генерация кандидатов + дедупликация
        t1 = time.time()
        candidates = torch.gather(
            beam.unsqueeze(1).expand(-1, len(generators), -1),
            2, gen.unsqueeze(0).expand(beam.size(0), -1, -1)
        ).reshape(-1, beam.size(1))
        candidates = get_unique_states_2(candidates, hasher)
        t_gen = time.time() - t1
        phase_times["gen_dedup"].append(t_gen)
       
        if candidates.shape[0] == 0:
            break
        if torch.any(torch.all(candidates == target, dim=1)):
            return True, step + 1, phase_times
       
        # ФАЗА 2: Скоринг
        t2 = time.time()
        if scorer == "Hamming":
            scores = torch.sum(candidates == target, dim=1).to(torch.float32)
        else:

            with torch.no_grad():
                if scorer == "Hamming":
                    scores = torch.sum(candidates == target, dim=1).to(torch.float32)
                else:
                    # Автоматически определяем dtype модели (FP32 или FP16)
                    model_dtype = next(scorer.parameters()).dtype
                    scores = scorer(candidates.to(model_dtype)).flatten()
        
        t_fwd = time.time() - t2
        phase_times["forward"].append(t_fwd)
       
        # ФАЗА 3: Top-k отбор
        t3 = time.time()
        if candidates.shape[0] > beam_width:
            if scorer == "Hamming":
                _, idx = torch.topk(scores, k=beam_width, largest=False)  # ищем ближе к целевому
            else:
                _, idx = torch.topk(scores, k=beam_width, largest=True)   # ищем максимум оценки
            beam = candidates[idx]
        else:
            beam = candidates
        t_topk = time.time() - t3
        phase_times["topk"].append(t_topk)
       
        # ФАЗА 4: Прочее
        t_misc = time.time() - t_total - t_gen - t_fwd - t_topk
        phase_times["misc"].append(max(t_misc, 0.0))
   
    return False, 20, phase_times

# ────────────────────── ДАННЫЕ ──────────────────────
list_generators = list_generators_cube333_12gensQTM
state_size = 54
target = torch.arange(state_size, device='cuda', dtype=torch.uint8)
hasher = torch.randint(-(2**62), 2**62, (state_size,), device='cuda', dtype=torch.int64)

def random_scramble(depth=11):
    state = list(range(state_size))
    for _ in np.random.randint(0, len(list_generators), depth):
        move = np.random.randint(len(list_generators))
        state = [state[i] for i in list_generators[move]]
    return state

# ────────────────────── МОДЕЛИ (ИСПРАВЛЕНО!) ──────────────────────
class NetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(54, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 1)
        )
    def forward(self, x):
        return self.net(x)

class NetPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 2048), nn.ReLU(),
            nn.Linear(2048, 2048), nn.ReLU(),
            nn.Linear(2048, 1)
        )
    def forward(self, x):
        x = F.pad(x, (0, 10))
        return self.net(x)

def make_model(model_class, devices=None, fp16=False):
    model = model_class().cuda()
    if fp16:
        model = model.half()          
    if devices is not None:
        model = nn.DataParallel(model, device_ids=devices)
    return model.eval()

models = {
    "1. Hamming": "Hamming",
    
    # FP32
    "01. FP32 → 2 GPU + NetBase" : make_model(NetBase, [0,1], fp16=False),
    "02. FP32 → 1 GPU + NetBase (DP)" : make_model(NetBase, [0], fp16=False),
    "03. FP32 → 1 GPU + NetBase (clean)" : make_model(NetBase, None, fp16=False),
    "04. FP32 → 2 GPU + NetPad" : make_model(NetPad, [0,1], fp16=False),
    "05. FP32 → 1 GPU + NetPad (DP)" : make_model(NetPad, [0], fp16=False),
    "06. FP32 1 GPU + NetPad (clean)" : make_model(NetPad, None, fp16=False),
    
    # FP16 — теперь всё работает!
    "07. FP16 → 2 GPU + NetBase" : make_model(NetBase, [0,1], fp16=True),
    "08. FP16 → 1 GPU + NetBase (DP)" : make_model(NetBase, [0], fp16=True),
    "09. FP16 1 GPU + NetBase (clean)" : make_model(NetBase, None, fp16=True),
    "10. FP16 → 2 GPU + NetPad" : make_model(NetPad, [0,1], fp16=True),
    "11. FP16 → 1 GPU + NetPad (DP)" : make_model(NetPad, [0], fp16=True),
    "12. FP16 1 GPU + NetPad (clean)" : make_model(NetPad, None, fp16=True),
}

# ────────────────────── GPU СТАТИСТИКА ──────────────────────
def get_gpu_stats():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        mem, util = [], []
        for line in out.strip().split('\n'):
            if line:
                m, u = map(int, line.split(', '))
                mem.append(m)
                util.append(u)
        return sum(mem)/len(mem)/1024, sum(util)/len(util)
    except:
        return 0.0, 0.0

# ────────────────────── БЕНЧМАРК ──────────────────────
N_TASKS = 100
BEAM = 5_000
results = []
total_time_results = []  # отдельная таблица «режим → полное время»


def run(name, scorer):
    print(f"\n{'='*100}")
    print(f"ЗАПУСК → {name} | {N_TASKS} скрамблов (глубина ~11) | beam={BEAM:,}")
    print(f"{'='*100}")
   
    all_phases = defaultdict(list)
    peak_mems = []
    utils = []
   
    for i in range(N_TASKS):
        if i > 0 and i % 20 == 0:
            print(f" → {i}/{N_TASKS} завершено")
       
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
       
        scramble = random_scramble(11)
        mem0, util0 = get_gpu_stats()
       
        found, steps, times = beam_search_profile(
            state_start=scramble,
            generators=list_generators,
            scorer=scorer,
            beam_width=BEAM,
            target=target,
            hasher=hasher
        )
       
        mem1, util1 = get_gpu_stats()
        peak_mems.append(torch.cuda.max_memory_allocated() / 1e9)
        utils.append((util0 + util1) / 2)
       
        for phase, tlist in times.items():
            all_phases[phase].extend(tlist)
   
    avg = {p: np.mean(all_phases[p]) for p in all_phases}
    total = sum(avg.values())
   
    results.append({
        "Режим": name,
        "Генерация+дедуп": f"{avg['gen_dedup']:.4f}",
        "Forward-пасс": f"{avg['forward']:.4f}",
        "Top-k": f"{avg['topk']:.4f}",
        "Прочее": f"{avg['misc']:.4f}",
        "Всего за шаг": f"{total:.4f}",
        "Память (пик), GB": f"{np.mean(peak_mems):.2f}",
        "GPU нагрузка, %": f"{np.mean(utils):.1f}"
    })

# ────────────────────── ЗАПУСК ВСЕХ РЕЖИМОВ ──────────────────────

for name, scorer in models.items():
    overall_start = time.time()
    run(name, scorer)
    overall_time = time.time() - overall_start
    # ← ЭТО ВАЖНО: сохраняем результат замера!
    total_time_results.append({
        "Режим": name,
        "Время на 10 скрамблов": f"{overall_time:.2f} с",
        "В среднем на 1 кубик": f"{overall_time/10:.3f} с"
    })
# ────────────────────── ТАБЛИЦА ──────────────────────
import pandas as pd
df = pd.DataFrame(results)
print("\n" + "="*100)
print("ВСЕ РЕЖИМЫ — KAGGLE T4 ×2")
print("="*100)
print(df.to_string(index=False))


print("\n" + "="*100)
df_time = pd.DataFrame(total_time_results)
print(df_time.to_string(index=False))
print("\n" + "="*100)