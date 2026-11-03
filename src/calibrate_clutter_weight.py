import os, csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from forest_generator import generate_forest
from lidar import Lidar
from exploration import run_exploration
from strategies import frontier_weighted_strategy

SEEDS = list(range(51, 91))
WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "calibrazione_peso_frontier.csv")
COLS = ["weight", "seed", "coverage_pct", "distance_m", "steps"]

def run(weight, seed):
    forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=seed)
    gm, _, st = run_exploration(forest, Lidar(n_rays=360, r_max=4.0), (10.0, 10.0, 0.0),
                                frontier_weighted_strategy, world_width=20, world_height=20,
                                resolution=0.2, macro_cell_size=4.0, clutter_weight=weight)
    return {"weight": weight, "seed": seed, "coverage_pct": round(100 * st["cells_known"] / gm.grid.size, 2),
            "distance_m": round(st["distance"], 1), "steps": st["steps"]}

if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        done = {(float(r["weight"]), int(r["seed"])) for r in csv.DictReader(open(OUT))}
    jobs = [(w, s) for w in WEIGHTS for s in SEEDS if (w, s) not in done]
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f, ProcessPoolExecutor(max_workers=6) as pool:
        wr = csv.DictWriter(f, fieldnames=COLS)
        if new: wr.writeheader()
        futs = [pool.submit(run, w, s) for w, s in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); wr.writerow(r); f.flush()
            print(f"[{k:3d}/{len(jobs)}] peso {r['weight']:<4} seed {r['seed']} | {r['coverage_pct']:5.1f}% | {r['distance_m']:6.1f} m", flush=True)
    rows = list(csv.DictReader(open(OUT)))
    print("\npeso | bloccati (<90%) | copertura media | mediana | distanza media completati")
    for w in WEIGHTS:
        rs = [r for r in rows if float(r["weight"]) == w]
        cov = np.array([float(r["coverage_pct"]) for r in rs]); dist = np.array([float(r["distance_m"]) for r in rs])
        ok = cov >= 90
        blocked = [int(r["seed"]) for r in rs if float(r["coverage_pct"]) < 90]
        print(f"{w:4} | {len(blocked):2d} {sorted(blocked)} | {cov.mean():5.2f} | {np.median(cov):5.2f} | {dist[ok].mean():6.1f} m (n={ok.sum()})")
