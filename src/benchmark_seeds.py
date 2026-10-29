
import os
import csv
import json
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from shapely.geometry import Point
from forest_generator import generate_forest
from lidar import Lidar
from exploration import run_exploration
from strategies import random_strategy, frontier_strategy, serpentine_strategy

SEEDS = list(range(1, 51))
N_WORKERS = 6   # run eseguiti in parallelo (1 = in sequenza)

STRATEGIES = {
    "random": random_strategy,
    "frontier": frontier_strategy,
    "serpentine": serpentine_strategy,
}

STRATEGIES_TO_RUN = ["random", "frontier", "serpentine"]   # sottoinsieme di STRATEGIES da eseguire
MACRO_CELL_SIZE = 4.0                # lato delle macro-celle della serpentina (m)

START_POSITIONS = {
    "center": (10.0, 10.0, 0.0),
    "corner": (0.1, 0.1, 0.0),   # centro della cella nell'angolo in basso a sinistra
}
START = "corner"                     # "center" | "corner"
START_POSITION = START_POSITIONS[START]

WORLD_WIDTH = 20.0
WORLD_HEIGHT = 20.0
RESOLUTION = 0.2

# i test gia' eseguiti mantengono i loro nomi di file: la dimensione delle macro-celle compare
# solo se diversa dal test originale da 1 m, il punto di partenza solo se diverso dal centro
name_parts = ["benchmark"]
if STRATEGIES_TO_RUN != list(STRATEGIES):
    name_parts.append("_".join(STRATEGIES_TO_RUN))
if not (STRATEGIES_TO_RUN == list(STRATEGIES) and MACRO_CELL_SIZE == 1.0):
    name_parts.append(f"macro{MACRO_CELL_SIZE:g}")
if START != "center":
    name_parts.append(START)
name_parts.append("50seeds")
OUTPUT_NAME = "_".join(name_parts)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CSV_PATH = os.path.join(RESULTS_DIR, f"{OUTPUT_NAME}.csv")
COVERAGE_PATH = os.path.join(RESULTS_DIR, f"{OUTPUT_NAME}_coverage.jsonl")
SUMMARY_PATH = os.path.join(RESULTS_DIR, f"{OUTPUT_NAME}_summary.csv")

CSV_COLUMNS = ["seed", "strategy", "coverage_pct", "distance_m", "steps", "moves", "time_ms"]
SUMMARY_COLUMNS = ["strategy", "n_runs", "coverage_mean", "coverage_median",
                   "distance_mean_m", "time_mean_ms"]


def load_done_runs():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline="") as f:
        return {(int(row["seed"]), row["strategy"]) for row in csv.DictReader(f)}


def run_single(seed, strategy_name):
    # il bosco va rigenerato prima di ogni run: generate_forest fissa anche il seed
    # del generatore casuale usato dalle strategie, ed e' questo a rendere il run riproducibile.
    # start_point lascia libera da ostacoli l'area attorno al punto di partenza del robot
    forest = generate_forest(20, 20, n_trees=15, n_rocks=3, seed=seed,
                             start_point=Point(START_POSITION[0], START_POSITION[1]))
    sensor = Lidar(n_rays=360, r_max=4.0)

    t0 = time.perf_counter()
    global_map, _, stats = run_exploration(
        forest, sensor, START_POSITION, STRATEGIES[strategy_name],
        world_width=WORLD_WIDTH, world_height=WORLD_HEIGHT, resolution=RESOLUTION,
        macro_cell_size=MACRO_CELL_SIZE
    )
    time_ms = (time.perf_counter() - t0) * 1000

    total_cells = global_map.n_rows * global_map.n_cols
    row = {
        "seed": seed,
        "strategy": strategy_name,
        "coverage_pct": round(100 * stats["cells_known"] / total_cells, 2),
        "distance_m": round(stats["distance"], 1),
        "steps": stats["steps"],
        "moves": len(stats["trajectory"]) - 1,
        "time_ms": round(time_ms),
    }
    return row, stats["coverage_history"]


def write_summary():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    summary = []
    for name in STRATEGIES_TO_RUN:
        runs = [r for r in rows if r["strategy"] == name]
        if not runs:
            continue
        coverage = np.array([float(r["coverage_pct"]) for r in runs])
        summary.append({
            "strategy": name,
            "n_runs": len(runs),
            "coverage_mean": round(float(coverage.mean()), 2),
            "coverage_median": round(float(np.median(coverage)), 2),
            "distance_mean_m": round(float(np.mean([float(r["distance_m"]) for r in runs])), 1),
            "time_mean_ms": round(float(np.mean([float(r["time_ms"]) for r in runs]))),
        })

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary)

    print("\nRiepilogo per strategia:")
    for s in summary:
        print(f"  {s['strategy']:10s} | run {s['n_runs']:3d} | copertura media {s['coverage_mean']:5.1f}% "
              f"mediana {s['coverage_median']:5.1f}% | distanza media {s['distance_mean_m']:6.1f} m | "
              f"tempo medio {s['time_mean_ms']} ms")


def sort_csv():
    # i run in parallelo finiscono in ordine sparso: riordino per seed e strategia
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    order = STRATEGIES_TO_RUN
    rows.sort(key=lambda r: (int(r["seed"]), order.index(r["strategy"])))
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    done = load_done_runs()
    new_file = not os.path.exists(CSV_PATH)
    pending = [(seed, name) for seed in SEEDS for name in STRATEGIES_TO_RUN if (seed, name) not in done]
    total = len(SEEDS) * len(STRATEGIES_TO_RUN)
    count = total - len(pending)

    with open(CSV_PATH, "a", newline="") as csv_file, open(COVERAGE_PATH, "a") as coverage_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()

        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = [pool.submit(run_single, seed, name) for seed, name in pending]
            for future in as_completed(futures):
                row, coverage_history = future.result()
                count += 1

                writer.writerow(row)
                csv_file.flush()
                coverage_file.write(json.dumps({"seed": row["seed"], "strategy": row["strategy"],
                                                "coverage_history": coverage_history}) + "\n")
                coverage_file.flush()

                print(f"[{count:3d}/{total}] seed {row['seed']:2d} | {row['strategy']:10s} | "
                      f"copertura {row['coverage_pct']:5.1f}% | distanza {row['distance_m']:6.1f} m | "
                      f"tempo {row['time_ms']} ms", flush=True)

    sort_csv()
    write_summary()


if __name__ == "__main__":
    main()
