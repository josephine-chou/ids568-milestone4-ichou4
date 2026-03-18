"""
benchmark.py
Tests pipeline performance across different worker counts.
Produces runtime vs. worker count visualization.
"""

import subprocess
import json
import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_DIR = "data/raw_10m/"
WORKER_COUNTS = [1, 2, 4]
results = []

for n in WORKER_COUNTS:
    print(f"\n=== Testing {n} worker(s) ===")
    out_dir = f"data/bench_output/workers_{n}"
    metrics_dir = f"metrics_bench/workers_{n}"
    os.makedirs(metrics_dir, exist_ok=True)

    start = time.time()
    subprocess.run([
        "python", "pipeline.py",
        "--input", INPUT_DIR,
        "--output", out_dir,
        "--metrics", metrics_dir,
        "--workers", str(n),
        "--mode", "both"
    ], check=True)
    elapsed = time.time() - start

    with open(f"{metrics_dir}/metrics.json") as f:
        m = json.load(f)

    results.append({
        "workers": n,
        "total_time_s": m["distributed"]["total_time_s"],
        "transform_time_s": m["distributed"]["transform_time_s"],
    })
    print(f"Workers={n}: {m['distributed']['total_time_s']:.2f}s")

# Save results
os.makedirs("charts", exist_ok=True)
with open("charts/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot runtime vs worker count
workers = [r["workers"] for r in results]
times   = [r["total_time_s"] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(workers, times, marker="o", color="darkorange", linewidth=2)
for w, t in zip(workers, times):
    plt.annotate(f"{t:.1f}s", (w, t), textcoords="offset points", xytext=(0, 10))
plt.xlabel("Number of Workers")
plt.ylabel("Total Runtime (seconds)")
plt.title("PySpark Runtime vs. Worker Count (10M rows)")
plt.xticks(workers)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("charts/runtime_vs_workers.png", dpi=150)
print("\nChart saved: charts/runtime_vs_workers.png")
print(json.dumps(results, indent=2))