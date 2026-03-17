"""
Features engineered:
  - amount_log        : log-transform of transaction amount (reduces skew)
  - amount_zscore     : z-score normalization of amount
  - amount_per_tenure : amount divided by tenure_days (derived ratio feature)
  - high_value_flag   : binary flag for transactions > 95th percentile
  - txn_velocity      : num_past_txns / tenure_days (transaction rate)
  - category_encoded  : one-hot encoding of transaction category
"""

import argparse
import logging
import os
import time
import traceback
import psutil

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# LOCAL (pandas) EXECUTION

def run_local(input_dir: str, output_dir: str) -> dict:
    """
    Run feature engineering using pandas (single-machine baseline).
    Returns a dict of performance metrics.
    """
    logger.info("=== LOCAL EXECUTION (pandas) ===")
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    process = psutil.Process()

    # Load all parquet files 
    logger.info("Loading data from %s", input_dir)
    parquet_files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir) if f.endswith(".parquet")
    ])
    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    load_time = time.time() - start_time
    logger.info("Loaded %d rows in %.2fs", len(df), load_time)

    # Feature Engineering
    transform_start = time.time()

    # 1. Log-transform amount (handles right-skewed distribution)
    df["amount_log"] = np.log1p(df["amount"])

    # 2. Z-score normalization of amount
    mean_amt = df["amount"].mean()
    std_amt  = df["amount"].std()
    df["amount_zscore"] = (df["amount"] - mean_amt) / std_amt

    # 3. Derived ratio: amount per day of account tenure
    df["amount_per_tenure"] = df["amount"] / (df["tenure_days"] + 1)  # +1 avoids divide-by-zero

    # 4. High-value flag (top 5%)
    threshold = df["amount"].quantile(0.95)
    df["high_value_flag"] = (df["amount"] > threshold).astype(int)

    # 5. Transaction velocity
    df["txn_velocity"] = df["num_past_txns"] / (df["tenure_days"] + 1)

    # 6. One-hot encoding for category
    category_dummies = pd.get_dummies(df["category"], prefix="cat")
    df = pd.concat([df, category_dummies], axis=1)
    df.drop(columns=["category"], inplace=True)

    transform_time = time.time() - transform_start

    # Save output
    save_start = time.time()
    out_path = os.path.join(output_dir, "features_local.parquet")
    df.to_parquet(out_path, index=False)
    save_time = time.time() - save_start

    total_time = time.time() - start_time
    peak_mem_gb = process.memory_info().rss / 1e9

    metrics = {
        "mode":            "local",
        "total_rows":      len(df),
        "load_time_s":     round(load_time, 3),
        "transform_time_s": round(transform_time, 3),
        "save_time_s":     round(save_time, 3),
        "total_time_s":    round(total_time, 3),
        "peak_memory_gb":  round(peak_mem_gb, 3),
        "partitions":      1,
        "shuffle_volume_mb": 0,  # no shuffle in pandas
        "output_path":     out_path,
    }

    logger.info("LOCAL done: %.2fs total | %.2f GB peak memory", total_time, peak_mem_gb)
    return metrics


# DISTRIBUTED (PySpark) EXECUTION

def run_distributed(input_dir: str, output_dir: str, n_workers: int = 4) -> dict:
    """
    Run feature engineering using PySpark in local mode (simulates distributed workers).
    Returns a dict of performance metrics.
    """
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F

    logger.info("=== DISTRIBUTED EXECUTION (PySpark local[%d]) ===", n_workers)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Spark
    spark = (
        SparkSession.builder
        .appName("M4_FeatureEngineering")
        .master(f"local[{n_workers}]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", str(n_workers * 2))
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    start_time = time.time()
    process = psutil.Process()

    # Load data
    logger.info("Loading data from %s", input_dir)
    df = spark.read.parquet(input_dir)
    df = df.repartition(n_workers * 4)  # optimize partition count
    row_count = df.count()
    load_time = time.time() - start_time
    logger.info("Loaded %d rows in %.2fs", row_count, load_time)

    # Feature Engineering
    transform_start = time.time()

    # 1. Log-transform amount
    df = df.withColumn("amount_log", F.log1p(F.col("amount")))

    # 2. Z-score normalization (requires computing mean & std across cluster)
    from pyspark.sql.functions import mean as spark_mean, stddev as spark_std
    stats = df.select(
        spark_mean("amount").alias("mean_amt"),
        spark_std("amount").alias("std_amt")
    ).collect()[0]
    df = df.withColumn(
        "amount_zscore",
        (F.col("amount") - stats["mean_amt"]) / stats["std_amt"]
    )

    # 3. Derived ratio
    df = df.withColumn(
        "amount_per_tenure",
        F.col("amount") / (F.col("tenure_days") + 1)
    )

    # 4. High-value flag (95th percentile via approxQuantile)
    threshold = df.approxQuantile("amount", [0.95], 0.001)[0]
    df = df.withColumn(
        "high_value_flag",
        (F.col("amount") > threshold).cast("int")
    )

    # 5. Transaction velocity
    df = df.withColumn(
        "txn_velocity",
        F.col("num_past_txns") / (F.col("tenure_days") + 1)
    )

    # 6. One-hot encoding for category
    categories = ["entertainment", "food", "health", "shopping", "travel"]
    for cat in categories:
        df = df.withColumn(
            f"cat_{cat}",
            (F.col("category") == cat).cast("int")
        )
    df = df.drop("category")

    transform_time = time.time() - transform_start

    # Save output
    save_start = time.time()
    out_path = os.path.join(output_dir, "features_distributed")
    df.write.mode("overwrite").parquet(out_path)
    save_time = time.time() - save_start

    total_time = time.time() - start_time
    peak_mem_gb = process.memory_info().rss / 1e9

    # Collect Spark metrics
    sc = spark.sparkContext
    status = sc.statusTracker()
    num_partitions = df.rdd.getNumPartitions()

    metrics = {
        "mode":             "distributed",
        "total_rows":       row_count,
        "load_time_s":      round(load_time, 3),
        "transform_time_s": round(transform_time, 3),
        "save_time_s":      round(save_time, 3),
        "total_time_s":     round(total_time, 3),
        "peak_memory_gb":   round(peak_mem_gb, 3),
        "partitions":       num_partitions,
        "n_workers":        n_workers,
        "shuffle_partitions": n_workers * 2,
        "output_path":      out_path,
    }

    spark.stop()
    logger.info("DISTRIBUTED done: %.2fs total | %.2f GB peak memory", total_time, peak_mem_gb)
    return metrics


# COMPARISON & VISUALIZATION

def save_comparison(local_metrics: dict, dist_metrics: dict, output_dir: str) -> None:
    """Save a JSON comparison and generate a performance chart."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Save metrics JSON
    comparison = {"local": local_metrics, "distributed": dist_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Metrics saved to %s/metrics.json", output_dir)

    # Bar chart: runtime breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Runtime breakdown
    phases = ["Load", "Transform", "Save", "Total"]
    local_times = [
        local_metrics["load_time_s"],
        local_metrics["transform_time_s"],
        local_metrics["save_time_s"],
        local_metrics["total_time_s"],
    ]
    dist_times = [
        dist_metrics["load_time_s"],
        dist_metrics["transform_time_s"],
        dist_metrics["save_time_s"],
        dist_metrics["total_time_s"],
    ]
    x = range(len(phases))
    axes[0].bar([i - 0.2 for i in x], local_times, width=0.4, label="Local (pandas)", color="steelblue")
    axes[0].bar([i + 0.2 for i in x], dist_times,  width=0.4, label="Distributed (PySpark)", color="darkorange")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(phases)
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title("Execution Time: Local vs. Distributed")
    axes[0].legend()

    # Memory usage
    labels = ["Local (pandas)", "Distributed (PySpark)"]
    mems   = [local_metrics["peak_memory_gb"], dist_metrics["peak_memory_gb"]]
    axes[1].bar(labels, mems, color=["steelblue", "darkorange"])
    axes[1].set_ylabel("Peak Memory (GB)")
    axes[1].set_title("Peak Memory Usage")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "performance_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info("Chart saved to %s", chart_path)


# MAIN

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed feature engineering pipeline.")
    parser.add_argument("--input",    type=str, default="data/raw/",     help="Input data directory")
    parser.add_argument("--output",   type=str, default="data/output/",  help="Output data directory")
    parser.add_argument("--metrics",  type=str, default="metrics/",      help="Metrics/chart output directory")
    parser.add_argument("--workers",  type=int, default=4,               help="Number of Spark local workers")
    parser.add_argument("--mode",     type=str, default="both",          choices=["local", "distributed", "both"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    local_metrics, dist_metrics = None, None

    try:
        if args.mode in ("local", "both"):
            local_metrics = run_local(args.input, os.path.join(args.output, "local"))

        if args.mode in ("distributed", "both"):
            dist_metrics = run_distributed(args.input, os.path.join(args.output, "distributed"), args.workers)

        if local_metrics and dist_metrics:
            save_comparison(local_metrics, dist_metrics, args.metrics)

            speedup = local_metrics["total_time_s"] / dist_metrics["total_time_s"]
            logger.info("Speedup: %.2fx (distributed vs. local)", speedup)

    except Exception:
        logger.error("Pipeline failed:\n%s", traceback.format_exc())
        raise