import argparse
import os
import time
import numpy as np
import pandas as pd


def generate_synthetic_data(n_rows: int, seed: int, output_dir: str, n_partitions: int = 10) -> None:
    """
    Generate synthetic user transaction data and save as partitioned Parquet files.

    Args:
        n_rows: Total number of rows to generate.
        seed: Random seed for reproducibility.
        output_dir: Directory to write output Parquet files.
        n_partitions: Number of file partitions to create.
    """
    rng = np.random.default_rng(seed)

    os.makedirs(output_dir, exist_ok=True)

    rows_per_partition = n_rows // n_partitions
    remainder = n_rows % n_partitions

    print(f"Generating {n_rows:,} rows → {n_partitions} partitions in '{output_dir}/'")
    start = time.time()

    for part_idx in range(n_partitions):
        size = rows_per_partition + (1 if part_idx < remainder else 0)

        df = pd.DataFrame({
            "user_id":          rng.integers(1, 1_000_000, size=size),
            "transaction_id":   rng.integers(1, 100_000_000, size=size),
            "amount":           rng.exponential(scale=50.0, size=size).round(2),
            "category":         rng.choice(["food", "travel", "shopping", "health", "entertainment"], size=size),
            "age":              rng.integers(18, 80, size=size),
            "tenure_days":      rng.integers(1, 3650, size=size),
            "num_past_txns":    rng.integers(0, 500, size=size),
            "avg_past_amount":  rng.exponential(scale=45.0, size=size).round(2),
            "is_international": rng.choice([0, 1], size=size, p=[0.85, 0.15]).astype(int),
            "hour_of_day":      rng.integers(0, 24, size=size),
            "day_of_week":      rng.integers(0, 7, size=size),
            "label":            rng.choice([0, 1], size=size, p=[0.97, 0.03]).astype(int),  # fraud flag
        })

        out_path = os.path.join(output_dir, f"part_{part_idx:04d}.parquet")
        df.to_parquet(out_path, index=False)

    elapsed = time.time() - start
    print(f"Done. Generated {n_rows:,} rows in {elapsed:.2f}s → {output_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data.")
    parser.add_argument("--rows",        type=int,   default=10_000_000, help="Total rows to generate")
    parser.add_argument("--seed",        type=int,   default=42,         help="Random seed for reproducibility")
    parser.add_argument("--output",      type=str,   default="data/raw/", help="Output directory")
    parser.add_argument("--partitions",  type=int,   default=10,         help="Number of output file partitions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_synthetic_data(
        n_rows=args.rows,
        seed=args.seed,
        output_dir=args.output,
        n_partitions=args.partitions,
    )