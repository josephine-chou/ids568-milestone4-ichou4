# IDS 568 Milestone 4_Distributed Feature Engineering Pipeline

## Overview

Distributed feature engineering pipeline comparing local (pandas) and distributed
(PySpark) execution on 10M+ rows of synthetic transaction data.

**Features engineered:** log-transform, z-score normalization, derived ratios,
high-value flag, transaction velocity, one-hot encoding.

---

## Repository Structure
```
ids568-milestone4-ichou4/
├── pipeline.py          # Main distributed feature engineering pipeline
├── generate_data.py     # Synthetic data generator (10M+ rows)
├── README.md            # This file
├── REPORT.md            # Performance analysis & architecture evaluation
└── requirements.txt     # Python dependencies
```

---

## Setup Instructions

### Prerequisites
- Python 3.9+
- Java 17 (required by PySpark): `java -version`
  - If not installed: `brew install openjdk@17`
  - After install: `echo 'export JAVA_HOME=$(brew --prefix openjdk@17)' >> ~/.zshrc && source ~/.zshrc`

### 1. Clone repository
```bash
git clone https://github.com/josephine-chou/ids568-milestone4-ichou4.git
cd ids568-milestone4-ichou4
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Execution Instructions

### Step 1: Generate synthetic data
```bash
# Generate 10M rows (full benchmark)
python generate_data.py --rows 10000000 --seed 42 --output data/raw/

# Generate 1000 rows (quick test)
python generate_data.py --rows 1000 --seed 42 --output data/test/
```

### Step 2: Run pipeline
```bash
# Run both local and distributed modes (recommended)
python pipeline.py \
  --input data/raw/ \
  --output data/output/ \
  --metrics metrics/ \
  --workers 4

# Run only distributed mode
python pipeline.py --input data/raw/ --output data/output/ --mode distributed
```

### Step 3: View results
```bash
# View metrics
cat metrics/metrics.json

# View performance chart
open metrics/performance_comparison.png
```

---

## Reproducibility

All results are reproducible using `--seed 42`. Verify:
```bash
python generate_data.py --rows 100 --seed 42 --output run1/
python generate_data.py --rows 100 --seed 42 --output run2/
diff run1/part_0000.parquet run2/part_0000.parquet && echo "✓ Reproducible"
```

---

## Expected Runtime (10M rows)

| Mode | Approx. Runtime |
|------|----------------|
| Local (pandas) | 30–90s |
| Distributed (PySpark local[4]) | 60–180s |

> Note: PySpark has ~15s JVM startup overhead regardless of data size.

---

## Sanity Check
```bash
# Run automated checks
for file in pipeline.py generate_data.py README.md REPORT.md requirements.txt; do
  [ -f "$file" ] && echo "✓ $file" || echo "✗ $file MISSING"
done

python generate_data.py --rows 1000 --output data/test/ && echo "✓ Data generation OK"
python pipeline.py --input data/test/ --output data/test_out/ --metrics metrics_test/ && echo "✓ Pipeline OK"
```