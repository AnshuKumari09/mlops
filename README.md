# MLOps Batch Job — Rolling Mean Signal Pipeline

A minimal MLOps-style batch job that loads OHLCV data, computes a rolling mean on `close`, generates a binary trading signal, and outputs structured metrics.

---

## Project Structure

```
mlops-task/
├── run.py            # Main pipeline script
├── config.yaml       # Job configuration (seed, window, version)
├── data.csv          # Input OHLCV dataset (10,000 rows)
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker build spec
├── metrics.json      # Sample output from a successful run
├── run.log           # Sample log from a successful run
└── README.md         # This file
```

---

## Configuration (`config.yaml`)

```yaml
seed: 42
window: 5
version: "v1"
```

| Field     | Description                          |
|-----------|--------------------------------------|
| `seed`    | NumPy random seed for reproducibility |
| `window`  | Rolling mean window size              |
| `version` | Pipeline version tag in output JSON   |

---

## Local Run Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

### 3. View outputs

```bash
cat metrics.json   # Structured metrics
cat run.log        # Detailed logs
```

---

## Docker Build & Run

### Build the image

```bash
docker build -t mlops-task .
```

### Run the container

```bash
docker run --rm mlops-task
```

The container:
- Includes `data.csv` and `config.yaml` at build time
- Runs the full pipeline
- Prints final `metrics.json` to **stdout**
- Exits with code `0` on success, non-zero on failure

### (Optional) Copy output files from container

```bash
docker run --rm -v $(pwd)/output:/app mlops-task
# metrics.json and run.log will be in ./output/
```

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 25.74,
  "seed": 42,
  "status": "success"
}
```

> **Note:** `rows_processed` is 9996 (not 10000) because the first `window-1 = 4` rows lack a full rolling window and are excluded from signal computation. This is by design and documented in `run.py`.

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' is missing. Found columns: ['open', 'high', 'low', 'volume']",
  "latency_ms": 3.21
}
```

---

## Signal Logic

| Condition              | Signal |
|------------------------|--------|
| `close > rolling_mean` | `1`    |
| `close ≤ rolling_mean` | `0`    |
| Warm-up rows (first `window-1`) | excluded (NaN) |

---

## Reproducibility

Every run with the same `config.yaml` and `data.csv` produces identical outputs. The `seed` field in config controls `numpy.random.seed()`.

---

## Error Handling

The pipeline handles and reports these failure modes cleanly:
- Missing input file
- Missing config file
- Invalid YAML / missing required keys
- Empty CSV
- Missing `close` column
- Non-numeric `close` values (rows are dropped with a warning)

In all error cases, `metrics.json` is still written with `"status": "error"`.