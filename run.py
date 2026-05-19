"""
MLOps Batch Job — Rolling Mean Signal Pipeline
Usage:
    python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps rolling-mean signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV file")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for log file")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler (stderr so stdout stays clean for JSON)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config loading + validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}

def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config file is empty or not a valid YAML mapping.")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

    # Type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(cfg['seed'])}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str):
        raise ValueError(f"'version' must be a string, got: {type(cfg['version'])}")

    logger.info(
        "Config loaded — version=%s | seed=%d | window=%d",
        cfg["version"], cfg["seed"], cfg["window"]
    )
    return cfg


# ---------------------------------------------------------------------------
# Dataset loading + validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty (no rows).")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' is missing. Found columns: {list(df.columns)}"
        )

    # Coerce close to numeric; rows that can't be parsed become NaN
    original_len = len(df)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    bad_rows = df["close"].isna().sum()
    if bad_rows > 0:
        logger.warning(
            "%d row(s) with non-numeric 'close' values will be dropped.", bad_rows
        )
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    logger.info("Dataset loaded — %d rows (%d dropped).", len(df), original_len - len(df))
    return df


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute rolling mean on 'close' with min_periods=window so the first
    (window-1) rows produce NaN and are excluded from signal computation.
    """
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()
    valid = df["rolling_mean"].notna().sum()
    logger.info(
        "Rolling mean computed — window=%d | valid rows=%d | NaN rows (warm-up)=%d",
        window, valid, len(df) - valid
    )
    return df


def compute_signal(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    signal = 1 if close > rolling_mean, else 0.
    Rows where rolling_mean is NaN (warm-up period) are excluded (signal = NaN).
    """
    df = df.copy()
    mask = df["rolling_mean"].notna()
    df["signal"] = np.nan
    df.loc[mask, "signal"] = (df.loc[mask, "close"] > df.loc[mask, "rolling_mean"]).astype(int)
    logger.info(
        "Signal generated — signal=1 count=%d | signal=0 count=%d | excluded (NaN)=%d",
        (df["signal"] == 1).sum(),
        (df["signal"] == 0).sum(),
        df["signal"].isna().sum()
    )
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_metrics(df: pd.DataFrame, cfg: dict, latency_ms: float) -> dict:
    valid_signals = df["signal"].dropna()
    signal_rate = round(float(valid_signals.mean()), 4)
    return {
        "version":        cfg["version"],
        "rows_processed": int(valid_signals.count()),
        "metric":         "signal_rate",
        "value":          signal_rate,
        "latency_ms":     round(latency_ms, 2),
        "seed":           cfg["seed"],
        "status":         "success",
    }


def write_metrics(metrics: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    logger = setup_logging(args.log_file)

    start_ts = time.time()
    logger.info("=== Job START ===")

    version = "unknown"
    try:
        # 1. Config
        cfg = load_config(args.config, logger)
        version = cfg["version"]

        # 2. Seed
        np.random.seed(cfg["seed"])
        logger.info("Random seed set to %d.", cfg["seed"])

        # 3. Dataset
        df = load_dataset(args.input, logger)

        # 4. Rolling mean
        df = compute_rolling_mean(df, cfg["window"], logger)

        # 5. Signal
        df = compute_signal(df, logger)

        # 6. Metrics
        latency_ms = (time.time() - start_ts) * 1000
        metrics = build_metrics(df, cfg, latency_ms)

        write_metrics(metrics, args.output)
        logger.info("Metrics written to %s.", args.output)
        logger.info(
            "Summary — rows_processed=%d | signal_rate=%.4f | latency_ms=%.2f",
            metrics["rows_processed"], metrics["value"], metrics["latency_ms"]
        )
        logger.info("=== Job END — status=success ===")

        # Print JSON to stdout (Docker requirement)
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        latency_ms = (time.time() - start_ts) * 1000
        logger.error("Pipeline failed: %s", exc, exc_info=True)

        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
            "latency_ms":    round(latency_ms, 2),
        }
        try:
            write_metrics(error_metrics, args.output)
            logger.info("Error metrics written to %s.", args.output)
        except Exception as write_exc:
            logger.error("Could not write error metrics: %s", write_exc)

        logger.info("=== Job END — status=error ===")
        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
