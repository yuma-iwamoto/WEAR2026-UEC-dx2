import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from tqdm import tqdm
from approach_base.src.visualize_confusion_matrix import plot_confusion_matrix, reorder_confusion_matrix_for_blocks
from approach_base.src.output_utils import make_experiment_dir
from approach_base.src.output_utils import save_run_metadata


LABEL_TO_ID = {
    "null": 0,
    "jogging": 1,
    "jogging (rotating arms)": 2,
    "jogging (skipping)": 3,
    "jogging (sidesteps)": 4,
    "jogging (butt-kicks)": 5,
    "stretching (triceps)": 6,
    "stretching (lunging)": 7,
    "stretching (shoulders)": 8,
    "stretching (hamstrings)": 9,
    "stretching (lumbar rotation)": 10,
    "push-ups": 11,
    "push-ups (complex)": 12,
    "sit-ups": 13,
    "sit-ups (complex)": 14,
    "burpees": 15,
    "lunges": 16,
    "lunges (complex)": 17,
    "bench-dips": 18,
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
N_CLASSES = len(LABEL_TO_ID)

SENSOR_COLS = {
    "ra": ["right_arm_acc_x", "right_arm_acc_y", "right_arm_acc_z"],
    "rl": ["right_leg_acc_x", "right_leg_acc_y", "right_leg_acc_z"],
    "ll": ["left_leg_acc_x", "left_leg_acc_y", "left_leg_acc_z"],
    "la": ["left_arm_acc_x", "left_arm_acc_y", "left_arm_acc_z"],
}
SENSOR_TO_ID = {"ra": 0, "la": 1, "rl": 2, "ll": 3}
CACHE_BASE_DIR = Path("/workspace/input/cache/approach_base/train_inertial_gbdt")
CACHE_SCHEMA_VERSION = 10
IMU_HZ = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost/CatBoost/LightGBM using inertial-only window features.")
    parser.add_argument("--root", type=Path, default=Path("/workspace/input/3rd-wear-dataset-challenge-hasca-2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/approach_base/output/inertial_gbdt"), help="Parent directory. Outputs are saved under <output-dir>/<model>/")
    parser.add_argument("--model", choices=["xgboost", "catboost", "lightgbm"], default="lightgbm")
    parser.add_argument("--val-subjects", type=int, nargs="+", default=[18, 19, 20, 21])
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--window-label-mode", choices=["purity", "majority", "strict"], default="purity")
    parser.add_argument("--min-label-purity", type=float, default=0.8)
    parser.add_argument("--sensor-keys", type=str, nargs="+", choices=list(SENSOR_COLS), default=list(SENSOR_COLS))
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--use-raw-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--normalization-mode", choices=["none", "global", "subject", "window"], default="none")
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--top-k-features", type=int, default=None)
    parser.add_argument("--feature-list-file", type=Path, default=None, help="CSV/text file of feature names to keep after feature-table build. CSV uses a feature column when available.")
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--xgb-subsample", type=float, default=0.7)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.6)
    parser.add_argument("--xgb-min-child-weight", type=float, default=5.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=2.0)
    parser.add_argument("--xgb-gamma", type=float, default=0.0)
    parser.add_argument("--lgb-feature-fraction", type=float, default=0.6)
    parser.add_argument("--lgb-bagging-fraction", type=float, default=0.7)
    parser.add_argument("--lgb-bagging-freq", type=int, default=1)
    parser.add_argument("--lgb-min-child-samples", type=int, default=50)
    parser.add_argument("--lgb-min-split-gain", type=float, default=0.1)
    parser.add_argument("--lgb-lambda-l1", type=float, default=0.0)
    parser.add_argument("--lgb-lambda-l2", type=float, default=2.0)
    parser.add_argument("--cat-l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--cat-random-strength", type=float, default=1.0)
    parser.add_argument("--cat-rsm", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-workers", type=int, default=1, help="Number of record-level workers for feature-table build. 1 preserves sequential behavior.")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exclude-file-id-suffix-2", action="store_true")
    parser.add_argument("--sensor-embedding-mode", choices=["sensor", "limb", "none"], default="sensor")
    parser.add_argument("--sensor-fusion-mode", choices=["single", "parallel4"], default="single")
    parser.add_argument("--add-bilateral-features", action=argparse.BooleanOptionalAction, default=False, help="Add left/right pair-difference features for sensor_fusion_mode=parallel4.")
    parser.add_argument("--canonicalize-left-limb", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def encode_label(value) -> int | None:
    if value != value:
        return None
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in {"", "null"}:
            return LABEL_TO_ID["null"]
        if lowered in {"nan", "none"}:
            return None
        if text in LABEL_TO_ID:
            return LABEL_TO_ID[text]
        return int(float(text))
    return int(value)


def normalize_sensor_location(value: str) -> str:
    text = str(value).lower().strip()
    mapping = {
        "right_arm": "ra",
        "right arm": "ra",
        "right_wrist": "ra",
        "right wrist": "ra",
        "ra": "ra",
        "left_arm": "la",
        "left arm": "la",
        "left_wrist": "la",
        "left wrist": "la",
        "la": "la",
        "right_leg": "rl",
        "right leg": "rl",
        "right_ankle": "rl",
        "right ankle": "rl",
        "rl": "rl",
        "left_leg": "ll",
        "left leg": "ll",
        "left_ankle": "ll",
        "left ankle": "ll",
        "ll": "ll",
    }
    if text not in mapping:
        raise ValueError(f"Unknown sensor location: {value}")
    return mapping[text]


def should_exclude_file_id_suffix_2(file_id: str) -> bool:
    parts = str(file_id).split("_")
    return len(parts) >= 3 and parts[-1] == "2"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def load_inertial_records(root_dir: Path, exclude_file_id_suffix_2: bool = False) -> list[dict]:
    records = []
    for path in sorted((root_dir / "train" / "inertial_feat").glob("*.csv")):
        if exclude_file_id_suffix_2 and should_exclude_file_id_suffix_2(path.stem):
            continue
        df = pd.read_csv(path, low_memory=False, keep_default_na=False)
        if "sbj_id" not in df.columns:
            raise ValueError(f"{path} does not have sbj_id column")
        label_ids = np.asarray([encode_label(value) for value in df["label"]], dtype=object)
        sensor_arrays = {}
        for sensor_key, sensor_cols in SENSOR_COLS.items():
            sensor_arrays[sensor_key] = df[sensor_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        records.append({
            "path": path,
            "file_id": path.stem,
            "sbj_id": int(df["sbj_id"].iloc[0]),
            "df": df,
            "label_ids": label_ids,
            "sensor_arrays": sensor_arrays,
        })
    if not records:
        raise FileNotFoundError(f"No train inertial CSVs found under {root_dir}")
    return records


def assign_window_label(
    labels: pd.Series,
    window_size: int,
    window_label_mode: str,
    min_label_purity: float,
) -> int | None:
    encoded = labels.map(encode_label)
    if encoded.isna().any():
        return None

    labels_np = encoded.astype(int).to_numpy()
    values, counts = np.unique(labels_np, return_counts=True)
    majority_label = int(values[counts.argmax()])
    majority_ratio = float(counts.max()) / float(window_size)

    if window_label_mode == "purity":
        if majority_ratio < min_label_purity:
            return None
        return majority_label

    if window_label_mode == "majority":
        return majority_label

    if window_label_mode == "strict":
        if len(values) == 1:
            return int(values[0])
        return None

    raise ValueError(f"Unknown window_label_mode: {window_label_mode}")


def assign_window_label_from_array(
    label_ids: np.ndarray,
    start: int,
    window_size: int,
    window_label_mode: str,
    min_label_purity: float,
) -> int | None:
    window = label_ids[start:start + window_size]
    if window.shape[0] != window_size:
        return None
    if any(value is None for value in window):
        return None

    labels_np = np.asarray(window, dtype=np.int64)
    values, counts = np.unique(labels_np, return_counts=True)
    majority_label = int(values[counts.argmax()])
    majority_ratio = float(counts.max()) / float(window_size)

    if window_label_mode == "purity":
        if majority_ratio < min_label_purity:
            return None
        return majority_label

    if window_label_mode == "majority":
        return majority_label

    if window_label_mode == "strict":
        if len(values) == 1:
            return int(values[0])
        return None

    raise ValueError(f"Unknown window_label_mode: {window_label_mode}")


def build_cache_fingerprint(args: argparse.Namespace) -> tuple[str, dict[str, object]]:
    config = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "window_label_mode": str(args.window_label_mode),
        "min_label_purity": float(args.min_label_purity),
        "sensor_keys": list(args.sensor_keys),
        "max_windows_per_record": None if args.max_windows_per_record is None else int(args.max_windows_per_record),
        "use_raw_features": bool(args.use_raw_features),
        "normalization_mode": str(args.normalization_mode),
        "smoothing_window": int(args.smoothing_window),
        "seed": int(args.seed),
        "sensor_embedding_mode": str(getattr(args, "sensor_embedding_mode", "sensor")),
        "sensor_fusion_mode": str(args.sensor_fusion_mode),
        "add_bilateral_features": bool(args.add_bilateral_features),
        "canonicalize_left_limb": bool(args.canonicalize_left_limb),
    }
    if args.normalization_mode in {"global", "subject"}:
        config["val_subjects"] = [int(v) for v in args.val_subjects]
    cache_key = hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_key, config


def make_record_cache_path(cache_key: str, file_id: str) -> Path:
    return CACHE_BASE_DIR / cache_key / f"{file_id}.pkl"


def select_window_starts(record_length: int, file_id: str, args: argparse.Namespace) -> list[int]:
    starts = list(range(0, record_length - args.window_size + 1, args.stride))
    if args.max_windows_per_record is not None and len(starts) > args.max_windows_per_record:
        seed_material = f"{args.seed}:{file_id}:{args.window_size}:{args.stride}:{args.max_windows_per_record}"
        local_seed = int(hashlib.sha1(seed_material.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(local_seed)
        starts = sorted(rng.choice(starts, size=args.max_windows_per_record, replace=False).tolist())
    return starts


def build_record_table(
    rec: dict,
    args: argparse.Namespace,
    normalization_stats: dict[str, object],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, int]]:
    rows = []
    labels = []
    meta_rows = []
    skipped_label = 0
    skipped_sensor = 0

    label_ids = rec["label_ids"]
    sensor_arrays = rec["sensor_arrays"]
    starts = select_window_starts(len(rec["df"]), rec["file_id"], args)

    for start in starts:
        y = assign_window_label_from_array(
            label_ids,
            start,
            args.window_size,
            args.window_label_mode,
            args.min_label_purity,
        )
        if y is None:
            skipped_label += 1
            continue
        if not has_only_finite_sensor_windows(sensor_arrays, args.sensor_keys, start, args.window_size):
            skipped_sensor += len(args.sensor_keys)
            continue

        if args.sensor_fusion_mode == "parallel4":
            normalized_windows = {}
            invalid_sensor = False
            for sensor_key in args.sensor_keys:
                window = canonicalize_left_limb_window(
                    sensor_arrays[sensor_key][start:start + args.window_size],
                    sensor_key,
                    args.canonicalize_left_limb,
                )
                window = normalize_inertial_window(
                    window,
                    sensor_key,
                    normalization_mode=args.normalization_mode,
                    normalization_stats=normalization_stats,
                    subject_id=rec["sbj_id"],
                )
                if not np.isfinite(window).all():
                    skipped_sensor += 1
                    invalid_sensor = True
                    break
                normalized_windows[sensor_key] = window
            if invalid_sensor:
                continue
            rows.append(
                make_parallel_feature_vector(
                    normalized_windows,
                    use_raw_features=args.use_raw_features,
                    smoothing_window=args.smoothing_window,
                    add_bilateral_features=args.add_bilateral_features,
                )
            )
            labels.append(y)
            meta_rows.append({
                "file_id": rec["file_id"],
                "sbj_id": rec["sbj_id"],
                "start": start,
                "sensor": fused_sensor_name(args.sensor_keys),
            })
            continue

        for sensor_key in args.sensor_keys:
            window = canonicalize_left_limb_window(
                sensor_arrays[sensor_key][start:start + args.window_size],
                sensor_key,
                args.canonicalize_left_limb,
            )
            window = normalize_inertial_window(
                window,
                sensor_key,
                normalization_mode=args.normalization_mode,
                normalization_stats=normalization_stats,
                subject_id=rec["sbj_id"],
            )
            if not np.isfinite(window).all():
                skipped_sensor += 1
                continue
            rows.append(
                make_feature_vector(
                    window,
                    sensor_key,
                    use_raw_features=args.use_raw_features,
                    smoothing_window=args.smoothing_window,
                    sensor_embedding_mode=getattr(args, "sensor_embedding_mode", "sensor"),
                )
            )
            labels.append(y)
            meta_rows.append({
                "file_id": rec["file_id"],
                "sbj_id": rec["sbj_id"],
                "start": start,
                "sensor": sensor_key,
            })

    if not rows:
        raise ValueError(f"No samples built for record {rec['file_id']}")
    stats = {"skipped_label": skipped_label, "skipped_sensor": skipped_sensor}
    return pd.DataFrame(rows), np.asarray(labels, dtype=np.int64), pd.DataFrame(meta_rows), stats


def load_or_build_record_table(
    rec: dict,
    args: argparse.Namespace,
    cache_key: str,
    normalization_stats: dict[str, object],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, int], str]:
    cache_path = make_record_cache_path(cache_key, rec["file_id"])
    if not args.rebuild_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                payload = pickle.load(f)
            return payload["x"], payload["y"], payload["meta"], payload["stats"], "hit"
        except Exception as exc:
            print(f"cache load failed for {cache_path}: {exc}; rebuilding cache")

    x_df, y, meta_df, stats = build_record_table(rec, args, normalization_stats)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(
            {
                "cache_key": cache_key,
                "file_id": rec["file_id"],
                "sbj_id": rec["sbj_id"],
                "x": x_df,
                "y": y,
                "meta": meta_df,
                "stats": stats,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return x_df, y, meta_df, stats, "miss"


def load_or_build_record_table_task(payload: tuple[dict, argparse.Namespace, str, dict[str, object]]):
    rec, args, cache_key, normalization_stats = payload
    return load_or_build_record_table(rec, args, cache_key, normalization_stats)


def _compute_mean_std(sum_vec: np.ndarray, sum_sq_vec: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        mean = np.zeros(3, dtype=np.float32)
        std = np.ones(3, dtype=np.float32)
        return mean, std
    mean = sum_vec / float(count)
    var = np.maximum(sum_sq_vec / float(count) - np.square(mean), 1e-6)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_normalization_stats(records: list[dict]) -> dict[str, object]:
    global_acc: dict[str, dict[str, object]] = {}
    subject_acc: dict[int, dict[str, dict[str, object]]] = {}

    for sensor_key in SENSOR_COLS:
        global_acc[sensor_key] = {
            "sum": np.zeros(3, dtype=np.float64),
            "sum_sq": np.zeros(3, dtype=np.float64),
            "count": 0,
        }

    for rec in records:
        sbj_id = int(rec["sbj_id"])
        subject_acc.setdefault(sbj_id, {})
        for sensor_key in SENSOR_COLS:
            subject_acc[sbj_id].setdefault(
                sensor_key,
                {"sum": np.zeros(3, dtype=np.float64), "sum_sq": np.zeros(3, dtype=np.float64), "count": 0},
            )
            values = np.nan_to_num(rec["sensor_arrays"][sensor_key].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            global_acc[sensor_key]["sum"] += values.sum(axis=0)
            global_acc[sensor_key]["sum_sq"] += np.square(values).sum(axis=0)
            global_acc[sensor_key]["count"] += values.shape[0]
            subject_acc[sbj_id][sensor_key]["sum"] += values.sum(axis=0)
            subject_acc[sbj_id][sensor_key]["sum_sq"] += np.square(values).sum(axis=0)
            subject_acc[sbj_id][sensor_key]["count"] += values.shape[0]

    global_stats = {
        sensor_key: _compute_mean_std(acc["sum"], acc["sum_sq"], int(acc["count"]))
        for sensor_key, acc in global_acc.items()
    }
    subject_stats = {
        sbj_id: {
            sensor_key: _compute_mean_std(acc["sum"], acc["sum_sq"], int(acc["count"]))
            for sensor_key, acc in sensor_map.items()
        }
        for sbj_id, sensor_map in subject_acc.items()
    }
    return {"global": global_stats, "subject": subject_stats}


def has_only_finite_raw_window(window: np.ndarray) -> bool:
    window = np.asarray(window)
    return bool(np.isfinite(window).all())


def has_only_finite_sensor_windows(
    sensor_arrays: dict[str, np.ndarray],
    sensor_keys: list[str],
    start: int,
    window_size: int,
) -> bool:
    for sensor_key in sensor_keys:
        window = sensor_arrays[sensor_key][start:start + window_size]
        if window.shape[0] != window_size:
            return False
        if not has_only_finite_raw_window(window):
            return False
    return True


def canonicalize_left_limb_window(window: np.ndarray, sensor_key: str, enabled: bool) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    if not enabled:
        return window
    window = window.copy()
    if sensor_key == "la":
        window[:, 0] *= -1.0
    elif sensor_key == "ll":
        window[:, 1] *= -1.0
    return window


def normalize_inertial_window(
    window: np.ndarray,
    sensor_key: str,
    normalization_mode: str,
    normalization_stats: dict[str, object] | None,
    subject_id: int | None = None,
) -> np.ndarray:
    window = np.nan_to_num(window.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if normalization_mode == "none":
        return window

    if normalization_mode == "window":
        mean = window.mean(axis=0)
        std = window.std(axis=0)
    elif normalization_mode == "global":
        mean, std = normalization_stats["global"][sensor_key]
    elif normalization_mode == "subject":
        mean = std = None
        if subject_id is not None:
            subject_map = normalization_stats["subject"].get(int(subject_id))
            if subject_map is not None and sensor_key in subject_map:
                mean, std = subject_map[sensor_key]
        if mean is None or std is None:
            mean, std = normalization_stats["global"][sensor_key]
    else:
        raise ValueError(f"Unknown normalization_mode: {normalization_mode}")

    std = np.where(np.asarray(std, dtype=np.float32) < 1e-6, 1.0, std)
    return ((window - mean) / std).astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spectral_entropy(power: np.ndarray) -> float:
    power = np.asarray(power, dtype=np.float64)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    p = power / total
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _variance(x: np.ndarray) -> float:
    return float(np.var(x))


def _iqr(x: np.ndarray) -> float:
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))


def _shape_factor(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x))))
    mean_abs = abs(float(np.mean(x)))
    return rms / max(mean_abs, 1e-6)


def _skewness(x: np.ndarray) -> float:
    std = float(np.std(x))
    if std < 1e-6:
        return 0.0
    centered = (x - np.mean(x)) / std
    return float(np.mean(np.power(centered, 3)))


def _kurtosis(x: np.ndarray) -> float:
    std = float(np.std(x))
    if std < 1e-6:
        return 0.0
    centered = (x - np.mean(x)) / std
    return float(np.mean(np.power(centered, 4)))


def _zero_crossings(x: np.ndarray, center: float = 0.0) -> float:
    centered = np.asarray(x, dtype=np.float64) - float(center)
    if centered.size < 2:
        return 0.0
    signs = np.sign(centered)
    nonzero = signs != 0
    if not np.any(nonzero):
        return 0.0
    signs = signs[nonzero]
    if signs.size < 2:
        return 0.0
    return float(np.sum(signs[:-1] * signs[1:] < 0))


def _mean_crossing_rate(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    crossings = _zero_crossings(x, center=float(np.mean(x)))
    return float(crossings / max(x.size - 1, 1))


def _signal_entropy_hist(x: np.ndarray, n_bins: int = 10) -> float:
    if x.size == 0:
        return 0.0
    hist, _ = np.histogram(x, bins=n_bins)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0.0:
        return 0.0
    p = hist / total
    p = p[p > 0.0]
    return float(-(p * np.log(p)).sum())


def _differential_entropy(x: np.ndarray) -> float:
    var = float(np.var(x))
    if var <= 1e-12:
        return 0.0
    return float(0.5 * np.log(2.0 * np.pi * np.e * var))


def _hjorth_mobility(x: np.ndarray) -> float:
    var0 = float(np.var(x))
    if var0 <= 1e-12 or x.size < 2:
        return 0.0
    var1 = float(np.var(np.diff(x)))
    return float(np.sqrt(var1 / var0))


def _hjorth_complexity(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    mobility = _hjorth_mobility(x)
    if mobility <= 1e-12:
        return 0.0
    diff_x = np.diff(x)
    var1 = float(np.var(diff_x))
    if var1 <= 1e-12:
        return 0.0
    var2 = float(np.var(np.diff(diff_x)))
    return float(np.sqrt(var2 / var1) / mobility)


def _petrosian_fd(x: np.ndarray) -> float:
    n = x.size
    if n < 3:
        return 0.0
    dx = np.diff(x)
    n_delta = _zero_crossings(dx)
    if n_delta < 0:
        return 0.0
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta + 1e-12))))


def _katz_fd(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 0.0
    length = float(np.sum(np.abs(np.diff(x))))
    distance = float(np.max(np.abs(x - x[0])))
    if length <= 1e-12 or distance <= 1e-12:
        return 0.0
    return float(np.log10(n) / (np.log10(distance / length) + np.log10(n)))


def _welch_spectral_entropy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 8:
        return 0.0
    seg_len = max(n // 2, 8)
    step = max(seg_len // 2, 1)
    window = np.hanning(seg_len)
    psd = None
    count = 0
    for start in range(0, n - seg_len + 1, step):
        segment = x[start:start + seg_len]
        segment = (segment - np.mean(segment)) * window
        power = np.abs(np.fft.rfft(segment)) ** 2
        psd = power if psd is None else psd + power
        count += 1
    if count == 0 or psd is None:
        return 0.0
    psd = psd / float(count)
    if psd.size > 0:
        psd = psd[1:]
    if psd.size == 0:
        return 0.0
    return _spectral_entropy(psd)


def _empty_peak_stats() -> dict[str, float]:
    return {
        "peak_count": 0.0,
        "peak_mean": 0.0,
        "peak_std": 0.0,
        "peak_max": 0.0,
        "peak_min": 0.0,
        "peak_interval_mean": 0.0,
        "peak_interval_std": 0.0,
        "peak_first_pos": 0.0,
        "peak_last_pos": 0.0,
        "peak_prominence_mean": 0.0,
        "peak_prominence_std": 0.0,
        "peak_prominence_max": 0.0,
    }


def _peak_stats(x: np.ndarray) -> dict[str, float]:
    if x.size < 3:
        return _empty_peak_stats()
    mask = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
    peak_positions = np.nonzero(mask)[0] + 1
    peaks = x[peak_positions]
    if peaks.size == 0:
        return _empty_peak_stats()
    intervals = np.diff(peak_positions)
    prominence = peaks - np.maximum(x[peak_positions - 1], x[peak_positions + 1])
    denom = float(max(x.size - 1, 1))
    return {
        "peak_count": float(peaks.size),
        "peak_mean": float(peaks.mean()),
        "peak_std": float(peaks.std()),
        "peak_max": float(peaks.max()),
        "peak_min": float(peaks.min()),
        "peak_interval_mean": float(intervals.mean()) if intervals.size > 0 else 0.0,
        "peak_interval_std": float(intervals.std()) if intervals.size > 0 else 0.0,
        "peak_first_pos": float(peak_positions[0]) / denom,
        "peak_last_pos": float(peak_positions[-1]) / denom,
        "peak_prominence_mean": float(prominence.mean()),
        "peak_prominence_std": float(prominence.std()),
        "peak_prominence_max": float(prominence.max()),
    }


def _orientation_features_from_mean(mean_values: np.ndarray, prefix: str = "") -> dict[str, float]:
    mean_x, mean_y, mean_z = [float(value) for value in mean_values]
    denom = float(np.sqrt(mean_x * mean_x + mean_y * mean_y + mean_z * mean_z))
    out_prefix = f"{prefix}_" if prefix else ""
    if denom <= 1e-6:
        tilt_x = tilt_y = tilt_z = 0.0
    else:
        tilt_x = float(np.arccos(np.clip(mean_x / denom, -1.0, 1.0)))
        tilt_y = float(np.arccos(np.clip(mean_y / denom, -1.0, 1.0)))
        tilt_z = float(np.arccos(np.clip(mean_z / denom, -1.0, 1.0)))
    return {
        f"{out_prefix}tilt_to_x": tilt_x,
        f"{out_prefix}tilt_to_y": tilt_y,
        f"{out_prefix}tilt_to_z": tilt_z,
        f"{out_prefix}pitch": float(np.arctan2(-mean_x, np.sqrt(mean_y * mean_y + mean_z * mean_z))),
        f"{out_prefix}roll": float(np.arctan2(mean_y, mean_z)),
    }


def _fft_stats(x: np.ndarray, sample_rate: float) -> dict[str, float]:
    centered = x - np.mean(x)
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    if power.size > 0:
        power = power[1:]
        freqs = freqs[1:]
    if power.size == 0:
        return {
            "fft_bandpower_low": 0.0,
            "fft_bandpower_mid": 0.0,
            "fft_bandpower_high": 0.0,
            "fft_dom_freq": 0.0,
            "fft_dom_power": 0.0,
            "fft_entropy": 0.0,
        }
    low = power[(freqs >= 0.0) & (freqs < 3.0)].sum()
    mid = power[(freqs >= 3.0) & (freqs < 10.0)].sum()
    high = power[freqs >= 10.0].sum()
    dom_idx = int(np.argmax(power))
    return {
        "fft_bandpower_low": float(low),
        "fft_bandpower_mid": float(mid),
        "fft_bandpower_high": float(high),
        "fft_dom_freq": float(freqs[dom_idx]),
        "fft_dom_power": float(power[dom_idx]),
        "fft_entropy": _spectral_entropy(power),
    }


def _moving_average_1d(x: np.ndarray, window_size: int) -> np.ndarray:
    window_size = max(int(window_size), 1)
    if window_size <= 1:
        return x.astype(np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _segment_stats(values: np.ndarray, prefix: str, n_segments: int) -> dict[str, float]:
    features: dict[str, float] = {}
    for seg_idx, segment in enumerate(np.array_split(values, n_segments), start=1):
        if segment.size == 0:
            mean = std = min_value = max_value = 0.0
        else:
            mean = float(np.mean(segment))
            std = float(np.std(segment))
            min_value = float(np.min(segment))
            max_value = float(np.max(segment))
        features[f"{prefix}_seg{n_segments}_{seg_idx}_mean"] = mean
        features[f"{prefix}_seg{n_segments}_{seg_idx}_std"] = std
        features[f"{prefix}_seg{n_segments}_{seg_idx}_min"] = min_value
        features[f"{prefix}_seg{n_segments}_{seg_idx}_max"] = max_value
    return features



def _axis_energy_ratio(window: np.ndarray) -> dict[str, float]:
    axis_energy = np.mean(window * window, axis=0)
    axis_energy_total = float(axis_energy.sum())
    if axis_energy_total <= 1e-12:
        return {"energy_ratio_x": 0.0, "energy_ratio_y": 0.0, "energy_ratio_z": 0.0}
    return {
        "energy_ratio_x": float(axis_energy[0] / axis_energy_total),
        "energy_ratio_y": float(axis_energy[1] / axis_energy_total),
        "energy_ratio_z": float(axis_energy[2] / axis_energy_total),
    }


def _multiaxis_segment_features(window: np.ndarray, prefix: str, n_segments: int) -> dict[str, float]:
    features: dict[str, float] = {}
    names = ("corr_xy", "corr_xz", "corr_yz", "cov_xx", "cov_yy", "cov_zz", "cov_xy", "cov_xz", "cov_yz")
    for seg_idx, segment in enumerate(np.array_split(window, n_segments), start=1):
        seg_prefix = f"{prefix}_seg{n_segments}_{seg_idx}"
        if segment.shape[0] < 2:
            for name in names:
                features[f"{seg_prefix}_{name}"] = 0.0
            for name in ("energy_ratio_x", "energy_ratio_y", "energy_ratio_z"):
                features[f"{seg_prefix}_{name}"] = 0.0
            continue
        features[f"{seg_prefix}_corr_xy"] = _safe_corr(segment[:, 0], segment[:, 1])
        features[f"{seg_prefix}_corr_xz"] = _safe_corr(segment[:, 0], segment[:, 2])
        features[f"{seg_prefix}_corr_yz"] = _safe_corr(segment[:, 1], segment[:, 2])
        cov = np.cov(segment, rowvar=False)
        features[f"{seg_prefix}_cov_xx"] = float(cov[0, 0])
        features[f"{seg_prefix}_cov_yy"] = float(cov[1, 1])
        features[f"{seg_prefix}_cov_zz"] = float(cov[2, 2])
        features[f"{seg_prefix}_cov_xy"] = float(cov[0, 1])
        features[f"{seg_prefix}_cov_xz"] = float(cov[0, 2])
        features[f"{seg_prefix}_cov_yz"] = float(cov[1, 2])
        for name, value in _axis_energy_ratio(segment).items():
            features[f"{seg_prefix}_{name}"] = value
    return features


def _edge_mean_diff_features(window: np.ndarray, prefix: str, n_frames: int) -> dict[str, float]:
    if window.shape[0] < n_frames * 2:
        return {
            f"{prefix}_edge{n_frames}_mean_diff_x": 0.0,
            f"{prefix}_edge{n_frames}_mean_diff_y": 0.0,
            f"{prefix}_edge{n_frames}_mean_diff_z": 0.0,
            f"{prefix}_edge{n_frames}_mean_diff_norm": 0.0,
            f"{prefix}_edge{n_frames}_euclidean_mean_diff": 0.0,
            f"{prefix}_edge{n_frames}_pitch_diff": 0.0,
            f"{prefix}_edge{n_frames}_roll_diff": 0.0,
            f"{prefix}_edge{n_frames}_tilt_to_x_diff": 0.0,
            f"{prefix}_edge{n_frames}_tilt_to_y_diff": 0.0,
            f"{prefix}_edge{n_frames}_tilt_to_z_diff": 0.0,
            f"{prefix}_edge{n_frames}_energy_ratio_x_diff": 0.0,
            f"{prefix}_edge{n_frames}_energy_ratio_y_diff": 0.0,
            f"{prefix}_edge{n_frames}_energy_ratio_z_diff": 0.0,
        }
    first = window[:n_frames]
    last = window[-n_frames:]
    first_mean = np.mean(first, axis=0)
    last_mean = np.mean(last, axis=0)
    diff = last_mean - first_mean
    features = {
        f"{prefix}_edge{n_frames}_mean_diff_x": float(diff[0]),
        f"{prefix}_edge{n_frames}_mean_diff_y": float(diff[1]),
        f"{prefix}_edge{n_frames}_mean_diff_z": float(diff[2]),
        f"{prefix}_edge{n_frames}_mean_diff_norm": float(np.linalg.norm(last_mean) - np.linalg.norm(first_mean)),
        f"{prefix}_edge{n_frames}_euclidean_mean_diff": float(np.linalg.norm(diff)),
    }
    first_orientation = _orientation_features_from_mean(first_mean, prefix="first")
    last_orientation = _orientation_features_from_mean(last_mean, prefix="last")
    for name in ("pitch", "roll", "tilt_to_x", "tilt_to_y", "tilt_to_z"):
        features[f"{prefix}_edge{n_frames}_{name}_diff"] = abs(last_orientation[f"last_{name}"] - first_orientation[f"first_{name}"])
    first_energy = _axis_energy_ratio(first)
    last_energy = _axis_energy_ratio(last)
    for name in ("energy_ratio_x", "energy_ratio_y", "energy_ratio_z"):
        features[f"{prefix}_edge{n_frames}_{name}_diff"] = abs(last_energy[name] - first_energy[name])
    return features


def _scalar_edge_diff_features(x: np.ndarray, prefix: str, n_frames: int) -> dict[str, float]:
    if x.size < n_frames * 2:
        return {
            f"{prefix}_edge{n_frames}_mean_diff": 0.0,
            f"{prefix}_edge{n_frames}_abs_mean_diff": 0.0,
            f"{prefix}_edge{n_frames}_std_diff": 0.0,
            f"{prefix}_edge{n_frames}_rms_diff": 0.0,
        }
    first = x[:n_frames]
    last = x[-n_frames:]
    mean_diff = float(np.mean(last) - np.mean(first))
    first_rms = float(np.sqrt(np.mean(first * first)))
    last_rms = float(np.sqrt(np.mean(last * last)))
    return {
        f"{prefix}_edge{n_frames}_mean_diff": mean_diff,
        f"{prefix}_edge{n_frames}_abs_mean_diff": abs(mean_diff),
        f"{prefix}_edge{n_frames}_std_diff": float(np.std(last) - np.std(first)),
        f"{prefix}_edge{n_frames}_rms_diff": float(last_rms - first_rms),
    }


def _smoothed_diff_features(x: np.ndarray, prefix: str, smoothing_window: int) -> dict[str, float]:
    smoothed = _moving_average_1d(x, smoothing_window)
    smoothed_diff = np.diff(smoothed)
    center_diff = np.gradient(smoothed)
    smoothed_diff_rms = float(np.sqrt(np.mean(smoothed_diff * smoothed_diff))) if smoothed_diff.size > 0 else 0.0
    center_diff_rms = float(np.sqrt(np.mean(center_diff * center_diff))) if center_diff.size > 0 else 0.0
    return {
        f"{prefix}_smdiff_mean": float(np.mean(smoothed_diff)) if smoothed_diff.size > 0 else 0.0,
        f"{prefix}_smdiff_std": float(np.std(smoothed_diff)) if smoothed_diff.size > 0 else 0.0,
        f"{prefix}_smdiff_abs_mean": float(np.mean(np.abs(smoothed_diff))) if smoothed_diff.size > 0 else 0.0,
        f"{prefix}_smdiff_rms": smoothed_diff_rms,
        f"{prefix}_cdiff_mean": float(np.mean(center_diff)) if center_diff.size > 0 else 0.0,
        f"{prefix}_cdiff_std": float(np.std(center_diff)) if center_diff.size > 0 else 0.0,
        f"{prefix}_cdiff_abs_mean": float(np.mean(np.abs(center_diff))) if center_diff.size > 0 else 0.0,
        f"{prefix}_cdiff_rms": center_diff_rms,
    }


def _sign_change_stats(x: np.ndarray) -> tuple[float, float]:
    signs = np.sign(np.asarray(x, dtype=np.float64))
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0, 0.0
    changes = float(np.sum(signs[:-1] != signs[1:]))
    rate = changes / float(signs.size - 1)
    return changes, rate


def _run_length_stats(mask: np.ndarray) -> tuple[float, float, float]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not np.any(mask):
        return 0.0, 0.0, 0.0
    padded = np.concatenate([[False], mask, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    lengths = (ends - starts).astype(np.float64)
    return float(lengths.size), float(lengths.mean()), float(lengths.max())


def _lag_diff_features(x: np.ndarray, prefix: str, lag: int) -> dict[str, float]:
    lag = int(lag)
    if lag <= 0 or x.size <= lag:
        return {
            f"{prefix}_lag{lag}_diff_mean": 0.0,
            f"{prefix}_lag{lag}_diff_std": 0.0,
            f"{prefix}_lag{lag}_diff_min": 0.0,
            f"{prefix}_lag{lag}_diff_max": 0.0,
            f"{prefix}_lag{lag}_diff_rms": 0.0,
            f"{prefix}_lag{lag}_diff_abs_mean": 0.0,
            f"{prefix}_lag{lag}_diff_skewness": 0.0,
            f"{prefix}_lag{lag}_diff_kurtosis": 0.0,
            f"{prefix}_lag{lag}_diff_zero_crossings": 0.0,
            f"{prefix}_lag{lag}_diff_mean_crossing_rate": 0.0,
            f"{prefix}_lag{lag}_diff_sign_change_count": 0.0,
            f"{prefix}_lag{lag}_diff_sign_change_rate": 0.0,
            f"{prefix}_lag{lag}_diff_pos_run_count": 0.0,
            f"{prefix}_lag{lag}_diff_pos_run_mean": 0.0,
            f"{prefix}_lag{lag}_diff_pos_run_max": 0.0,
            f"{prefix}_lag{lag}_diff_neg_run_count": 0.0,
            f"{prefix}_lag{lag}_diff_neg_run_mean": 0.0,
            f"{prefix}_lag{lag}_diff_neg_run_max": 0.0,
            f"{prefix}_lag{lag}_diff_pos_ratio": 0.0,
            f"{prefix}_lag{lag}_diff_neg_ratio": 0.0,
            f"{prefix}_lag{lag}_diff_zero_ratio": 0.0,
            f"{prefix}_lag{lag}_diff_pos_mean": 0.0,
            f"{prefix}_lag{lag}_diff_neg_mean": 0.0,
        }
    diff = x[lag:] - x[:-lag]
    pos = diff > 0
    neg = diff < 0
    zero = diff == 0
    pos_mean = float(np.mean(diff[pos])) if np.any(pos) else 0.0
    neg_mean = float(np.mean(diff[neg])) if np.any(neg) else 0.0
    sign_change_count, sign_change_rate = _sign_change_stats(diff)
    pos_run_count, pos_run_mean, pos_run_max = _run_length_stats(pos)
    neg_run_count, neg_run_mean, neg_run_max = _run_length_stats(neg)
    return {
        f"{prefix}_lag{lag}_diff_mean": float(np.mean(diff)),
        f"{prefix}_lag{lag}_diff_std": float(np.std(diff)),
        f"{prefix}_lag{lag}_diff_min": float(np.min(diff)),
        f"{prefix}_lag{lag}_diff_max": float(np.max(diff)),
        f"{prefix}_lag{lag}_diff_rms": float(np.sqrt(np.mean(diff * diff))),
        f"{prefix}_lag{lag}_diff_abs_mean": float(np.mean(np.abs(diff))),
        f"{prefix}_lag{lag}_diff_skewness": _skewness(diff),
        f"{prefix}_lag{lag}_diff_kurtosis": _kurtosis(diff),
        f"{prefix}_lag{lag}_diff_zero_crossings": _zero_crossings(diff),
        f"{prefix}_lag{lag}_diff_mean_crossing_rate": _mean_crossing_rate(diff),
        f"{prefix}_lag{lag}_diff_sign_change_count": sign_change_count,
        f"{prefix}_lag{lag}_diff_sign_change_rate": sign_change_rate,
        f"{prefix}_lag{lag}_diff_pos_run_count": pos_run_count,
        f"{prefix}_lag{lag}_diff_pos_run_mean": pos_run_mean,
        f"{prefix}_lag{lag}_diff_pos_run_max": pos_run_max,
        f"{prefix}_lag{lag}_diff_neg_run_count": neg_run_count,
        f"{prefix}_lag{lag}_diff_neg_run_mean": neg_run_mean,
        f"{prefix}_lag{lag}_diff_neg_run_max": neg_run_max,
        f"{prefix}_lag{lag}_diff_pos_ratio": float(np.mean(pos)),
        f"{prefix}_lag{lag}_diff_neg_ratio": float(np.mean(neg)),
        f"{prefix}_lag{lag}_diff_zero_ratio": float(np.mean(zero)),
        f"{prefix}_lag{lag}_diff_pos_mean": pos_mean,
        f"{prefix}_lag{lag}_diff_neg_mean": neg_mean,
    }


def add_sensor_position_features(features: dict[str, float], sensor_key: str, sensor_embedding_mode: str) -> None:
    if sensor_embedding_mode == "sensor":
        sensor_id = SENSOR_TO_ID[sensor_key]
        features["sensor_id"] = float(sensor_id)
        for key, value in SENSOR_TO_ID.items():
            features[f"sensor_is_{key}"] = float(sensor_id == value)
        return
    if sensor_embedding_mode == "limb":
        limb_id = 0 if sensor_key in {"ra", "la"} else 1
        features["sensor_id"] = float(limb_id)
        features["sensor_is_arm"] = float(limb_id == 0)
        features["sensor_is_leg"] = float(limb_id == 1)
        return
    if sensor_embedding_mode == "none":
        return
    raise ValueError(f"Unsupported sensor_embedding_mode: {sensor_embedding_mode}")


def make_feature_vector(
    window: np.ndarray,
    sensor_key: str,
    use_raw_features: bool = True,
    smoothing_window: int = 5,
    include_sensor_features: bool = True,
    sensor_embedding_mode: str = "sensor",
) -> dict[str, float]:
    if window.shape == (3, 50):
        window = window.T
    if window.shape[1] != 3:
        raise ValueError(f"Expected 3-axis window, got shape={window.shape}")

    window = np.nan_to_num(window.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    features: dict[str, float] = {}
    axis_names = ["x", "y", "z"]

    axis_means = []
    for axis_idx, axis_name in enumerate(axis_names):
        x = window[:, axis_idx]
        dx = np.diff(x)
        ddx = np.diff(dx)
        half = x.shape[0] // 2
        first_half = x[:half]
        second_half = x[half:]
        axis_means.append(float(np.mean(x)))
        features[f"{axis_name}_mean"] = float(np.mean(x))
        features[f"{axis_name}_variance"] = _variance(x)
        features[f"{axis_name}_std"] = float(np.std(x))
        features[f"{axis_name}_min"] = float(np.min(x))
        features[f"{axis_name}_max"] = float(np.max(x))
        features[f"{axis_name}_median"] = float(np.median(x))
        features[f"{axis_name}_q25"] = float(np.quantile(x, 0.25))
        features[f"{axis_name}_q75"] = float(np.quantile(x, 0.75))
        features[f"{axis_name}_iqr"] = _iqr(x)
        features[f"{axis_name}_range"] = float(np.max(x) - np.min(x))
        features[f"{axis_name}_rms"] = float(np.sqrt(np.mean(x * x)))
        features[f"{axis_name}_shape_factor"] = _shape_factor(x)
        features[f"{axis_name}_abs_mean"] = float(np.mean(np.abs(x)))
        features[f"{axis_name}_skewness"] = _skewness(x)
        features[f"{axis_name}_kurtosis"] = _kurtosis(x)
        features[f"{axis_name}_zero_crossings"] = _zero_crossings(x)
        features[f"{axis_name}_mean_crossing_rate"] = _mean_crossing_rate(x)
        features[f"{axis_name}_signal_entropy"] = _signal_entropy_hist(x)
        features[f"{axis_name}_diff_entropy"] = _differential_entropy(x)
        features[f"{axis_name}_hjorth_mobility"] = _hjorth_mobility(x)
        features[f"{axis_name}_hjorth_complexity"] = _hjorth_complexity(x)
        features[f"{axis_name}_petrosian_fd"] = _petrosian_fd(x)
        features[f"{axis_name}_katz_fd"] = _katz_fd(x)
        features[f"{axis_name}_diff_mean"] = float(np.mean(dx)) if dx.size > 0 else 0.0
        features[f"{axis_name}_diff_std"] = float(np.std(dx)) if dx.size > 0 else 0.0
        features[f"{axis_name}_diff_abs_mean"] = float(np.mean(np.abs(dx))) if dx.size > 0 else 0.0
        features[f"{axis_name}_jerk_mean"] = float(np.mean(ddx)) if ddx.size > 0 else 0.0
        features[f"{axis_name}_jerk_std"] = float(np.std(ddx)) if ddx.size > 0 else 0.0
        features[f"{axis_name}_jerk_abs_mean"] = float(np.mean(np.abs(ddx))) if ddx.size > 0 else 0.0
        features[f"{axis_name}_half_mean_diff"] = float(np.abs(np.mean(second_half) - np.mean(first_half)))
        features.update(_smoothed_diff_features(x, axis_name, smoothing_window))
        for lag in (5, 10, 20, 30):
            features.update(_lag_diff_features(x, axis_name, lag))
        for n_frames in (10, 20):
            features.update(_scalar_edge_diff_features(x, axis_name, n_frames))
        features.update(_segment_stats(x, axis_name, 3))
        features.update(_segment_stats(x, axis_name, 5))
        for name, value in _peak_stats(x).items():
            features[f"{axis_name}_{name}"] = value
        for name, value in _fft_stats(x, IMU_HZ).items():
            features[f"{axis_name}_{name}"] = value

    norm = np.linalg.norm(window, axis=1)
    norm_l1 = np.sum(np.abs(window), axis=1)
    dnorm = np.diff(norm)
    ddnorm = np.diff(dnorm)
    diff_window = np.diff(window, axis=0)
    diff_norm = np.linalg.norm(diff_window, axis=1)
    jerk_window = np.diff(diff_window, axis=0)
    jerk_norm = np.linalg.norm(jerk_window, axis=1)
    half = norm.shape[0] // 2
    features["sma"] = float(np.mean(np.sum(np.abs(window), axis=1)))
    features["total_mean_signed"] = float(np.sum(axis_means))
    features["total_mean_euclidean"] = float(np.sqrt(np.sum(np.square(axis_means))))
    features["norm_mean"] = float(np.mean(norm))
    features["norm_variance"] = _variance(norm)
    features["norm_std"] = float(np.std(norm))
    features["norm_min"] = float(np.min(norm))
    features["norm_max"] = float(np.max(norm))
    features["norm_peak_to_peak"] = float(np.max(norm) - np.min(norm))
    features["norm_iqr"] = _iqr(norm)
    features["norm_rms"] = float(np.sqrt(np.mean(norm * norm)))
    features["norm_l1_mean"] = float(np.mean(norm_l1))
    features["norm_l1_std"] = float(np.std(norm_l1))
    features["norm_skewness"] = _skewness(norm)
    features["norm_kurtosis"] = _kurtosis(norm)
    features["norm_diff_abs_mean"] = float(np.mean(np.abs(dnorm))) if dnorm.size > 0 else 0.0
    features["diff_norm_mean"] = float(np.mean(diff_norm)) if diff_norm.size > 0 else 0.0
    features["diff_norm_std"] = float(np.std(diff_norm)) if diff_norm.size > 0 else 0.0
    features["diff_norm_min"] = float(np.min(diff_norm)) if diff_norm.size > 0 else 0.0
    features["diff_norm_max"] = float(np.max(diff_norm)) if diff_norm.size > 0 else 0.0
    features["diff_norm_rms"] = float(np.sqrt(np.mean(diff_norm * diff_norm))) if diff_norm.size > 0 else 0.0
    features["diff_norm_abs_mean"] = float(np.mean(np.abs(diff_norm))) if diff_norm.size > 0 else 0.0
    features["diff_norm_zero_crossings"] = _zero_crossings(dnorm) if dnorm.size > 0 else 0.0
    features["jerk_norm_mean"] = float(np.mean(jerk_norm)) if jerk_norm.size > 0 else 0.0
    features["jerk_norm_std"] = float(np.std(jerk_norm)) if jerk_norm.size > 0 else 0.0
    features["jerk_norm_min"] = float(np.min(jerk_norm)) if jerk_norm.size > 0 else 0.0
    features["jerk_norm_max"] = float(np.max(jerk_norm)) if jerk_norm.size > 0 else 0.0
    features["jerk_norm_rms"] = float(np.sqrt(np.mean(jerk_norm * jerk_norm))) if jerk_norm.size > 0 else 0.0
    features["jerk_norm_abs_mean"] = float(np.mean(np.abs(jerk_norm))) if jerk_norm.size > 0 else 0.0
    features["norm_jerk_abs_mean"] = float(np.mean(np.abs(ddnorm))) if ddnorm.size > 0 else 0.0
    features["norm_half_mean_diff"] = float(np.abs(np.mean(norm[half:]) - np.mean(norm[:half])))
    features["norm_mean_crossing_rate"] = _mean_crossing_rate(norm)
    features["norm_signal_entropy"] = _signal_entropy_hist(norm)
    features["norm_diff_entropy"] = _differential_entropy(norm)
    features["norm_hjorth_mobility"] = _hjorth_mobility(norm)
    features["norm_hjorth_complexity"] = _hjorth_complexity(norm)
    features["norm_petrosian_fd"] = _petrosian_fd(norm)
    features["norm_katz_fd"] = _katz_fd(norm)
    features["norm_welch_entropy"] = _welch_spectral_entropy(norm)
    features.update(_smoothed_diff_features(norm, "norm", smoothing_window))
    for lag in (5, 10, 20, 30):
        features.update(_lag_diff_features(norm, "norm", lag))
    for n_frames in (10, 20):
        features.update(_scalar_edge_diff_features(norm, "norm", n_frames))
    features.update(_segment_stats(norm, "norm", 3))
    features.update(_segment_stats(norm, "norm", 5))
    for name, value in _peak_stats(norm).items():
        features[f"norm_{name}"] = value
    for name, value in _fft_stats(norm, IMU_HZ).items():
        features[f"norm_{name}"] = value

    orientation = _orientation_features_from_mean(np.asarray(axis_means, dtype=np.float32))
    features.update(orientation)
    features["tilt_angle"] = orientation["tilt_to_z"]

    features.update(_axis_energy_ratio(window))
    for n_frames in (10, 20):
        features.update(_edge_mean_diff_features(window, "window", n_frames))

    if half > 0 and norm.shape[0] > half:
        first_orientation = _orientation_features_from_mean(np.mean(window[:half], axis=0), prefix="first_half")
        second_orientation = _orientation_features_from_mean(np.mean(window[half:], axis=0), prefix="second_half")
        for name in ("pitch", "roll", "tilt_to_x", "tilt_to_y", "tilt_to_z"):
            features[f"{name}_half_diff"] = abs(second_orientation[f"second_half_{name}"] - first_orientation[f"first_half_{name}"])
    else:
        for name in ("pitch", "roll", "tilt_to_x", "tilt_to_y", "tilt_to_z"):
            features[f"{name}_half_diff"] = 0.0

    for seg_idx, segment in enumerate(np.array_split(window, 3), start=1):
        if segment.size == 0:
            seg_orientation = {
                f"seg3_{seg_idx}_pitch": 0.0,
                f"seg3_{seg_idx}_roll": 0.0,
                f"seg3_{seg_idx}_tilt_to_x": 0.0,
                f"seg3_{seg_idx}_tilt_to_y": 0.0,
                f"seg3_{seg_idx}_tilt_to_z": 0.0,
            }
        else:
            seg_orientation = _orientation_features_from_mean(np.mean(segment, axis=0), prefix=f"seg3_{seg_idx}")
        features.update(seg_orientation)
    features.update(_multiaxis_segment_features(window, "window", 3))

    features["corr_xy"] = _safe_corr(window[:, 0], window[:, 1])
    features["corr_xz"] = _safe_corr(window[:, 0], window[:, 2])
    features["corr_yz"] = _safe_corr(window[:, 1], window[:, 2])
    cov = np.cov(window, rowvar=False)
    features["cov_xx"] = float(cov[0, 0])
    features["cov_yy"] = float(cov[1, 1])
    features["cov_zz"] = float(cov[2, 2])
    features["cov_xy"] = float(cov[0, 1])
    features["cov_xz"] = float(cov[0, 2])
    features["cov_yz"] = float(cov[1, 2])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.clip(eigvals, a_min=0.0, a_max=None))[::-1]
    eig_total = float(eigvals.sum())
    features["pca_eig1"] = float(eigvals[0])
    features["pca_eig2"] = float(eigvals[1])
    features["pca_eig3"] = float(eigvals[2])
    features["pca_ratio_12"] = float(eigvals[0] / eigvals[1]) if eigvals[1] > 0.0 else 0.0
    features["pca_ratio_13"] = float(eigvals[0] / eigvals[2]) if eigvals[2] > 0.0 else 0.0
    features["pca_energy_1"] = float(eigvals[0] / eig_total) if eig_total > 0.0 else 0.0
    features["pca_energy_2"] = float(eigvals[1] / eig_total) if eig_total > 0.0 else 0.0
    features["pca_energy_3"] = float(eigvals[2] / eig_total) if eig_total > 0.0 else 0.0

    if use_raw_features:
        for t in range(window.shape[0]):
            for axis_idx, axis_name in enumerate(axis_names):
                features[f"raw_{t:02d}_{axis_name}"] = float(window[t, axis_idx])
        for t in range(diff_window.shape[0]):
            for axis_idx, axis_name in enumerate(axis_names):
                features[f"raw_diff_{t:02d}_{axis_name}"] = float(diff_window[t, axis_idx])
            features[f"raw_diff_norm_{t:02d}"] = float(diff_norm[t]) if diff_norm.size > t else 0.0
        for lag in (5, 10, 20, 30):
            if window.shape[0] > lag:
                lagged_diff = window[lag:] - window[:-lag]
                lagged_norm = norm[lag:] - norm[:-lag]
                for t in range(lagged_diff.shape[0]):
                    for axis_idx, axis_name in enumerate(axis_names):
                        features[f"raw_diff_lag{lag}_{t:02d}_{axis_name}"] = float(lagged_diff[t, axis_idx])
                    features[f"raw_diff_lag{lag}_norm_{t:02d}"] = float(lagged_norm[t])

    if include_sensor_features:
        add_sensor_position_features(features, sensor_key, sensor_embedding_mode)
    return features


def prefix_feature_dict(features: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}__{key}": value for key, value in features.items()}


def make_scalar_series_feature_vector(
    series: np.ndarray,
    prefix: str,
    use_raw_features: bool = True,
    smoothing_window: int = 5,
) -> dict[str, float]:
    x = np.asarray(series, dtype=np.float32).reshape(-1)
    dx = np.diff(x)
    ddx = np.diff(dx) if dx.size > 1 else np.asarray([], dtype=np.float32)
    features = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_q25": float(np.quantile(x, 0.25)),
        f"{prefix}_q75": float(np.quantile(x, 0.75)),
        f"{prefix}_iqr": _iqr(x),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x * x))),
        f"{prefix}_abs_mean": float(np.mean(np.abs(x))),
        f"{prefix}_diff_mean": float(np.mean(dx)) if dx.size > 0 else 0.0,
        f"{prefix}_diff_std": float(np.std(dx)) if dx.size > 0 else 0.0,
        f"{prefix}_diff_abs_mean": float(np.mean(np.abs(dx))) if dx.size > 0 else 0.0,
        f"{prefix}_second_diff_abs_mean": float(np.mean(np.abs(ddx))) if ddx.size > 0 else 0.0,
        f"{prefix}_zero_crossings": _zero_crossings(x),
        f"{prefix}_mean_crossing_rate": _mean_crossing_rate(x),
        f"{prefix}_signal_entropy": _signal_entropy_hist(x),
        f"{prefix}_welch_entropy": _welch_spectral_entropy(x),
    }
    features.update(_smoothed_diff_features(x, prefix, smoothing_window))
    for lag in (5, 10, 20, 30):
        features.update(_lag_diff_features(x, prefix, lag))
    for n_frames in (10, 20):
        features.update(_scalar_edge_diff_features(x, prefix, n_frames))
    features.update(_segment_stats(x, prefix, 3))
    features.update(_segment_stats(x, prefix, 5))
    if use_raw_features:
        for t, value in enumerate(x):
            features[f"{prefix}_raw_{t:02d}"] = float(value)
    return features


def make_bilateral_feature_vector(
    sensor_windows: dict[str, np.ndarray],
    use_raw_features: bool = True,
    smoothing_window: int = 5,
) -> dict[str, float]:
    features: dict[str, float] = {}
    pair_defs = {
        "arm": ("ra", "la"),
        "leg": ("rl", "ll"),
    }
    for pair_name, (right_key, left_key) in pair_defs.items():
        right = sensor_windows[right_key]
        left = sensor_windows[left_key]
        right_minus_left = right - left
        abs_diff = np.zeros_like(right_minus_left, dtype=np.float32)
        abs_diff[1:] = np.abs(right_minus_left[1:] - right_minus_left[:-1])
        abs_second_diff = np.zeros_like(right_minus_left, dtype=np.float32)
        abs_second_diff[2:] = np.abs(right_minus_left[2:] - 2.0 * right_minus_left[1:-1] + right_minus_left[:-2])
        right_mag = np.linalg.norm(right, axis=1, keepdims=True)
        left_mag = np.linalg.norm(left, axis=1, keepdims=True)
        vector_windows = {
            "abs_right_minus_left": np.abs(right_minus_left),
            "abs_diff_right_minus_left": abs_diff,
            "abs_second_diff_right_minus_left": abs_second_diff,
        }
        for derived_name, derived_window in vector_windows.items():
            features.update(
                prefix_feature_dict(
                    make_feature_vector(
                        derived_window.astype(np.float32, copy=False),
                        right_key,
                        use_raw_features=use_raw_features,
                        smoothing_window=smoothing_window,
                        include_sensor_features=False,
                    ),
                    f"bilateral_{pair_name}_{derived_name}",
                )
            )
        features.update(
            make_scalar_series_feature_vector(
                np.abs(right_mag - left_mag),
                f"bilateral_{pair_name}_abs_right_left_magnitude_diff",
                use_raw_features=use_raw_features,
                smoothing_window=smoothing_window,
            )
        )
    return features


def make_parallel_feature_vector(
    sensor_windows: dict[str, np.ndarray],
    use_raw_features: bool = True,
    smoothing_window: int = 5,
    add_bilateral_features: bool = False,
) -> dict[str, float]:
    features: dict[str, float] = {}
    for sensor_key in SENSOR_COLS:
        if sensor_key not in sensor_windows:
            raise ValueError(f"Missing sensor window for {sensor_key}")
        features.update(
            prefix_feature_dict(
                make_feature_vector(
                    sensor_windows[sensor_key],
                    sensor_key,
                    use_raw_features=use_raw_features,
                    smoothing_window=smoothing_window,
                ),
                sensor_key,
            )
        )
    if add_bilateral_features:
        features.update(
            make_bilateral_feature_vector(
                sensor_windows,
                use_raw_features=use_raw_features,
                smoothing_window=smoothing_window,
            )
        )
    return features


def fused_sensor_name(sensor_keys: list[str]) -> str:
    return "+".join(sensor_keys)


def validate_sensor_fusion_configuration(args: argparse.Namespace) -> None:
    if args.sensor_fusion_mode == "parallel4" and list(args.sensor_keys) != list(SENSOR_COLS):
        raise ValueError(
            f"sensor_fusion_mode=parallel4 requires sensor_keys={list(SENSOR_COLS)}, got {list(args.sensor_keys)}"
        )



def build_table(
    records: list[dict],
    args: argparse.Namespace,
    split_name: str,
    normalization_stats: dict[str, object],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    cache_key, cache_config = build_cache_fingerprint(args)
    cache_dir = CACHE_BASE_DIR / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / "cache_config.json"
    if not config_path.exists() or args.rebuild_cache:
        config_path.write_text(json.dumps(cache_config, indent=2, sort_keys=True) + "\n")

    frames = []
    labels = []
    metas = []
    skipped_label = 0
    skipped_sensor = 0
    cache_hits = 0
    cache_misses = 0

    build_workers = max(int(getattr(args, "build_workers", 1)), 1)
    if build_workers == 1 or len(records) <= 1:
        iterator = (load_or_build_record_table(rec, args, cache_key, normalization_stats) for rec in records)
        results_iter = tqdm(iterator, total=len(records), desc=f"build_{split_name}", unit="record")
        for x_df, y, meta_df, stats, cache_status in results_iter:
            if cache_status == "hit":
                cache_hits += 1
            else:
                cache_misses += 1
            skipped_label += int(stats.get("skipped_label", 0))
            skipped_sensor += int(stats.get("skipped_sensor", 0))
            frames.append(x_df)
            labels.append(y)
            metas.append(meta_df)
    else:
        tasks = [(rec, args, cache_key, normalization_stats) for rec in records]
        with ProcessPoolExecutor(max_workers=build_workers) as executor:
            results_iter = executor.map(load_or_build_record_table_task, tasks)
            for x_df, y, meta_df, stats, cache_status in tqdm(results_iter, total=len(records), desc=f"build_{split_name}", unit="record"):
                if cache_status == "hit":
                    cache_hits += 1
                else:
                    cache_misses += 1
                skipped_label += int(stats.get("skipped_label", 0))
                skipped_sensor += int(stats.get("skipped_sensor", 0))
                frames.append(x_df)
                labels.append(y)
                metas.append(meta_df)

    x_out = pd.concat(frames, axis=0, ignore_index=True)
    y_out = np.concatenate(labels, axis=0)
    meta_out = pd.concat(metas, axis=0, ignore_index=True)
    print(
        f"[{split_name}] samples={len(x_out)} skipped_label={skipped_label} skipped_sensor={skipped_sensor} "
        f"cache_hits={cache_hits} cache_misses={cache_misses} cache_key={cache_key}"
    )
    return sanitize_feature_table(x_out, split_name), y_out, meta_out


def sanitize_feature_table(x_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    values = x_df.to_numpy(dtype=np.float32, copy=True)
    invalid_mask = ~np.isfinite(values)
    invalid_count = int(invalid_mask.sum())
    if invalid_count == 0:
        return x_df

    affected_rows = int(invalid_mask.any(axis=1).sum())
    affected_cols = int(invalid_mask.any(axis=0).sum())
    print(
        f"[{split_name}] sanitize_feature_table invalid_values={invalid_count} "
        f"affected_rows={affected_rows} affected_cols={affected_cols}"
    )
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return pd.DataFrame(values, columns=x_df.columns, index=x_df.index)


def load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "feature" in df.columns:
            values = df["feature"].astype(str).tolist()
        else:
            values = df.iloc[:, 0].astype(str).tolist()
    else:
        values = [line.strip() for line in path.read_text().splitlines()]
    features = [value for value in values if value]
    if not features:
        raise ValueError(f"No feature names loaded from {path}")
    return features


def apply_feature_list(x_train: pd.DataFrame, x_val: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    missing = [feature for feature in feature_names if feature not in x_train.columns]
    if missing:
        raise ValueError(f"feature_list_file contains {len(missing)} missing features. First missing: {missing[:20]}")
    return x_train[feature_names].copy(), x_val.reindex(columns=feature_names, fill_value=0.0).copy(), list(feature_names)


def select_top_k_features(x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, top_k: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    if top_k <= 0:
        raise ValueError(f"top_k_features must be positive, got {top_k}")
    if top_k >= x_train.shape[1]:
        scores = pd.DataFrame({"feature": x_train.columns, "score": np.nan, "p_value": np.nan})
        return x_train, x_val, list(x_train.columns), scores

    scores, p_values = f_classif(x_train.to_numpy(dtype=np.float32), y_train)
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    ranking = pd.DataFrame({
        "feature": x_train.columns,
        "score": scores,
        "p_value": p_values,
    }).sort_values(["score", "feature"], ascending=[False, True], kind="stable")
    selected = ranking.head(top_k)["feature"].tolist()
    return x_train[selected].copy(), x_val[selected].copy(), selected, ranking


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def balanced_class_weights(y: np.ndarray) -> tuple[list[float], dict[int, float]]:
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    nonzero = counts > 0
    weights = np.ones(N_CLASSES, dtype=np.float64)
    weights[nonzero] = len(y) / (float(nonzero.sum()) * counts[nonzero])
    return weights.tolist(), {int(i): float(weights[i]) for i in np.where(nonzero)[0]}


def build_sample_weights(y: np.ndarray, class_weights: dict[int, float]) -> np.ndarray:
    return np.asarray([class_weights[int(label)] for label in y], dtype=np.float32)


def train_model(args: argparse.Namespace, x_train: pd.DataFrame, y_train: np.ndarray, x_val: pd.DataFrame, y_val: np.ndarray):
    cat_weights, lgb_weights = balanced_class_weights(y_train)
    print(
        f"train_model model={args.model} train_samples={len(x_train)} "
        f"val_samples={len(x_val)} n_features={x_train.shape[1]} "
        f"iterations={args.iterations}"
    )

    if args.model == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is not installed. Install it before running --model xgboost.") from exc

        sample_weight = None
        if args.class_weight == "balanced":
            sample_weight = build_sample_weights(y_train, lgb_weights)

        xgb_device = "cuda" if args.device == "gpu" else "cpu"
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=N_CLASSES,
            n_estimators=args.iterations,
            learning_rate=args.learning_rate,
            max_depth=args.depth,
            random_state=args.seed,
            n_jobs=-1,
            eval_metric="mlogloss",
            tree_method="hist",
            device=xgb_device,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            gamma=args.xgb_gamma,
        )
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(x_val, y_val)],
            verbose=args.log_every,
        )
        model.actual_training_device = xgb_device
        return model

    if args.model == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("catboost is not installed. Install it before running --model catboost.") from exc

        catboost_kwargs = dict(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            random_seed=args.seed,
            class_weights=cat_weights if args.class_weight == "balanced" else None,
            allow_writing_files=False,
            verbose=args.log_every,
            l2_leaf_reg=args.cat_l2_leaf_reg,
            random_strength=args.cat_random_strength,
            rsm=args.cat_rsm,
        )
        if args.device == "gpu":
            catboost_kwargs.update(task_type="GPU", devices="0")
        model = CatBoostClassifier(**catboost_kwargs)
        model.fit(
            x_train,
            y_train,
            eval_set=(x_val, y_val),
            early_stopping_rounds=args.early_stopping_rounds,
            use_best_model=True,
        )
        model.actual_training_device = "gpu:0" if args.device == "gpu" else "cpu"
        return model

    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    except ImportError as exc:
        raise ImportError("lightgbm is not installed. Install it before running --model lightgbm.") from exc

    def make_lgbm_model(device: str):
        return LGBMClassifier(
            objective="multiclass",
            num_class=N_CLASSES,
            n_estimators=args.iterations,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.depth,
            random_state=args.seed,
            class_weight=lgb_weights if args.class_weight == "balanced" else None,
            n_jobs=-1,
            verbosity=1,
            device=device,
            feature_fraction=args.lgb_feature_fraction,
            bagging_fraction=args.lgb_bagging_fraction,
            bagging_freq=args.lgb_bagging_freq,
            min_child_samples=args.lgb_min_child_samples,
            min_split_gain=args.lgb_min_split_gain,
            reg_alpha=args.lgb_lambda_l1,
            reg_lambda=args.lgb_lambda_l2,
        )

    requested_devices = ["cuda"] if args.device == "gpu" else ["cpu"]
    if args.device == "gpu":
        requested_devices.append("gpu")
    requested_devices.append("cpu")

    last_error = None
    for lgbm_device in requested_devices:
        try:
            if lgbm_device != requested_devices[0]:
                print(f"lightgbm retry with device={lgbm_device}")
            model = make_lgbm_model(lgbm_device)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(args.early_stopping_rounds), log_evaluation(args.log_every)],
            )
            model.actual_training_device = lgbm_device
            print(f"lightgbm trained with device={lgbm_device}")
            return model
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if args.device == "gpu" and lgbm_device in {"cuda", "gpu"}:
                print(f"lightgbm {lgbm_device} unavailable: {message}")
                continue
            raise

    raise last_error


def align_probabilities(probabilities: np.ndarray, classes) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 2:
        raise ValueError(f"Expected 2D probabilities, got shape={probabilities.shape}")

    aligned = np.zeros((probabilities.shape[0], N_CLASSES), dtype=np.float32)
    classes_array = np.asarray(classes)
    if classes_array.shape[0] != probabilities.shape[1]:
        raise ValueError("Mismatch between classes and probability columns")

    for src_idx, class_id in enumerate(classes_array):
        aligned[:, int(class_id)] = probabilities[:, src_idx]
    return aligned


def predict_probabilities(model, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = getattr(model, "classes_", np.arange(probabilities.shape[1]))
    return align_probabilities(probabilities, classes)


def get_actual_training_device(model, fallback: str | None = None) -> str | None:
    return getattr(model, "actual_training_device", fallback)


def save_feature_importance(model, feature_names: list[str], output_dir: Path, top_k: int = 50) -> None:
    if not hasattr(model, "feature_importances_"):
        print(f"feature importance is not available for model type={type(model).__name__}")
        return

    importances = np.asarray(model.feature_importances_, dtype=np.float64)
    if importances.shape[0] != len(feature_names):
        raise ValueError("Mismatch between feature importance length and feature names")

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False, kind="stable")
    df.to_csv(output_dir / "feature_importance.csv", index=False)

    top_df = df.head(top_k).iloc[::-1]
    fig_height = max(6.0, 0.28 * len(top_df))
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=150)
    ax.barh(top_df["feature"], top_df["importance"], color="#2563eb")
    ax.set_title(f"Feature Importance ({type(model).__name__})")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance_top.png", bbox_inches="tight")
    plt.close(fig)


def save_eval_outputs(output_dir: Path, name: str, meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> None:
    pred_df = meta.copy()
    pred_df["y_true"] = y_true.astype(int)
    pred_df["y_pred"] = y_pred.astype(int)
    pred_df["y_true_label"] = [ID_TO_LABEL[int(v)] for v in y_true]
    pred_df["y_pred_label"] = [ID_TO_LABEL[int(v)] for v in y_pred]
    pred_df.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    labels = [ID_TO_LABEL[i] for i in range(N_CLASSES)]
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = output_dir / f"{name}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)
    reordered_cm, reordered_labels, _ = reorder_confusion_matrix_for_blocks(cm, labels)
    pd.DataFrame(reordered_cm, index=reordered_labels, columns=reordered_labels).to_csv(output_dir / f"{name}_confusion_matrix_reordered.csv")
    reordered_normalized = reordered_cm.astype(np.float64)
    reordered_row_sums = reordered_normalized.sum(axis=1, keepdims=True)
    reordered_row_sums[reordered_row_sums == 0.0] = 1.0
    reordered_normalized = reordered_normalized / reordered_row_sums
    plot_confusion_matrix(reordered_cm, reordered_normalized, reordered_labels, reordered_labels, output_dir / f"{name}_confusion_matrix_reordered.png", f"{name.title()} Confusion Matrix (Reordered)", "both", 150, None)
    normalized = cm.astype(np.float64)
    row_sums = normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    normalized = normalized / row_sums
    plot_confusion_matrix(cm, normalized, labels, labels, output_dir / f"{name}_confusion_matrix.png", f"{name.title()} Confusion Matrix", "both", 150, None)

    report_text = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)), target_names=labels, digits=3, zero_division=0)
    (output_dir / f"{name}_classification_report.txt").write_text(report_text + "\n")
    report_dict = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)), target_names=labels, digits=3, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).T.round(3)
    if "support" in report_df.columns:
        report_df["support"] = report_df["support"].fillna(0).astype(int)
    report_df.to_csv(output_dir / f"{name}_classification_report.csv")

    if y_prob is not None:
        np.save(output_dir / f"{name}_probabilities.npy", y_prob.astype(np.float32))
        from approach_base.src.export_cv_eval_from_oof import save_sensor_group_eval_bundles
        save_sensor_group_eval_bundles(
            output_dir,
            name,
            pred_df,
            y_true,
            y_prob,
            normalize="true",
            annot="both",
            dpi=150,
        )


def build_test_table(
    root_dir: Path,
    args: argparse.Namespace,
    normalization_stats: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.sensor_fusion_mode != "single":
        raise ValueError("build_test_table is only available for sensor_fusion_mode=single")
    test_dir = root_dir / "test"
    inertial = np.load(test_dir / "test_inertial_data.npy", mmap_mode="r")
    meta = pd.read_csv(test_dir / "test_meta_data.csv")
    sensor_col = "inertial_sensor_location" if "inertial_sensor_location" in meta.columns else "sensor_location"
    subject_col = "subject_id" if "subject_id" in meta.columns else None
    rows = []
    for idx, row in meta.iterrows():
        sensor_key = normalize_sensor_location(row[sensor_col])
        subject_id = int(row[subject_col]) if subject_col is not None and pd.notna(row[subject_col]) else None
        window = canonicalize_left_limb_window(
            np.asarray(inertial[idx], dtype=np.float32),
            sensor_key,
            args.canonicalize_left_limb,
        )
        window = normalize_inertial_window(
            window,
            sensor_key,
            normalization_mode=args.normalization_mode,
            normalization_stats=normalization_stats,
            subject_id=subject_id,
        )
        rows.append(
            make_feature_vector(
                window,
                sensor_key,
                use_raw_features=args.use_raw_features,
                smoothing_window=args.smoothing_window,
                sensor_embedding_mode=getattr(args, "sensor_embedding_mode", "sensor"),
            )
        )
    return sanitize_feature_table(pd.DataFrame(rows), "test"), meta


def save_submission(root_dir: Path, output_path: Path, test_meta: pd.DataFrame, preds: np.ndarray) -> None:
    pred_df = pd.DataFrame({"id": test_meta["id"].astype(int), "target_value": preds.astype(int)})
    sub = pd.read_csv(root_dir / "sample_submission.csv")
    sub = sub[["id"]].merge(pred_df, on="id", how="left")
    if sub["target_value"].isna().any():
        missing = sub[sub["target_value"].isna()]["id"].tolist()[:10]
        raise ValueError(f"Missing predictions for ids: {missing}")
    sub["target_value"] = sub["target_value"].astype(int)
    sub.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    validate_sensor_fusion_configuration(args)
    args.output_dir = make_experiment_dir(args.output_dir / args.model, args.exp_name)
    print(
        f"start model={args.model} device={args.device} window_label_mode={args.window_label_mode} "
        f"window_size={args.window_size} stride={args.stride} normalization_mode={args.normalization_mode} "
        f"smoothing_window={args.smoothing_window} sensor_embedding_mode={args.sensor_embedding_mode} "
        f"build_workers={args.build_workers} "
        f"sensor_fusion_mode={args.sensor_fusion_mode} output_dir={args.output_dir} cache_base_dir={CACHE_BASE_DIR}"
    )

    records = load_inertial_records(args.root, exclude_file_id_suffix_2=args.exclude_file_id_suffix_2)
    val_subjects = set(args.val_subjects)
    train_records = [record for record in records if record["sbj_id"] not in val_subjects]
    val_records = [record for record in records if record["sbj_id"] in val_subjects]
    normalization_stats = compute_normalization_stats(train_records)
    print(f"train_subjects={sorted({r['sbj_id'] for r in train_records})}")
    print(f"val_subjects={sorted({r['sbj_id'] for r in val_records})}")

    x_train, y_train, train_meta = build_table(train_records, args, "train", normalization_stats)
    x_val, y_val, val_meta = build_table(val_records, args, "val", normalization_stats)
    feature_ranking_df = None
    selected_feature_names = list(x_train.columns)
    if args.feature_list_file is not None:
        requested_feature_names = load_feature_list(args.feature_list_file)
        x_train, x_val, selected_feature_names = apply_feature_list(x_train, x_val, requested_feature_names)
        print(
            f"feature_list_file={args.feature_list_file} selected_features={len(selected_feature_names)} "
            f"dropped_features={len(requested_feature_names) - len(selected_feature_names)}"
        )
    if args.top_k_features is not None:
        x_train, x_val, selected_feature_names, feature_ranking_df = select_top_k_features(
            x_train, y_train, x_val, args.top_k_features
        )
        print(
            f"top_k_features={args.top_k_features} selected_features={len(selected_feature_names)} "
            f"dropped_features={feature_ranking_df.shape[0] - len(selected_feature_names)}"
        )
    print(
        f"feature_tables train_shape={x_train.shape} val_shape={x_val.shape} "
        f"train_classes={np.bincount(y_train, minlength=N_CLASSES).tolist()} "
        f"val_classes={np.bincount(y_val, minlength=N_CLASSES).tolist()}"
    )
    model = train_model(args, x_train, y_train, x_val, y_val)

    train_prob = predict_probabilities(model, x_train)
    val_prob = predict_probabilities(model, x_val)
    train_pred = train_prob.argmax(axis=1).astype(int)
    val_pred = val_prob.argmax(axis=1).astype(int)
    train_metrics = compute_metrics(y_train, train_pred)
    val_metrics = compute_metrics(y_val, val_pred)
    train_loss = float(log_loss(y_train, train_prob, labels=list(range(N_CLASSES))))
    val_loss = float(log_loss(y_val, val_prob, labels=list(range(N_CLASSES))))
    print(f"train_metrics={train_metrics}")
    print(f"val_metrics={val_metrics}")
    print(f"valid_loss={val_loss:.6f}")

    save_eval_outputs(args.output_dir, "train", train_meta, y_train, train_pred, train_prob)
    save_eval_outputs(args.output_dir, "val", val_meta, y_val, val_pred, val_prob)

    actual_training_device = get_actual_training_device(model, args.device)
    with (args.output_dir / f"{args.model}_inertial.pkl").open("wb") as f:
        pickle.dump(
            {
                "model": model,
                "args": vars(args),
                "feature_columns": list(x_train.columns),
                "actual_training_device": actual_training_device,
                "normalization_mode": args.normalization_mode,
                "smoothing_window": args.smoothing_window,
            },
            f,
        )

    if feature_ranking_df is not None:
        feature_ranking_df.to_csv(args.output_dir / "feature_selection_scores.csv", index=False)
    if args.feature_list_file is not None or feature_ranking_df is not None:
        pd.DataFrame({"feature": selected_feature_names}).to_csv(args.output_dir / "selected_features.csv", index=False)

    save_feature_importance(model, list(x_train.columns), args.output_dir)

    pd.DataFrame(
        [
            {"split": "train", "actual_training_device": actual_training_device, "loss": train_loss, **train_metrics},
            {"split": "val", "actual_training_device": actual_training_device, "loss": val_loss, **val_metrics},
        ]
    ).to_csv(
        args.output_dir / "metrics.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "train_loss": train_loss,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        ]
    ).to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "train_metrics": {"loss": train_loss, **train_metrics},
                "val_metrics": {"loss": val_loss, **val_metrics},
                "actual_training_device": actual_training_device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.predict_test:
        if args.sensor_fusion_mode != "single":
            raise ValueError("predict-test is not supported with sensor_fusion_mode=parallel4")
        x_test, test_meta = build_test_table(args.root, args, normalization_stats)
        x_test = x_test.reindex(columns=selected_feature_names, fill_value=0.0)
        test_prob = predict_probabilities(model, x_test)
        test_pred = test_prob.argmax(axis=1).astype(int)
        np.save(args.output_dir / "test_probabilities.npy", test_prob.astype(np.float32))
        test_meta.to_csv(args.output_dir / "test_meta.csv", index=False)
        output_path = args.output_dir / f"submission_{args.model}_inertial.csv"
        save_submission(args.root, output_path, test_meta, test_pred)
        print(f"saved submission: {output_path}")

    save_run_metadata(
        args.output_dir,
        args=args,
        inputs={"root": args.root},
        outputs={
            "train_predictions": args.output_dir / "train_predictions.csv",
            "train_probabilities": args.output_dir / "train_probabilities.npy",
            "val_predictions": args.output_dir / "val_predictions.csv",
            "val_probabilities": args.output_dir / "val_probabilities.npy",
            "metrics": args.output_dir / "metrics.csv",
            "history": args.output_dir / "history.csv",
            "summary": args.output_dir / "summary.json",
            "model_artifact": args.output_dir / f"{args.model}_inertial.pkl",
            "test_probabilities": args.output_dir / "test_probabilities.npy" if args.predict_test else None,
            "test_meta": args.output_dir / "test_meta.csv" if args.predict_test else None,
            "submission": output_path if args.predict_test else None,
        },
        metrics={"train": {"loss": train_loss, **train_metrics}, "val": {"loss": val_loss, **val_metrics}},
        extra={"actual_training_device": actual_training_device, "selected_feature_count": len(selected_feature_names)},
    )
    print(f"saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
