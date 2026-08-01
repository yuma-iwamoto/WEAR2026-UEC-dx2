import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from approach_base.src.output_utils import make_experiment_dir, save_run_metadata
from approach_base.src.teacher_transfer_common import load_oof_bundle
from approach_base.src.train_inertial_cnn import save_eval_outputs, save_submission
from approach_XceptionTime.src.models.xceptiontime import XceptionTime
from approach_base.src.train_inertial_gbdt import (
    LABEL_TO_ID,
    N_CLASSES,
    SENSOR_COLS,
    SENSOR_TO_ID,
    assign_window_label_from_array,
    has_only_finite_sensor_windows,
    load_inertial_records,
    normalize_sensor_location,
)
from approach_base.src.video_only_common import center_crop_video_window, normalize_video_window

BASE_SENSOR_CHANNELS = 3
CNN8_CHANNELS = 8
VIDEO_DIM = 768
VIDEO_SCALAR_DIM = 56
ARM_SENSORS = {"ra", "la"}
LEG_SENSORS = {"rl", "ll"}

KD_GROUPS = {
    "null_stretching": [
        LABEL_TO_ID["null"],
        LABEL_TO_ID["stretching (triceps)"],
        LABEL_TO_ID["stretching (lunging)"],
        LABEL_TO_ID["stretching (shoulders)"],
        LABEL_TO_ID["stretching (hamstrings)"],
        LABEL_TO_ID["stretching (lumbar rotation)"],
    ],
    "jogging_family": [
        LABEL_TO_ID["jogging"],
        LABEL_TO_ID["jogging (rotating arms)"],
        LABEL_TO_ID["jogging (skipping)"],
        LABEL_TO_ID["jogging (sidesteps)"],
        LABEL_TO_ID["jogging (butt-kicks)"],
    ],
    "pushup_pair": [LABEL_TO_ID["push-ups"], LABEL_TO_ID["push-ups (complex)"]],
    "situp_pair": [LABEL_TO_ID["sit-ups"], LABEL_TO_ID["sit-ups (complex)"]],
    "lunge_pair": [LABEL_TO_ID["lunges"], LABEL_TO_ID["lunges (complex)"]],
}
KD_MARGIN_PAIRS = {
    "canonical": [
        (LABEL_TO_ID["jogging"], LABEL_TO_ID["jogging (rotating arms)"]),
        (LABEL_TO_ID["push-ups"], LABEL_TO_ID["push-ups (complex)"]),
        (LABEL_TO_ID["sit-ups"], LABEL_TO_ID["sit-ups (complex)"]),
        (LABEL_TO_ID["lunges"], LABEL_TO_ID["lunges (complex)"]),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train exp024 CNN8 + VideoMAE auxiliary mid-fusion model.")
    parser.add_argument("--root", type=Path, default=Path("/workspace/input/3rd-wear-dataset-challenge-hasca-2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/approach_base/output/exp024_cnn8_videomae_aux"))
    parser.add_argument("--val-subjects", type=int, nargs="+", default=[18, 19, 20, 21])
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--window-label-mode", choices=["purity", "majority", "strict"], default="purity")
    parser.add_argument("--min-label-purity", type=float, default=0.8)
    parser.add_argument("--sensor-keys", type=str, nargs="+", choices=list(SENSOR_COLS), default=list(SENSOR_COLS))
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--precompute-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-hz", type=int, default=30)
    parser.add_argument("--video-window-size", type=int, default=15)
    parser.add_argument("--sensor-emb-dim", type=int, default=8)
    parser.add_argument("--sensor-embedding-mode", choices=["sensor", "limb", "none"], default="sensor")
    parser.add_argument("--add-diff-5", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-10", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-20", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-30", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-bilateral-features", action=argparse.BooleanOptionalAction, default=False, help="Add left/right pair-difference channels for sensor_fusion_mode=parallel4.")
    parser.add_argument("--inertial-backbone", choices=["cnn8", "xceptiontime"], default="cnn8")
    parser.add_argument("--cnn-base-channels", type=int, default=64)
    parser.add_argument("--cnn-dropout", type=float, default=0.20)
    parser.add_argument("--xception-nf", type=int, default=16)
    parser.add_argument("--xception-kernel-size", type=int, default=40)
    parser.add_argument("--xception-adaptive-size", type=int, default=8)
    parser.add_argument("--inertial-feature-scale", type=float, default=1.0, help="Scale inertial features just before fusion. 1.0 preserves baseline behavior.")
    parser.add_argument("--inertial-feature-dropout", type=float, default=0.0, help="Dropout applied to inertial features just before fusion.")
    parser.add_argument("--video-hidden-dim", type=int, default=128)
    parser.add_argument("--video-out-dim", type=int, default=128)
    parser.add_argument("--video-dropout", type=float, default=0.50)
    parser.add_argument("--classifier-hidden", type=int, default=256)
    parser.add_argument("--classifier-dropout", type=float, default=0.49)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--early-stopping-rounds", type=int, default=10)
    parser.add_argument("--class-weight", choices=["none", "balanced", "sqrt"], default="sqrt")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler-patience", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exclude-file-id-suffix-2", action="store_true")
    parser.add_argument("--sensor-fusion-mode", choices=["single", "parallel4"], default="single")
    parser.add_argument("--limb-fusion-mode", choices=["shared", "arm_leg"], default="shared")
    parser.add_argument("--canonicalize-left-limb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-weight-file", type=Path, default=None, help="CSV with file_id,start,sensor,weight for teacher-guided sample weighting.")
    parser.add_argument("--teacher-prob-dir", type=Path, default=None, help="Teacher OOF directory for selective/margin KD.")
    parser.add_argument("--selective-kd-weight", type=float, default=0.0)
    parser.add_argument("--selective-kd-groups", type=str, nargs="*", default=[])
    parser.add_argument("--selective-kd-conf-min", type=float, default=0.0)
    parser.add_argument("--selective-kd-apply-mode", choices=["true_label"], default="true_label")
    parser.add_argument("--margin-kd-weight", type=float, default=0.0)
    parser.add_argument("--margin-kd-pairs", choices=["none", "canonical"], default="none")
    parser.add_argument("--margin-kd-conf-min", type=float, default=0.0)
    parser.add_argument("--margin-kd-apply-mode", choices=["true_label"], default="true_label")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def safe_array(x, dtype=np.float32) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=dtype), nan=0.0, posinf=0.0, neginf=0.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_class_weights(y: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.maximum(np.bincount(y, minlength=N_CLASSES).astype(np.float32), 1.0)
    weights = len(y) / (N_CLASSES * counts)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    weights = weights / weights.mean()
    return torch.tensor(weights.astype(np.float32), dtype=torch.float32)


def fused_sensor_name(sensor_keys: list[str]) -> str:
    return '+'.join(sensor_keys)


def sensor_group_index(sensor_key: str) -> int:
    if sensor_key in ARM_SENSORS:
        return 0
    if sensor_key in LEG_SENSORS:
        return 1
    raise ValueError(f"Unsupported sensor key for limb fusion: {sensor_key}")


def sensor_embedding_num_embeddings(args: argparse.Namespace) -> int:
    mode = getattr(args, "sensor_embedding_mode", "sensor")
    if mode == "sensor":
        return len(SENSOR_TO_ID) + (1 if args.sensor_fusion_mode == "parallel4" else 0)
    if mode == "limb":
        if args.sensor_fusion_mode == "parallel4":
            raise ValueError("sensor_embedding_mode=limb is only defined for sensor_fusion_mode=single")
        return 2
    if mode == "none":
        return 0
    raise ValueError(f"Unsupported sensor_embedding_mode: {mode}")


def sensor_embedding_dim(args: argparse.Namespace) -> int:
    return 0 if getattr(args, "sensor_embedding_mode", "sensor") == "none" else int(args.sensor_emb_dim)


def sensor_embedding_index(sensor_key: str, args: argparse.Namespace) -> int:
    mode = getattr(args, "sensor_embedding_mode", "sensor")
    if mode == "sensor":
        if sensor_key == fused_sensor_name(args.sensor_keys):
            return len(SENSOR_TO_ID)
        return SENSOR_TO_ID[sensor_key]
    if mode == "limb":
        return sensor_group_index(sensor_key)
    if mode == "none":
        return 0
    raise ValueError(f"Unsupported sensor_embedding_mode: {mode}")


def load_sample_weight_map(path: Path | None) -> dict[tuple[str, int, str], float]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    required = {"file_id", "start", "sensor", "weight"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sample_weight_file missing columns: {sorted(missing)}")
    weights: dict[tuple[str, int, str], float] = {}
    for row in df.itertuples(index=False):
        key = (str(getattr(row, "file_id")), int(getattr(row, "start")), str(getattr(row, "sensor")))
        weights[key] = float(getattr(row, "weight"))
    return weights


def load_teacher_prob_map(path: Path | None) -> dict[tuple[str, int, str], np.ndarray]:
    if path is None:
        return {}
    teacher_meta, teacher_prob = load_oof_bundle(path, is_teacher=True)
    teacher_meta = teacher_meta.reset_index(drop=True).copy()
    teacher_sensor_is_fused = "sensor" in teacher_meta.columns and teacher_meta["sensor"].astype(str).str.contains("+", regex=False).all()
    prob_map: dict[tuple[str, int, str], np.ndarray] = {}
    if "sensor" in teacher_meta.columns and not teacher_sensor_is_fused:
        for idx, row in teacher_meta.iterrows():
            prob_map[(str(row["file_id"]), int(row["start"]), str(row["sensor"]))] = teacher_prob[idx].astype(np.float32)
    else:
        for idx, row in teacher_meta.iterrows():
            prob_map[(str(row["file_id"]), int(row["start"]), "*")] = teacher_prob[idx].astype(np.float32)
    return prob_map


def validate_sensor_fusion_configuration(args: argparse.Namespace) -> None:
    if args.sensor_fusion_mode == 'parallel4' and list(args.sensor_keys) != list(SENSOR_COLS):
        raise ValueError(f"sensor_fusion_mode=parallel4 requires sensor_keys={list(SENSOR_COLS)}, got {list(args.sensor_keys)}")
    if args.limb_fusion_mode == "arm_leg" and args.sensor_fusion_mode != "single":
        raise ValueError("limb_fusion_mode=arm_leg requires sensor_fusion_mode=single")
    sensor_embedding_num_embeddings(args)
    if (args.selective_kd_weight > 0.0 or args.margin_kd_weight > 0.0) and args.teacher_prob_dir is None:
        raise ValueError("--teacher-prob-dir is required when selective or margin KD is enabled")
    if args.selective_kd_weight > 0.0 and not args.selective_kd_groups:
        raise ValueError("--selective-kd-groups is required when --selective-kd-weight > 0")
    if args.margin_kd_weight > 0.0 and args.margin_kd_pairs == "none":
        raise ValueError("--margin-kd-pairs must be set when --margin-kd-weight > 0")
    unknown_groups = sorted(set(args.selective_kd_groups) - set(KD_GROUPS))
    if unknown_groups:
        raise ValueError(f"Unknown selective_kd_groups: {unknown_groups}; choices={sorted(KD_GROUPS)}")


def enabled_diff_lags(args: argparse.Namespace | None) -> list[int]:
    if args is None:
        return []
    return [lag for lag in (5, 10, 20, 30) if getattr(args, f"add_diff_{lag}", False)]


def cnn8_channels(args: argparse.Namespace | None) -> int:
    return CNN8_CHANNELS + 4 * len(enabled_diff_lags(args))


def bilateral_channels(args: argparse.Namespace) -> int:
    if not getattr(args, "add_bilateral_features", False):
        return 0
    # Per pair: abs(R-L) xyz, abs(diff(R-L)) xyz, abs(diff2(R-L)) xyz, abs(|R|-|L|) scalar.
    return 10 * 2


def cnn_input_channels(args: argparse.Namespace) -> int:
    if args.sensor_fusion_mode == 'parallel4':
        return cnn8_channels(args) * len(args.sensor_keys) + bilateral_channels(args)
    return cnn8_channels(args)


def compute_lagged_diff(window: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag}")
    diff = np.zeros_like(window, dtype=np.float32)
    if window.shape[0] > lag:
        diff[lag:] = window[lag:] - window[:-lag]
    return diff


def make_bilateral_sensor_inputs(sensor_windows: list[np.ndarray], args: argparse.Namespace) -> list[np.ndarray]:
    windows = dict(zip(args.sensor_keys, sensor_windows))
    derived_inputs = []
    for right_key, left_key in (("ra", "la"), ("rl", "ll")):
        right = safe_array(windows[right_key])
        left = safe_array(windows[left_key])
        right_minus_left = right - left
        abs_diff = np.zeros_like(right_minus_left, dtype=np.float32)
        abs_diff[1:] = np.abs(right_minus_left[1:] - right_minus_left[:-1])
        abs_second_diff = np.zeros_like(right_minus_left, dtype=np.float32)
        abs_second_diff[2:] = np.abs(right_minus_left[2:] - 2.0 * right_minus_left[1:-1] + right_minus_left[:-2])
        right_mag = np.linalg.norm(right, axis=1, keepdims=True)
        left_mag = np.linalg.norm(left, axis=1, keepdims=True)
        magnitude_diff = np.abs(right_mag - left_mag)
        derived_inputs.extend([
            np.abs(right_minus_left).T,
            abs_diff.T,
            abs_second_diff.T,
            magnitude_diff.T,
        ])
    return [safe_array(x) for x in derived_inputs]


def make_cnn8_inertial(window: np.ndarray, args: argparse.Namespace | None = None, eps: float = 1e-6) -> np.ndarray:
    window = safe_array(window)
    if window.ndim != 2 or window.shape[1] != BASE_SENSOR_CHANNELS:
        raise ValueError(f"Unexpected inertial window shape: {window.shape}")
    x, y, z = window[:, 0], window[:, 1], window[:, 2]
    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2 + eps)
    diff = np.diff(window, axis=0, prepend=window[:1])
    abs_diff = np.abs(diff)
    diff_mag = np.linalg.norm(diff, axis=1)
    channels = [x, y, z, mag, abs_diff[:, 0], abs_diff[:, 1], abs_diff[:, 2], diff_mag]
    for lag in enabled_diff_lags(args):
        lag_diff = compute_lagged_diff(window, lag)
        abs_lag_diff = np.abs(lag_diff)
        lag_diff_mag = np.linalg.norm(lag_diff, axis=1)
        channels.extend([abs_lag_diff[:, 0], abs_lag_diff[:, 1], abs_lag_diff[:, 2], lag_diff_mag])
    return safe_array(np.stack(channels, axis=0))


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


def _row_cos(lhs: np.ndarray, rhs: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    lhs = safe_array(lhs)
    rhs = safe_array(rhs)
    return ((lhs * rhs).sum(axis=1) / (np.linalg.norm(lhs, axis=1) * np.linalg.norm(rhs, axis=1) + eps)).astype(np.float32)


def _stats_1d(x: np.ndarray) -> np.ndarray:
    x = safe_array(x)
    q10, q25, q50, q75, q90 = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
    return safe_array([
        x.mean(), x.std(), x.min(), x.max(), x.max() - x.min(),
        q10, q25, q50, q75, q90, q75 - q25,
        np.sqrt(np.mean(x ** 2)), np.sum(x ** 2), x[0], x[-1], x[-1] - x[0],
    ])


def make_video_scalar_features(video_window: np.ndarray) -> np.ndarray:
    video_window = safe_array(video_window)
    first, last = video_window[0], video_window[-1]
    first5 = video_window[:5].mean(axis=0)
    mid5 = video_window[5:10].mean(axis=0)
    last5 = video_window[10:].mean(axis=0)
    frame_norm = np.linalg.norm(video_window, axis=1)
    diff_norm = np.linalg.norm(np.diff(video_window, axis=0), axis=1)
    frame_cos = _row_cos(video_window[:-1], video_window[1:])
    scalar = np.asarray([
        np.linalg.norm(last - first), _row_cos(first[None, :], last[None, :])[0],
        np.linalg.norm(mid5 - first5), np.linalg.norm(last5 - mid5), np.linalg.norm(last5 - first5),
        _row_cos(first5[None, :], mid5[None, :])[0], _row_cos(mid5[None, :], last5[None, :])[0],
        _row_cos(first5[None, :], last5[None, :])[0],
    ], dtype=np.float32)
    return safe_array(np.concatenate([scalar, _stats_1d(frame_norm), _stats_1d(diff_norm), _stats_1d(frame_cos)]))


def make_video_aggregate_features(video_window: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    video_window = safe_array(video_window)
    return (
        video_window.mean(axis=0).astype(np.float32),
        video_window.std(axis=0).astype(np.float32),
        (video_window[-5:].mean(axis=0) - video_window[:5].mean(axis=0)).astype(np.float32),
    )


def scalar_or_channel_std(std: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.maximum(std.astype(np.float32), eps)


def fix_test_inertial_window(window: np.ndarray, window_size: int) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    if window.ndim != 2:
        raise ValueError(f"Unexpected inertial ndim: {window.shape}")
    if window.shape[1] == BASE_SENSOR_CHANNELS:
        pass
    elif window.shape[0] == BASE_SENSOR_CHANNELS:
        window = window.T
    else:
        raise ValueError(f"Unexpected inertial shape: {window.shape}")
    if window.shape[0] == window_size:
        return window
    out = np.zeros((window_size, BASE_SENSOR_CHANNELS), dtype=np.float32)
    n = min(window_size, window.shape[0])
    out[:n] = window[:n]
    return out


class Exp024Dataset(Dataset):
    def __init__(self, records: list[dict], args: argparse.Namespace, split_name: str):
        self.records = records
        self.args = args
        self.samples = []
        self.cnn_mean = self.cnn_std = self.scalar_mean = self.scalar_std = None
        self.x_cnn = None
        self.video_feature_indices = None
        self.vi_mean = self.vi_std = self.vi_delta5 = self.video_scalar = None
        self.sample_weight_map = load_sample_weight_map(args.sample_weight_file)
        self.teacher_prob_map = load_teacher_prob_map(args.teacher_prob_dir)
        skipped_label = skipped_sensor = skipped_video = 0
        for record_idx, rec in enumerate(tqdm(records, desc=f"build_{split_name}", unit="record")):
            starts = list(range(0, len(rec["df"]) - args.window_size + 1, args.stride))
            if args.max_windows_per_record is not None and len(starts) > args.max_windows_per_record:
                starts = sorted(np.random.choice(starts, size=args.max_windows_per_record, replace=False).tolist())
            for start in starts:
                y = assign_window_label_from_array(rec["label_ids"], start, args.window_size, args.window_label_mode, args.min_label_purity)
                if y is None:
                    skipped_label += 1
                    continue
                if not has_only_finite_sensor_windows(rec["sensor_arrays"], args.sensor_keys, start, args.window_size):
                    skipped_sensor += len(args.sensor_keys)
                    continue
                video_window = center_crop_video_window(
                    np.asarray(rec["video"]), start, args.window_size, video_hz=args.video_hz, video_window_size=args.video_window_size
                )
                if not np.isfinite(video_window).all():
                    skipped_video += 1
                    continue
                if args.sensor_fusion_mode == "parallel4":
                    self.samples.append({
                        "record_idx": record_idx,
                        "start": start,
                        "sensor_key": fused_sensor_name(args.sensor_keys),
                        "sensor": sensor_embedding_index(fused_sensor_name(args.sensor_keys), args),
                        "y": int(y),
                    })
                else:
                    for sensor_key in args.sensor_keys:
                        self.samples.append({"record_idx": record_idx, "start": start, "sensor_key": sensor_key, "sensor": sensor_embedding_index(sensor_key, args), "y": int(y)})
        if self.sample_weight_map:
            matched = 0
            weights = []
            for sample in self.samples:
                key = (str(self.records[sample["record_idx"]]["file_id"]), int(sample["start"]), str(sample["sensor_key"]))
                weight = float(self.sample_weight_map.get(key, 1.0))
                sample["weight"] = weight
                weights.append(weight)
                matched += int(key in self.sample_weight_map)
            print(f"[{split_name}] sample_weights matched={matched}/{len(self.samples)} mean={float(np.mean(weights)):.4f} min={float(np.min(weights)):.4f} max={float(np.max(weights)):.4f}")
        if self.teacher_prob_map:
            matched = 0
            for sample in self.samples:
                rec = self.records[sample["record_idx"]]
                key = (str(rec["file_id"]), int(sample["start"]), str(sample["sensor_key"]))
                window_key = (str(rec["file_id"]), int(sample["start"]), "*")
                teacher_prob = self.teacher_prob_map.get(key, self.teacher_prob_map.get(window_key))
                if teacher_prob is not None:
                    sample["teacher_prob"] = teacher_prob
                    matched += 1
            print(f"[{split_name}] teacher_probs matched={matched}/{len(self.samples)}")
        print(f"[{split_name}] samples={len(self.samples)} skipped_label={skipped_label} skipped_sensor={skipped_sensor} skipped_video={skipped_video}")
        if not self.samples:
            raise ValueError(f"No samples built for {split_name}")
        if getattr(args, "precompute_features", True):
            self.precompute_all_features(split_name)

    def __len__(self) -> int:
        return len(self.samples)

    def set_normalization(self, cnn_mean, cnn_std, scalar_mean, scalar_std) -> None:
        self.cnn_mean = cnn_mean.astype(np.float32)
        self.cnn_std = scalar_or_channel_std(cnn_std)
        self.scalar_mean = scalar_mean.astype(np.float32)
        self.scalar_std = scalar_or_channel_std(scalar_std)

    def _make_x_cnn(self, sample: dict) -> np.ndarray:
        rec = self.records[sample["record_idx"]]
        start = sample["start"]
        if self.args.sensor_fusion_mode == "parallel4":
            sensor_windows = [
                canonicalize_left_limb_window(
                    rec["sensor_arrays"][sensor_key][start:start + self.args.window_size],
                    sensor_key,
                    self.args.canonicalize_left_limb,
                )
                for sensor_key in self.args.sensor_keys
            ]
            features = [make_cnn8_inertial(window, self.args) for window in sensor_windows]
            if self.args.add_bilateral_features:
                features.extend(make_bilateral_sensor_inputs(sensor_windows, self.args))
            return np.concatenate(features, axis=0).astype(np.float32, copy=False)
        sensor_key = sample["sensor_key"]
        sensor_window = canonicalize_left_limb_window(
            rec["sensor_arrays"][sensor_key][start:start + self.args.window_size],
            sensor_key,
            self.args.canonicalize_left_limb,
        )
        return make_cnn8_inertial(sensor_window, self.args)

    def _make_video_features(self, sample: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rec = self.records[sample["record_idx"]]
        start = sample["start"]
        video_window = normalize_video_window(center_crop_video_window(
            np.asarray(rec["video"]), start, self.args.window_size, video_hz=self.args.video_hz, video_window_size=self.args.video_window_size
        ))
        vi_mean, vi_std, vi_delta5 = make_video_aggregate_features(video_window)
        video_scalar = make_video_scalar_features(video_window)
        return vi_mean, vi_std, vi_delta5, video_scalar

    def precompute_all_features(self, split_name: str) -> None:
        x_cnn = np.empty((len(self.samples), cnn_input_channels(self.args), self.args.window_size), dtype=np.float32)
        video_feature_indices = np.empty(len(self.samples), dtype=np.int64)
        video_key_to_idx: dict[tuple[int, int], int] = {}
        vi_mean_list = []
        vi_std_list = []
        vi_delta5_list = []
        video_scalar_list = []

        for idx, sample in enumerate(tqdm(self.samples, desc=f"precompute_{split_name}", unit="sample")):
            x_cnn[idx] = self._make_x_cnn(sample)
            video_key = (int(sample["record_idx"]), int(sample["start"]))
            video_idx = video_key_to_idx.get(video_key)
            if video_idx is None:
                video_idx = len(vi_mean_list)
                video_key_to_idx[video_key] = video_idx
                vi_mean, vi_std, vi_delta5, video_scalar = self._make_video_features(sample)
                vi_mean_list.append(vi_mean)
                vi_std_list.append(vi_std)
                vi_delta5_list.append(vi_delta5)
                video_scalar_list.append(video_scalar)
            video_feature_indices[idx] = video_idx

        self.x_cnn = x_cnn
        self.video_feature_indices = video_feature_indices
        self.vi_mean = np.stack(vi_mean_list).astype(np.float32, copy=False)
        self.vi_std = np.stack(vi_std_list).astype(np.float32, copy=False)
        self.vi_delta5 = np.stack(vi_delta5_list).astype(np.float32, copy=False)
        self.video_scalar = np.stack(video_scalar_list).astype(np.float32, copy=False)
        print(
            f"[{split_name}] precomputed x_cnn={self.x_cnn.shape} "
            f"video_features={self.vi_mean.shape[0]}"
        )

    def _raw_features(self, idx: int):
        if self.x_cnn is not None:
            video_idx = int(self.video_feature_indices[idx])
            return (
                self.x_cnn[idx],
                self.vi_mean[video_idx],
                self.vi_std[video_idx],
                self.vi_delta5[video_idx],
                self.video_scalar[video_idx],
            )
        sample = self.samples[idx]
        x_cnn = self._make_x_cnn(sample)
        vi_mean, vi_std, vi_delta5, video_scalar = self._make_video_features(sample)
        return x_cnn, vi_mean, vi_std, vi_delta5, video_scalar

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        x_cnn, vi_mean, vi_std, vi_delta5, video_scalar = self._raw_features(idx)
        if self.cnn_mean is not None:
            x_cnn = (x_cnn - self.cnn_mean) / self.cnn_std
        if self.scalar_mean is not None:
            video_scalar = (video_scalar - self.scalar_mean) / self.scalar_std
        return {
            "id": torch.tensor(idx, dtype=torch.long),
            "x_cnn": torch.tensor(x_cnn, dtype=torch.float32),
            "vi_mean": torch.tensor(vi_mean, dtype=torch.float32),
            "vi_std": torch.tensor(vi_std, dtype=torch.float32),
            "vi_delta5": torch.tensor(vi_delta5, dtype=torch.float32),
            "video_scalar": torch.tensor(video_scalar, dtype=torch.float32),
            "sensor": torch.tensor(sample["sensor"], dtype=torch.long),
            "sensor_group": torch.tensor(-1 if sample["sensor_key"] == fused_sensor_name(self.args.sensor_keys) else sensor_group_index(sample["sensor_key"]), dtype=torch.long),
            "sample_weight": torch.tensor(float(sample.get("weight", 1.0)), dtype=torch.float32),
            "teacher_prob": torch.tensor(sample.get("teacher_prob", np.zeros(N_CLASSES, dtype=np.float32)), dtype=torch.float32),
            "has_teacher_prob": torch.tensor("teacher_prob" in sample, dtype=torch.bool),
            "y": torch.tensor(sample["y"], dtype=torch.long),
        }

    def metadata_frame(self) -> pd.DataFrame:
        rows = []
        for idx, sample in enumerate(self.samples):
            rec = self.records[sample["record_idx"]]
            rows.append({
                "id": idx, "file_id": rec["file_id"], "sbj_id": rec["sbj_id"], "start": sample["start"],
                "sensor": sample["sensor_key"], "sensor_group": -1 if sample["sensor_key"] == fused_sensor_name(self.args.sensor_keys) else sensor_group_index(sample["sensor_key"]), "y_true": sample["y"],
                "sample_weight": float(sample.get("weight", 1.0)),
            })
        return pd.DataFrame(rows)


class Exp024TestDataset(Dataset):
    def __init__(self, root_dir: Path, args: argparse.Namespace, cnn_mean, cnn_std, scalar_mean, scalar_std):
        test_dir = root_dir / "test"
        self.inertial = np.load(test_dir / "test_inertial_data.npy", mmap_mode="r")
        self.video = np.load(test_dir / "test_videomae_data.npy", mmap_mode="r")
        self.meta = pd.read_csv(test_dir / "test_meta_data.csv")
        self.args = args
        self.cnn_mean = cnn_mean.astype(np.float32)
        self.cnn_std = scalar_or_channel_std(cnn_std)
        self.scalar_mean = scalar_mean.astype(np.float32)
        self.scalar_std = scalar_or_channel_std(scalar_std)
        if len(self.inertial) != len(self.video) or len(self.inertial) != len(self.meta):
            raise ValueError("test inertial/video/meta length mismatch")

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.meta.iloc[idx]
        sensor_col = "inertial_sensor_location" if "inertial_sensor_location" in self.meta.columns else "sensor_location"
        sensor_key = normalize_sensor_location(row[sensor_col])
        sensor_id = sensor_embedding_index(sensor_key, self.args)
        inertial_window = canonicalize_left_limb_window(
            fix_test_inertial_window(self.inertial[idx], self.args.window_size),
            sensor_key,
            self.args.canonicalize_left_limb,
        )
        x_cnn = (make_cnn8_inertial(inertial_window, self.args) - self.cnn_mean) / self.cnn_std
        video_window = normalize_video_window(self.video[idx])
        vi_mean, vi_std, vi_delta5 = make_video_aggregate_features(video_window)
        video_scalar = (make_video_scalar_features(video_window) - self.scalar_mean) / self.scalar_std
        return {
            "id": torch.tensor(int(row["id"]), dtype=torch.long),
            "x_cnn": torch.tensor(x_cnn, dtype=torch.float32),
            "vi_mean": torch.tensor(vi_mean, dtype=torch.float32),
            "vi_std": torch.tensor(vi_std, dtype=torch.float32),
            "vi_delta5": torch.tensor(vi_delta5, dtype=torch.float32),
            "video_scalar": torch.tensor(video_scalar, dtype=torch.float32),
            "sensor": torch.tensor(sensor_id, dtype=torch.long),
            "sensor_group": torch.tensor(sensor_group_index(sensor_key), dtype=torch.long),
        }


def compute_normalization_stats(dataset: Exp024Dataset):
    if dataset.x_cnn is not None:
        cnn_mean = dataset.x_cnn.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
        cnn_std = dataset.x_cnn.std(axis=(0, 2), dtype=np.float64).astype(np.float32)[:, None]
        scalar_by_sample = dataset.video_scalar[dataset.video_feature_indices]
        scalar_mean = scalar_by_sample.mean(axis=0, dtype=np.float64).astype(np.float32)
        scalar_std = scalar_by_sample.std(axis=0, dtype=np.float64).astype(np.float32)
        return cnn_mean, cnn_std, scalar_mean, scalar_std

    sum_cnn = np.zeros((cnn_input_channels(dataset.args), 1), dtype=np.float64)
    sumsq_cnn = np.zeros((cnn_input_channels(dataset.args), 1), dtype=np.float64)
    sum_scalar = np.zeros(VIDEO_SCALAR_DIM, dtype=np.float64)
    sumsq_scalar = np.zeros(VIDEO_SCALAR_DIM, dtype=np.float64)
    n_cnn = n_scalar = 0
    for idx in tqdm(range(len(dataset)), desc="fit_normalization", unit="sample"):
        x_cnn, _, _, _, video_scalar = dataset._raw_features(idx)
        sum_cnn += x_cnn.sum(axis=1, keepdims=True)
        sumsq_cnn += (x_cnn ** 2).sum(axis=1, keepdims=True)
        n_cnn += x_cnn.shape[1]
        sum_scalar += video_scalar
        sumsq_scalar += video_scalar ** 2
        n_scalar += 1
    cnn_mean = (sum_cnn / max(n_cnn, 1)).astype(np.float32)
    cnn_std = np.sqrt(np.maximum((sumsq_cnn / max(n_cnn, 1)) - cnn_mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    scalar_mean = (sum_scalar / max(n_scalar, 1)).astype(np.float32)
    scalar_std = np.sqrt(np.maximum((sumsq_scalar / max(n_scalar, 1)) - scalar_mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    return cnn_mean, cnn_std, scalar_mean, scalar_std


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_video_projection(args: argparse.Namespace) -> nn.ModuleDict:
    vh = args.video_hidden_dim
    return nn.ModuleDict({
        "mean": nn.Sequential(nn.Linear(VIDEO_DIM, vh), nn.LayerNorm(vh), nn.ReLU(), nn.Dropout(args.video_dropout)),
        "std": nn.Sequential(nn.Linear(VIDEO_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(args.video_dropout)),
        "delta5": nn.Sequential(nn.Linear(VIDEO_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(args.video_dropout)),
        "scalar": nn.Sequential(nn.Linear(VIDEO_SCALAR_DIM, vh // 2), nn.LayerNorm(vh // 2), nn.ReLU(), nn.Dropout(args.video_dropout)),
        "out": nn.Sequential(nn.Linear(vh + vh // 2 + vh // 2 + vh // 2, args.video_out_dim), nn.LayerNorm(args.video_out_dim), nn.ReLU(), nn.Dropout(args.video_dropout)),
    })


def project_video_features(projection: nn.ModuleDict, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return projection["out"](torch.cat([
        projection["mean"](batch["vi_mean"]),
        projection["std"](batch["vi_std"]),
        projection["delta5"](batch["vi_delta5"]),
        projection["scalar"](batch["video_scalar"]),
    ], dim=1))


def make_fusion_classifier(args: argparse.Namespace, total_dim: int) -> nn.Sequential:
    h = args.classifier_hidden
    return nn.Sequential(
        nn.Linear(total_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(args.classifier_dropout),
        nn.Linear(h, h // 2), nn.BatchNorm1d(h // 2), nn.ReLU(), nn.Dropout(args.classifier_dropout),
        nn.Linear(h // 2, N_CLASSES),
    )


class CNN8VideoAuxNet(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        base = args.cnn_base_channels
        vh = args.video_hidden_dim
        self.inertial_backbone = args.inertial_backbone
        self.inertial_feature_dim = base * 2 * 2
        if self.inertial_backbone == "cnn8":
            self.cnn = nn.Sequential(
                ConvBlock(cnn_input_channels(args), base, 5, args.cnn_dropout), nn.MaxPool1d(2),
                ConvBlock(base, base * 2, 3, args.cnn_dropout), nn.MaxPool1d(2),
                ConvBlock(base * 2, base * 2, 3, args.cnn_dropout),
            )
            self.xception = None
        elif self.inertial_backbone == "xceptiontime":
            self.cnn = None
            self.xception = XceptionTime(
                c_in=cnn_input_channels(args),
                c_out=self.inertial_feature_dim,
                nf=args.xception_nf,
                adaptive_size=args.xception_adaptive_size,
                residual=True,
                bottleneck=True,
                ks=args.xception_kernel_size,
            )
        else:
            raise ValueError(f"Unsupported inertial_backbone: {self.inertial_backbone}")
        self.limb_fusion_mode = args.limb_fusion_mode
        self.inertial_feature_scale = float(args.inertial_feature_scale)
        self.inertial_feature_dropout = nn.Dropout(float(args.inertial_feature_dropout))
        sensor_feat_dim = sensor_embedding_dim(args)
        num_sensor_embeddings = sensor_embedding_num_embeddings(args)
        self.sensor_emb = nn.Embedding(num_sensor_embeddings, sensor_feat_dim) if sensor_feat_dim > 0 else None
        total_dim = self.inertial_feature_dim + args.video_out_dim + sensor_feat_dim
        self.video_projection = make_video_projection(args)
        self.classifier = make_fusion_classifier(args, total_dim)
        if self.limb_fusion_mode == "arm_leg":
            self.arm_video_projection = make_video_projection(args)
            self.leg_video_projection = make_video_projection(args)
            self.arm_classifier = make_fusion_classifier(args, total_dim)
            self.leg_classifier = make_fusion_classifier(args, total_dim)
        else:
            self.arm_video_projection = None
            self.leg_video_projection = None
            self.arm_classifier = None
            self.leg_classifier = None

    def encode_inertial(self, x_cnn: torch.Tensor) -> torch.Tensor:
        if self.inertial_backbone == "cnn8":
            h = self.cnn(x_cnn)
            return torch.cat([F.adaptive_avg_pool1d(h, 1).squeeze(-1), F.adaptive_max_pool1d(h, 1).squeeze(-1)], dim=1)
        return self.xception(x_cnn)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        inertial_feat = self.encode_inertial(batch["x_cnn"])
        inertial_feat = self.inertial_feature_dropout(inertial_feat) * self.inertial_feature_scale
        sensor_feat = self.sensor_emb(batch["sensor"]) if self.sensor_emb is not None else None
        if self.limb_fusion_mode == "shared":
            video_feat = project_video_features(self.video_projection, batch)
            features = [inertial_feat, video_feat] if sensor_feat is None else [inertial_feat, video_feat, sensor_feat]
            return self.classifier(torch.cat(features, dim=1))

        arm_video_feat = project_video_features(self.arm_video_projection, batch)
        leg_video_feat = project_video_features(self.leg_video_projection, batch)
        arm_features = [inertial_feat, arm_video_feat] if sensor_feat is None else [inertial_feat, arm_video_feat, sensor_feat]
        leg_features = [inertial_feat, leg_video_feat] if sensor_feat is None else [inertial_feat, leg_video_feat, sensor_feat]
        arm_logits = self.arm_classifier(torch.cat(arm_features, dim=1))
        leg_logits = self.leg_classifier(torch.cat(leg_features, dim=1))
        is_arm = batch["sensor_group"].eq(0).unsqueeze(1)
        return torch.where(is_arm, arm_logits, leg_logits)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def masked_group_kl(logits: torch.Tensor, teacher_prob: torch.Tensor, mask: torch.Tensor, group_idx: list[int]) -> torch.Tensor:
    if not mask.any():
        return logits.new_tensor(0.0)
    idx = torch.tensor(group_idx, device=logits.device, dtype=torch.long)
    log_p_s = F.log_softmax(logits.index_select(1, idx), dim=1)
    p_t = teacher_prob.index_select(1, idx)
    p_t = p_t / torch.clamp(p_t.sum(dim=1, keepdim=True), min=1e-8)
    loss = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=1)
    return loss[mask].mean()


def distillation_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> torch.Tensor:
    if args.selective_kd_weight <= 0.0 and args.margin_kd_weight <= 0.0:
        return logits.new_tensor(0.0)
    teacher_prob = batch["teacher_prob"]
    has_teacher = batch["has_teacher_prob"].bool()
    teacher_conf = teacher_prob.max(dim=1).values
    y = batch["y"]
    loss = logits.new_tensor(0.0)

    if args.selective_kd_weight > 0.0:
        for group_name in args.selective_kd_groups:
            group_idx = KD_GROUPS[group_name]
            group_tensor = torch.tensor(group_idx, device=logits.device, dtype=torch.long)
            in_group = (y.unsqueeze(1) == group_tensor.unsqueeze(0)).any(dim=1)
            mask = has_teacher & in_group & (teacher_conf >= args.selective_kd_conf_min)
            loss = loss + args.selective_kd_weight * masked_group_kl(logits, teacher_prob, mask, group_idx)

    if args.margin_kd_weight > 0.0 and args.margin_kd_pairs != "none":
        z_t = torch.log(torch.clamp(teacher_prob, min=1e-8))
        z_t = z_t - z_t.mean(dim=1, keepdim=True)
        pair_losses = []
        for a, b in KD_MARGIN_PAIRS[args.margin_kd_pairs]:
            mask = has_teacher & ((y == a) | (y == b)) & (teacher_conf >= args.margin_kd_conf_min)
            if mask.any():
                margin_t = z_t[:, a] - z_t[:, b]
                margin_s = logits[:, a] - logits[:, b]
                pair_losses.append(F.mse_loss(margin_s[mask], margin_t[mask].detach()))
        if pair_losses:
            loss = loss + args.margin_kd_weight * torch.stack(pair_losses).mean()
    return loss


def train_one_epoch(model, loader, optimizer, criterion, device, log_every: int, args: argparse.Namespace) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="train", unit="batch"), start=1):
        batch = move_batch(batch, device)
        logits = model(batch)
        loss_values = criterion(logits, batch["y"])
        if loss_values.ndim == 0:
            loss = loss_values
        else:
            weights = batch.get("sample_weight")
            if weights is None:
                loss = loss_values.mean()
            else:
                loss = (loss_values * weights).sum() / torch.clamp(weights.sum(), min=1e-6)
        loss = loss + distillation_loss(logits, batch, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        if batch_idx % log_every == 0:
            print(f"train_batch={batch_idx} loss={loss.item():.6f}")
    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate_probabilities_and_loss(model, loader, criterion, device):
    model.eval()
    all_ids, all_prob, all_y = [], [], []
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="valid", unit="batch"):
        ids = batch["id"].numpy()
        y = batch.get("y")
        batch = move_batch(batch, device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        all_ids.append(ids)
        all_prob.append(prob)
        if y is not None:
            loss_values = criterion(logits, batch["y"])
            total_loss += float(loss_values.mean().item() if loss_values.ndim > 0 else loss_values.item())
            n_batches += 1
            all_y.append(y.numpy())
    return np.concatenate(all_ids), np.concatenate(all_prob), np.concatenate(all_y) if all_y else None, total_loss / max(n_batches, 1)


@torch.inference_mode()
def predict_probabilities(model, loader, device):
    return evaluate_probabilities_and_loss(model, loader, nn.CrossEntropyLoss(), device)[:3]


def load_records_with_video(root: Path, exclude_file_id_suffix_2: bool) -> list[dict]:
    records = load_inertial_records(root, exclude_file_id_suffix_2=exclude_file_id_suffix_2)
    video_dir = root / "train" / "videomae_feat"
    for rec in records:
        video_path = video_dir / f"{rec['file_id']}.npy"
        if not video_path.exists():
            raise FileNotFoundError(f"Missing VideoMAE features for {rec['file_id']}: {video_path}")
        rec["video_path"] = video_path
        rec["video"] = np.load(video_path, mmap_mode="r")
    return records


def main() -> None:
    args = parse_args()
    validate_sensor_fusion_configuration(args)
    if args.predict_test and args.sensor_fusion_mode == "parallel4":
        raise ValueError("predict-test is not supported with sensor_fusion_mode=parallel4")
    seed_everything(args.seed)
    args.output_dir = make_experiment_dir(args.output_dir, args.exp_name)
    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(
        f"start exp024_cnn8_videomae_aux device={device.type} window_size={args.window_size} "
        f"stride={args.stride} batch_size={args.batch_size} precompute_features={args.precompute_features} "
        f"inertial_backbone={args.inertial_backbone} xception_nf={args.xception_nf} sensor_fusion_mode={args.sensor_fusion_mode} "
        f"sensor_embedding_mode={args.sensor_embedding_mode} diff_lags={enabled_diff_lags(args)} "
        f"inertial_feature_scale={args.inertial_feature_scale} inertial_feature_dropout={args.inertial_feature_dropout} "
        f"selective_kd_weight={args.selective_kd_weight} margin_kd_weight={args.margin_kd_weight}"
    )

    records = load_records_with_video(args.root, args.exclude_file_id_suffix_2)
    val_subjects = set(args.val_subjects)
    train_records = [record for record in records if record["sbj_id"] not in val_subjects]
    val_records = [record for record in records if record["sbj_id"] in val_subjects]
    print(f"train_subjects={sorted({record['sbj_id'] for record in train_records})}")
    print(f"val_subjects={sorted({record['sbj_id'] for record in val_records})}")

    train_ds = Exp024Dataset(train_records, args, "train")
    val_ds = Exp024Dataset(val_records, args, "val")
    train_meta = train_ds.metadata_frame()
    val_meta = val_ds.metadata_frame()
    cnn_mean, cnn_std, scalar_mean, scalar_std = compute_normalization_stats(train_ds)
    train_ds.set_normalization(cnn_mean, cnn_std, scalar_mean, scalar_std)
    val_ds.set_normalization(cnn_mean, cnn_std, scalar_mean, scalar_std)
    np.savez(args.output_dir / "feature_normalization.npz", cnn_mean=cnn_mean, cnn_std=cnn_std, scalar_mean=scalar_mean, scalar_std=scalar_std)
    print(f"dataset train_samples={len(train_ds)} val_samples={len(val_ds)} train_classes={np.bincount(train_meta['y_true'].to_numpy(dtype=np.int64), minlength=N_CLASSES).tolist()} val_classes={np.bincount(val_meta['y_true'].to_numpy(dtype=np.int64), minlength=N_CLASSES).tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = CNN8VideoAuxNet(args).to(device)
    class_weights = compute_class_weights(train_meta["y_true"].to_numpy(dtype=np.int64), args.class_weight)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None, label_smoothing=args.label_smoothing, reduction="none" if args.sample_weight_file is not None else "mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=args.scheduler_factor, patience=args.scheduler_patience)

    best_score = -1.0
    best_epoch = None
    best_loss = None
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.log_every, args)
        _, val_prob, y_val, val_loss = evaluate_probabilities_and_loss(model, val_loader, criterion, device)
        val_metrics = compute_metrics(y_val, val_prob.argmax(axis=1))
        scheduler.step(val_metrics["macro_f1"])
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, **val_metrics})
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_macro_f1={val_metrics['macro_f1']:.6f} lr={lr:.2e}")
        if val_metrics["macro_f1"] > best_score:
            best_score = val_metrics["macro_f1"]
            best_epoch = epoch
            best_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            bad_epochs = 0
            torch.save({"model_state_dict": best_state, "args": vars(args), "best_epoch": best_epoch, "best_macro_f1": best_score}, args.output_dir / "exp024_cnn8_videomae_aux_best.pt")
            print(f"saved best epoch={epoch} macro_f1={best_score:.6f}")
        else:
            bad_epochs += 1
            print(f"bad_epochs={bad_epochs}/{args.early_stopping_rounds}")
        if bad_epochs >= args.early_stopping_rounds:
            print("early stopping")
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")

    model.load_state_dict(best_state)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    _, train_prob, y_train = predict_probabilities(model, train_eval_loader, device)
    _, val_prob, y_val = predict_probabilities(model, val_loader, device)
    train_metrics = compute_metrics(y_train, train_prob.argmax(axis=1))
    val_metrics = compute_metrics(y_val, val_prob.argmax(axis=1))
    print(f"final_best epoch={best_epoch} train_metrics={train_metrics} val_metrics={val_metrics}")

    save_eval_outputs(args.output_dir, "train", train_meta, train_prob)
    save_eval_outputs(args.output_dir, "val", val_meta, val_prob)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"best_epoch": best_epoch, "best_valid_loss": float(best_loss), "train_metrics": train_metrics, "val_metrics": val_metrics}, indent=2) + "\n")

    output_path = None
    if args.predict_test:
        test_ds = Exp024TestDataset(args.root, args, cnn_mean, cnn_std, scalar_mean, scalar_std)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        test_ids, test_prob, _ = predict_probabilities(model, test_loader, device)
        np.save(args.output_dir / "test_probabilities.npy", test_prob.astype(np.float32))
        test_ds.meta.to_csv(args.output_dir / "test_meta.csv", index=False)
        output_path = args.output_dir / "submission_exp024_cnn8_videomae_aux.csv"
        save_submission(args.root, output_path, test_ids, test_prob)
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
            "history": args.output_dir / "history.csv",
            "summary": args.output_dir / "summary.json",
            "checkpoint": args.output_dir / "exp024_cnn8_videomae_aux_best.pt",
            "feature_normalization": args.output_dir / "feature_normalization.npz",
            "test_probabilities": args.output_dir / "test_probabilities.npy" if args.predict_test else None,
            "test_meta": args.output_dir / "test_meta.csv" if args.predict_test else None,
            "submission": output_path,
        },
        metrics={"train": train_metrics, "val": val_metrics},
        extra={"best_epoch": best_epoch, "best_valid_loss": best_loss},
    )
    print(f"saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
