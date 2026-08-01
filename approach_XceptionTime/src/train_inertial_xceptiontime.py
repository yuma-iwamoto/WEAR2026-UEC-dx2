import argparse
import json
from pathlib import Path
from typing import Final

ARM_SENSORS: Final[set[str]] = {"ra", "la"}
LEG_SENSORS: Final[set[str]] = {"rl", "ll"}


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from approach_XceptionTime.src.models.inceptiontime import InceptionTime
from approach_XceptionTime.src.models import ResNet1DClassifier
from approach_XceptionTime.src.models.xceptiontime import XceptionTime
from approach_base.src.output_utils import make_experiment_dir
from approach_base.src.output_utils import save_run_metadata
from approach_base.src.train_inertial_gbdt import (
    ID_TO_LABEL,
    N_CLASSES,
    SENSOR_COLS,
    assign_window_label_from_array,
    compute_normalization_stats,
    has_only_finite_sensor_windows,
    load_inertial_records,
    normalize_inertial_window,
    normalize_sensor_location,
)
from approach_base.src.visualize_confusion_matrix import plot_confusion_matrix, reorder_confusion_matrix_for_blocks


BASE_SENSOR_CHANNELS: Final[int] = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an inertial-only XceptionTime/InceptionTime/ResNet1D baseline.")
    parser.add_argument("--root", type=Path, default=Path("/workspace/input/3rd-wear-dataset-challenge-hasca-2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/approach_XceptionTime/output/inertial_xceptiontime"))
    parser.add_argument("--model", choices=["xceptiontime", "inceptiontime", "resnet1d"], default="xceptiontime")
    parser.add_argument("--val-subjects", type=int, nargs="+", default=[18, 19, 20, 21])
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--window-label-mode", choices=["purity", "majority", "strict"], default="purity")
    parser.add_argument("--min-label-purity", type=float, default=0.8)
    parser.add_argument("--sensor-keys", type=str, nargs="+", choices=list(SENSOR_COLS), default=list(SENSOR_COLS))
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--normalization-mode", choices=["none", "global", "subject", "window"], default="global")
    parser.add_argument("--add-magnitude", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-diff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-magnitude", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-5", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-5-magnitude", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-10", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-10-magnitude", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-20", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-20-magnitude", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-30", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-30-magnitude", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-40", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--add-diff-40-magnitude", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--aug-noise-std", type=float, default=0.0)
    parser.add_argument("--aug-scale-std", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, nargs="+", default=None)
    parser.add_argument("--adaptive-size", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=40)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--input-proj-dim", type=int, default=None)
    parser.add_argument("--disable-residual", action="store_true")
    parser.add_argument("--disable-bottleneck", action="store_true")
    parser.add_argument("--early-stopping-rounds", type=int, default=8)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--lr-scheduler-patience", type=int, default=3)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--exclude-file-id-suffix-2", action="store_true")
    parser.add_argument("--specialized-head", choices=["none", "arm_leg"], default="none")
    parser.add_argument("--sensor-embedding-mode", choices=["sensor", "limb", "none"], default="sensor")
    parser.add_argument("--sensor-fusion-mode", choices=["single", "parallel4"], default="single")
    parser.add_argument("--add-bilateral-features", action=argparse.BooleanOptionalAction, default=False, help="Add left/right pair-difference channels for sensor_fusion_mode=parallel4.")
    parser.add_argument("--canonicalize-left-limb", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


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


def sensor_group_index(sensor_key: str) -> int:
    if sensor_key in ARM_SENSORS:
        return 0
    if sensor_key in LEG_SENSORS:
        return 1
    raise ValueError(f"Unsupported sensor key for specialization: {sensor_key}")


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray) -> dict[str, dict[str, float]]:
    metrics = {}
    for group_id, group_name in ((0, "arm"), (1, "leg")):
        mask = group_ids == group_id
        if mask.any():
            metrics[group_name] = compute_metrics(y_true[mask], y_pred[mask])
    return metrics


def balanced_class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
    nonzero = counts > 0
    weights = np.ones(N_CLASSES, dtype=np.float32)
    weights[nonzero] = len(y) / (float(nonzero.sum()) * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32)


def resolve_channels(model_name: str, channels: list[int] | None) -> list[int]:
    if channels is not None:
        return [int(channel) for channel in channels]
    if model_name == 'resnet1d':
        return [64, 128, 128]
    if model_name == 'inceptiontime':
        return [32]
    return [16]


def resolve_dropout(model_name: str, dropout: float | None) -> float:
    if dropout is not None:
        return float(dropout)
    if model_name == 'resnet1d':
        return 0.2
    return 0.0


def raw_sensor_channels_per_sensor(args: argparse.Namespace) -> int:
    channels = BASE_SENSOR_CHANNELS
    if args.add_magnitude:
        channels += 1
    if args.add_diff:
        channels += BASE_SENSOR_CHANNELS
    if args.add_diff_magnitude:
        channels += 1
    if args.add_diff_5:
        channels += BASE_SENSOR_CHANNELS
    if args.add_diff_5_magnitude:
        channels += 1
    if args.add_diff_10:
        channels += BASE_SENSOR_CHANNELS
    if args.add_diff_10_magnitude:
        channels += 1
    if getattr(args, "add_diff_20", False):
        channels += BASE_SENSOR_CHANNELS
    if getattr(args, "add_diff_20_magnitude", False):
        channels += 1
    if getattr(args, "add_diff_30", False):
        channels += BASE_SENSOR_CHANNELS
    if getattr(args, "add_diff_30_magnitude", False):
        channels += 1
    if getattr(args, "add_diff_40", False):
        channels += BASE_SENSOR_CHANNELS
    if getattr(args, "add_diff_40_magnitude", False):
        channels += 1
    return channels


def sensor_position_feature_dim(args: argparse.Namespace) -> int:
    mode = getattr(args, "sensor_embedding_mode", "sensor")
    if mode == "sensor":
        return len(args.sensor_keys)
    if mode == "limb":
        return 2
    if mode == "none":
        return 0
    raise ValueError(f"Unsupported sensor_embedding_mode: {mode}")


def sensor_channels_per_sensor(args: argparse.Namespace) -> int:
    return raw_sensor_channels_per_sensor(args) + sensor_position_feature_dim(args)


def bilateral_channels(args: argparse.Namespace) -> int:
    if not args.add_bilateral_features:
        return 0
    # Per pair: abs(R-L) xyz, abs(diff(R-L)) xyz, abs(diff2(R-L)) xyz, abs(|R|-|L|) scalar.
    return 10 * 2


def model_input_channels(args: argparse.Namespace) -> int:
    if args.sensor_fusion_mode == "parallel4":
        return raw_sensor_channels_per_sensor(args) * len(args.sensor_keys) + bilateral_channels(args)
    return sensor_channels_per_sensor(args)


def raw_input_channels(args: argparse.Namespace) -> int:
    if args.sensor_fusion_mode == "parallel4":
        return raw_sensor_channels_per_sensor(args) * len(args.sensor_keys)
    return raw_sensor_channels_per_sensor(args)


def resize_window_length(window: np.ndarray, target_length: int) -> np.ndarray:
    if window.ndim != 2 or window.shape[1] != BASE_SENSOR_CHANNELS:
        raise ValueError(f"Unexpected inertial shape: {window.shape}")
    if window.shape[0] == target_length:
        return window.astype(np.float32, copy=False)
    if window.shape[0] > target_length:
        return window[:target_length].astype(np.float32, copy=False)
    out = np.zeros((target_length, window.shape[1]), dtype=np.float32)
    out[:window.shape[0]] = window.astype(np.float32, copy=False)
    return out


def infer_test_window_size(root_dir: Path) -> int:
    test_dir = root_dir / "test"
    inertial = np.load(test_dir / "test_inertial_data.npy", mmap_mode="r")
    if inertial.ndim != 3:
        raise ValueError(f"Unexpected test inertial ndim: {inertial.shape}")
    sample_shape = inertial.shape[1:]
    if len(sample_shape) != 2:
        raise ValueError(f"Unexpected test inertial sample shape: {sample_shape}")
    if sample_shape[1] == BASE_SENSOR_CHANNELS:
        return int(sample_shape[0])
    if sample_shape[0] == BASE_SENSOR_CHANNELS:
        return int(sample_shape[1])
    raise ValueError(f"Unexpected test inertial sample shape: {sample_shape}")


def validate_test_configuration(root_dir: Path, args: argparse.Namespace) -> None:
    test_window_size = infer_test_window_size(root_dir)
    if int(args.window_size) != test_window_size:
        raise ValueError(
            f"predict-test requires window_size={test_window_size} to match test inertial windows, "
            f"but got window_size={args.window_size}"
        )


def compute_lagged_diff(window_t: np.ndarray, lag: int) -> np.ndarray:
    if lag <= 0:
        raise ValueError(f"lag must be positive, got {lag}")
    if window_t.shape[1] <= lag:
        return np.zeros_like(window_t, dtype=np.float32)
    diff = np.zeros_like(window_t, dtype=np.float32)
    diff[:, lag:] = window_t[:, lag:] - window_t[:, :-lag]
    return diff


def build_sensor_input(window_t: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    features = [window_t]
    if args.add_magnitude:
        magnitude = np.linalg.norm(window_t, axis=0, keepdims=True)
        features.append(magnitude.astype(np.float32, copy=False))

    diff_1 = None
    if args.add_diff or args.add_diff_magnitude:
        diff_1 = np.diff(window_t, axis=1, prepend=window_t[:, :1]).astype(np.float32, copy=False)
    if args.add_diff:
        features.append(diff_1)
    if args.add_diff_magnitude:
        diff_magnitude = np.linalg.norm(diff_1, axis=0, keepdims=True)
        features.append(diff_magnitude.astype(np.float32, copy=False))

    diff_5 = None
    if args.add_diff_5 or args.add_diff_5_magnitude:
        diff_5 = compute_lagged_diff(window_t, lag=5)
    if args.add_diff_5:
        features.append(diff_5)
    if args.add_diff_5_magnitude:
        diff_5_magnitude = np.linalg.norm(diff_5, axis=0, keepdims=True)
        features.append(diff_5_magnitude.astype(np.float32, copy=False))

    diff_10 = None
    if args.add_diff_10 or args.add_diff_10_magnitude:
        diff_10 = compute_lagged_diff(window_t, lag=10)
    if args.add_diff_10:
        features.append(diff_10)
    if args.add_diff_10_magnitude:
        diff_10_magnitude = np.linalg.norm(diff_10, axis=0, keepdims=True)
        features.append(diff_10_magnitude.astype(np.float32, copy=False))

    diff_20 = None
    if getattr(args, "add_diff_20", False) or getattr(args, "add_diff_20_magnitude", False):
        diff_20 = compute_lagged_diff(window_t, lag=20)
    if getattr(args, "add_diff_20", False):
        features.append(diff_20)
    if getattr(args, "add_diff_20_magnitude", False):
        diff_20_magnitude = np.linalg.norm(diff_20, axis=0, keepdims=True)
        features.append(diff_20_magnitude.astype(np.float32, copy=False))

    diff_30 = None
    if getattr(args, "add_diff_30", False) or getattr(args, "add_diff_30_magnitude", False):
        diff_30 = compute_lagged_diff(window_t, lag=30)
    if getattr(args, "add_diff_30", False):
        features.append(diff_30)
    if getattr(args, "add_diff_30_magnitude", False):
        diff_30_magnitude = np.linalg.norm(diff_30, axis=0, keepdims=True)
        features.append(diff_30_magnitude.astype(np.float32, copy=False))

    diff_40 = None
    if getattr(args, "add_diff_40", False) or getattr(args, "add_diff_40_magnitude", False):
        diff_40 = compute_lagged_diff(window_t, lag=40)
    if getattr(args, "add_diff_40", False):
        features.append(diff_40)
    if getattr(args, "add_diff_40_magnitude", False):
        diff_40_magnitude = np.linalg.norm(diff_40, axis=0, keepdims=True)
        features.append(diff_40_magnitude.astype(np.float32, copy=False))

    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def append_sensor_one_hot(sensor_features: np.ndarray, sensor_key: str, args: argparse.Namespace) -> np.ndarray:
    mode = getattr(args, "sensor_embedding_mode", "sensor")
    feature_dim = sensor_position_feature_dim(args)
    if feature_dim == 0:
        return sensor_features.astype(np.float32, copy=False)
    one_hot = np.zeros((feature_dim, sensor_features.shape[1]), dtype=np.float32)
    if mode == "sensor":
        one_hot[args.sensor_keys.index(sensor_key), :] = 1.0
    elif mode == "limb":
        one_hot[sensor_group_index(sensor_key), :] = 1.0
    else:
        raise ValueError(f"Unsupported sensor_embedding_mode: {mode}")
    return np.concatenate([sensor_features, one_hot], axis=0).astype(np.float32, copy=False)


def build_bilateral_sensor_inputs(sensor_windows_t: list[np.ndarray], args: argparse.Namespace) -> list[np.ndarray]:
    windows = dict(zip(args.sensor_keys, sensor_windows_t))
    derived_inputs = []
    for right_key, left_key in (("ra", "la"), ("rl", "ll")):
        right = windows[right_key]
        left = windows[left_key]
        right_minus_left = right - left
        abs_diff = np.zeros_like(right_minus_left, dtype=np.float32)
        abs_diff[:, 1:] = np.abs(right_minus_left[:, 1:] - right_minus_left[:, :-1])
        abs_second_diff = np.zeros_like(right_minus_left, dtype=np.float32)
        abs_second_diff[:, 2:] = np.abs(right_minus_left[:, 2:] - 2.0 * right_minus_left[:, 1:-1] + right_minus_left[:, :-2])
        right_mag = np.linalg.norm(right, axis=0, keepdims=True)
        left_mag = np.linalg.norm(left, axis=0, keepdims=True)
        magnitude_diff = np.abs(right_mag - left_mag)
        derived_inputs.extend([
            np.abs(right_minus_left).astype(np.float32, copy=False),
            abs_diff.astype(np.float32, copy=False),
            abs_second_diff.astype(np.float32, copy=False),
            magnitude_diff.astype(np.float32, copy=False),
        ])
    return derived_inputs


def build_parallel_sensor_input(sensor_windows_t: list[np.ndarray], args: argparse.Namespace) -> np.ndarray:
    features = [build_sensor_input(window_t, args) for window_t in sensor_windows_t]
    if args.add_bilateral_features:
        features.extend(build_bilateral_sensor_inputs(sensor_windows_t, args))
    return np.concatenate(features, axis=0).astype(np.float32, copy=False)


def fused_sensor_name(sensor_keys: list[str]) -> str:
    return "+".join(sensor_keys)


def validate_sensor_fusion_configuration(args: argparse.Namespace) -> None:
    if args.sensor_fusion_mode == "parallel4":
        if args.specialized_head != "none":
            raise ValueError("specialized_head is not supported with sensor_fusion_mode=parallel4")
        if list(args.sensor_keys) != list(SENSOR_COLS):
            raise ValueError(
                f"sensor_fusion_mode=parallel4 requires sensor_keys={list(SENSOR_COLS)}, got {list(args.sensor_keys)}"
            )




def augment_sensor_input(sensor_input: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    augmented = sensor_input.astype(np.float32, copy=True)
    raw_channels = raw_input_channels(args)

    if args.aug_scale_std > 0.0:
        scales = np.random.normal(loc=1.0, scale=args.aug_scale_std, size=(raw_channels, 1)).astype(np.float32)
        augmented[:raw_channels] *= scales

    if args.aug_noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=args.aug_noise_std, size=augmented[:raw_channels].shape).astype(np.float32)
        augmented[:raw_channels] += noise

    return augmented


class WindowedInertialDataset(Dataset):
    def __init__(self, records: list[dict], args: argparse.Namespace, split_name: str, normalization_stats: dict[str, object]):
        self.samples = []
        self.args = args
        self.apply_augmentation = split_name == "train"
        skipped_label = 0
        skipped_sensor = 0

        for rec in tqdm(records, desc=f"build_{split_name}", unit="record"):
            label_ids = rec["label_ids"]
            sensor_arrays = rec["sensor_arrays"]
            starts = list(range(0, len(rec["df"]) - args.window_size + 1, args.stride))
            if args.max_windows_per_record is not None and len(starts) > args.max_windows_per_record:
                starts = sorted(np.random.choice(starts, size=args.max_windows_per_record, replace=False).tolist())

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
                    sensor_windows_t = []
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
                        sensor_windows_t.append(np.nan_to_num(window.T, nan=0.0, posinf=0.0, neginf=0.0))
                    if invalid_sensor:
                        continue
                    self.samples.append({
                        "imu": build_parallel_sensor_input(sensor_windows_t, args),
                        "y": int(y),
                        "file_id": rec["file_id"],
                        "sbj_id": rec["sbj_id"],
                        "start": start,
                        "sensor": fused_sensor_name(args.sensor_keys),
                        "sensor_group": -1,
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

                    sensor_id = args.sensor_keys.index(sensor_key)
                    imu = append_sensor_one_hot(
                        build_sensor_input(
                            np.nan_to_num(window.T, nan=0.0, posinf=0.0, neginf=0.0),
                            args,
                        ),
                        sensor_key,
                        args,
                    )
                    self.samples.append({
                        "imu": imu,
                        "y": int(y),
                        "file_id": rec["file_id"],
                        "sbj_id": rec["sbj_id"],
                        "start": start,
                        "sensor": sensor_key,
                        "sensor_group": sensor_group_index(sensor_key),
                    })

        print(f"[{split_name}] samples={len(self.samples)} skipped_label={skipped_label} skipped_sensor={skipped_sensor}")
        if not self.samples:
            raise ValueError(f"No samples built for {split_name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        imu = sample["imu"]
        if self.apply_augmentation:
            imu = augment_sensor_input(imu, self.args)
        return {
            "id": torch.tensor(idx, dtype=torch.long),
            "imu": torch.tensor(imu.T, dtype=torch.float32),
            "y": torch.tensor(sample["y"], dtype=torch.long),
            "sensor_group": torch.tensor(sample["sensor_group"], dtype=torch.long),
        }

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "id": np.arange(len(self.samples), dtype=np.int64),
            "file_id": [sample["file_id"] for sample in self.samples],
            "sbj_id": [sample["sbj_id"] for sample in self.samples],
            "start": [sample["start"] for sample in self.samples],
            "sensor": [sample["sensor"] for sample in self.samples],
            "sensor_group": [sample["sensor_group"] for sample in self.samples],
            "y_true": [sample["y"] for sample in self.samples],
        })


class TestInertialDataset(Dataset):
    def __init__(self, root_dir: Path, args: argparse.Namespace, normalization_stats: dict[str, object]):
        test_dir = root_dir / "test"
        self.inertial = np.load(test_dir / "test_inertial_data.npy", mmap_mode="r")
        self.meta = pd.read_csv(test_dir / "test_meta_data.csv")
        self.args = args
        self.normalization_stats = normalization_stats
        if self.args.sensor_fusion_mode != "single":
            raise ValueError("Test dataset is only available for sensor_fusion_mode=single")
        if len(self.inertial) != len(self.meta):
            raise ValueError("test inertial and meta length mismatch")

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.meta.iloc[idx]
        sensor_col = "inertial_sensor_location" if "inertial_sensor_location" in self.meta.columns else "sensor_location"
        subject_col = "subject_id" if "subject_id" in self.meta.columns else None
        sensor_key = normalize_sensor_location(row[sensor_col])
        subject_id = int(row[subject_col]) if subject_col is not None and pd.notna(row[subject_col]) else None
        window = np.asarray(self.inertial[idx], dtype=np.float32)
        if window.ndim != 2:
            raise ValueError(f"Unexpected inertial ndim: {window.shape}")
        if window.shape[1] == BASE_SENSOR_CHANNELS:
            pass
        elif window.shape[0] == BASE_SENSOR_CHANNELS:
            window = window.T
        else:
            raise ValueError(f"Unexpected inertial shape: {window.shape}")
        window = resize_window_length(window, self.args.window_size)
        window = canonicalize_left_limb_window(window, sensor_key, self.args.canonicalize_left_limb)
        window = normalize_inertial_window(
            window,
            sensor_key,
            normalization_mode=self.args.normalization_mode,
            normalization_stats=self.normalization_stats,
            subject_id=subject_id,
        )
        sensor_id = self.args.sensor_keys.index(sensor_key)
        imu = append_sensor_one_hot(
            build_sensor_input(
                np.nan_to_num(window.T, nan=0.0, posinf=0.0, neginf=0.0),
                self.args,
            ),
            sensor_key,
            self.args,
        )
        if sensor_key not in self.args.sensor_keys:
            raise ValueError(f"Unexpected sensor_key for test sample: {sensor_key}")
        return {
            "id": torch.tensor(int(row["id"]), dtype=torch.long),
            "imu": torch.tensor(imu.T, dtype=torch.float32),
            "sensor_group": torch.tensor(sensor_group_index(sensor_key), dtype=torch.long),
        }


class InertialTimeSeriesClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        input_dim: int,
        n_classes: int,
        channels: list[int],
        adaptive_size: int,
        kernel_size: int,
        depth: int,
        residual: bool,
        bottleneck: bool,
        dropout: float,
        input_proj_dim: int | None,
        specialized_head: str = "none",
    ):
        super().__init__()
        self.model_name = model_name
        self.specialized_head = specialized_head
        self.n_classes = n_classes
        feature_dim = n_classes if specialized_head == "none" else n_classes

        if model_name == "xceptiontime":
            self.backbone = XceptionTime(
                c_in=input_dim,
                c_out=feature_dim,
                nf=channels[0],
                adaptive_size=adaptive_size,
                residual=residual,
                bottleneck=bottleneck,
                ks=kernel_size,
            )
        elif model_name == "inceptiontime":
            self.backbone = InceptionTime(
                c_in=input_dim,
                c_out=feature_dim,
                nf=channels[0],
                residual=residual,
                depth=depth,
                bottleneck=bottleneck,
                ks=kernel_size,
            )
        elif model_name == "resnet1d":
            self.backbone = ResNet1DClassifier(
                input_dim=input_dim,
                n_classes=feature_dim,
                channels=channels,
                dropout=dropout,
                input_proj_dim=input_proj_dim,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.dropout = nn.Identity() if model_name == "resnet1d" else (nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        if specialized_head == "arm_leg":
            self.arm_head = nn.Linear(feature_dim, n_classes)
            self.leg_head = nn.Linear(feature_dim, n_classes)
        elif specialized_head != "none":
            raise ValueError(f"Unsupported specialized_head: {specialized_head}")

    def encode(self, imu: torch.Tensor) -> torch.Tensor:
        imu = self.dropout(imu)
        if self.model_name == "resnet1d":
            return self.backbone(imu)
        return self.backbone(imu.transpose(1, 2).contiguous())

    def forward(self, imu: torch.Tensor, sensor_group: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encode(imu)
        if self.specialized_head == "none":
            return features
        if sensor_group is None:
            raise ValueError("sensor_group is required when specialized_head is enabled")

        logits = torch.empty(features.shape[0], self.n_classes, device=features.device, dtype=features.dtype)
        arm_mask = sensor_group == 0
        leg_mask = sensor_group == 1
        if arm_mask.any():
            logits[arm_mask] = self.arm_head(features[arm_mask])
        if leg_mask.any():
            logits[leg_mask] = self.leg_head(features[leg_mask])
        other_mask = ~(arm_mask | leg_mask)
        if other_mask.any():
            raise ValueError("Unexpected sensor_group values encountered")
        return logits




def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.inference_mode()
def predict_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    all_ids = []
    all_prob = []
    all_y = []
    has_target = False

    for batch in tqdm(loader, desc="predict", unit="batch"):
        ids = batch["id"].numpy()
        y = batch.get("y")
        batch = move_batch(batch, device)
        logits = model(batch["imu"], batch.get("sensor_group"))
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        all_ids.append(ids)
        all_prob.append(prob)
        if y is not None:
            has_target = True
            all_y.append(y.numpy())

    ids_np = np.concatenate(all_ids)
    prob_np = np.concatenate(all_prob)
    y_np = np.concatenate(all_y) if has_target else None
    return ids_np, prob_np, y_np


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device, log_every: int) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", unit="batch"), start=1):
        batch = move_batch(batch, device)
        logits = model(batch["imu"], batch.get("sensor_group"))
        loss = criterion(logits, batch["y"])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        if batch_idx % log_every == 0:
            print(f"train_batch={batch_idx} loss={loss.item():.6f}")

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate_probabilities_and_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    model.eval()
    all_ids = []
    all_prob = []
    all_y = []
    total_loss = 0.0
    n_batches = 0
    has_target = False
    for batch in tqdm(loader, desc="valid", unit="batch"):
        ids = batch["id"].numpy()
        y = batch.get("y")
        batch = move_batch(batch, device)
        logits = model(batch["imu"], batch.get("sensor_group"))
        if y is not None:
            has_target = True
            loss = criterion(logits, batch["y"])
            total_loss += float(loss.item())
            n_batches += 1
            all_y.append(y.numpy())
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        all_ids.append(ids)
        all_prob.append(prob)
    ids_np = np.concatenate(all_ids)
    prob_np = np.concatenate(all_prob)
    y_np = np.concatenate(all_y) if has_target else None
    loss_np = total_loss / max(n_batches, 1)
    return ids_np, prob_np, y_np, loss_np


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    return evaluate_probabilities_and_loss(model, loader, criterion, device)[3]


def save_eval_outputs(output_dir: Path, name: str, meta: pd.DataFrame, probabilities: np.ndarray) -> None:
    y_pred = probabilities.argmax(axis=1).astype(int)
    df = meta.copy()
    df["y_pred"] = y_pred
    df["y_true_label"] = [ID_TO_LABEL[int(v)] for v in df["y_true"]]
    df["y_pred_label"] = [ID_TO_LABEL[int(v)] for v in y_pred]
    df.to_csv(output_dir / f"{name}_predictions.csv", index=False)

    cm = confusion_matrix(df["y_true"].to_numpy(dtype=np.int64), y_pred, labels=list(range(N_CLASSES)))
    labels = [ID_TO_LABEL[i] for i in range(N_CLASSES)]
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / f"{name}_confusion_matrix.csv")
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
    y_true = df["y_true"].to_numpy(dtype=np.int64)
    report_text = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)), target_names=labels, digits=3, zero_division=0)
    (output_dir / f"{name}_classification_report.txt").write_text(report_text + "\n")
    report_dict = classification_report(y_true, y_pred, labels=list(range(N_CLASSES)), target_names=labels, digits=3, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(report_dict).T.round(3)
    if "support" in report_df.columns:
        report_df["support"] = report_df["support"].fillna(0).astype(int)
    report_df.to_csv(output_dir / f"{name}_classification_report.csv")
    np.save(output_dir / f"{name}_probabilities.npy", probabilities.astype(np.float32))
    from approach_base.src.export_cv_eval_from_oof import save_sensor_group_eval_bundles
    save_sensor_group_eval_bundles(
        output_dir,
        name,
        df,
        y_true,
        probabilities,
        normalize="true",
        annot="both",
        dpi=150,
    )


def save_submission(root_dir: Path, output_path: Path, ids: np.ndarray, probabilities: np.ndarray) -> None:
    preds = probabilities.argmax(axis=1).astype(int)
    pred_df = pd.DataFrame({"id": ids.astype(int), "target_value": preds})
    sub = pd.read_csv(root_dir / "sample_submission.csv")
    sub = sub[["id"]].merge(pred_df, on="id", how="left")
    if sub["target_value"].isna().any():
        missing = sub[sub["target_value"].isna()]["id"].tolist()[:10]
        raise ValueError(f"Missing predictions for ids: {missing}")
    sub["target_value"] = sub["target_value"].astype(int)
    sub.to_csv(output_path, index=False)


def checkpoint_name(model_name: str) -> str:
    return f"{model_name}_best.pt"


def submission_name(model_name: str) -> str:
    return f"submission_{model_name}_inertial.csv"


def main() -> None:
    args = parse_args()
    args.channels = resolve_channels(args.model, args.channels)
    args.dropout = resolve_dropout(args.model, args.dropout)
    validate_sensor_fusion_configuration(args)
    seed_everything(args.seed)
    args.output_dir = make_experiment_dir(args.output_dir, args.exp_name)

    device = torch.device("cuda" if args.device == "gpu" and torch.cuda.is_available() else "cpu")
    print(
        f"start model={args.model} device={device.type} window_label_mode={args.window_label_mode} "
        f"window_size={args.window_size} stride={args.stride} normalization_mode={args.normalization_mode} "
        f"add_magnitude={args.add_magnitude} add_diff={args.add_diff} add_diff_magnitude={args.add_diff_magnitude} "
        f"add_diff_5={args.add_diff_5} add_diff_5_magnitude={args.add_diff_5_magnitude} "
        f"add_diff_10={args.add_diff_10} add_diff_10_magnitude={args.add_diff_10_magnitude} "
        f"add_diff_20={getattr(args, 'add_diff_20', False)} add_diff_20_magnitude={getattr(args, 'add_diff_20_magnitude', False)} "
        f"add_diff_30={getattr(args, 'add_diff_30', False)} add_diff_30_magnitude={getattr(args, 'add_diff_30_magnitude', False)} "
        f"add_diff_40={getattr(args, 'add_diff_40', False)} add_diff_40_magnitude={getattr(args, 'add_diff_40_magnitude', False)} "
        f"channels={args.channels} input_proj_dim={args.input_proj_dim} specialized_head={args.specialized_head} "
        f"sensor_embedding_mode={args.sensor_embedding_mode} sensor_fusion_mode={args.sensor_fusion_mode}"
    )

    records = load_inertial_records(args.root, exclude_file_id_suffix_2=args.exclude_file_id_suffix_2)
    val_subjects = set(args.val_subjects)
    train_records = [record for record in records if record["sbj_id"] not in val_subjects]
    val_records = [record for record in records if record["sbj_id"] in val_subjects]
    normalization_stats = compute_normalization_stats(train_records)
    print(f"train_subjects={sorted({r['sbj_id'] for r in train_records})}")
    print(f"val_subjects={sorted({r['sbj_id'] for r in val_records})}")

    train_ds = WindowedInertialDataset(train_records, args, "train", normalization_stats)
    val_ds = WindowedInertialDataset(val_records, args, "val", normalization_stats)
    train_meta = train_ds.metadata_frame()
    val_meta = val_ds.metadata_frame()
    print(
        f"dataset train_samples={len(train_ds)} val_samples={len(val_ds)} "
        f"train_classes={np.bincount(train_meta['y_true'].to_numpy(dtype=np.int64), minlength=N_CLASSES).tolist()} "
        f"val_classes={np.bincount(val_meta['y_true'].to_numpy(dtype=np.int64), minlength=N_CLASSES).tolist()}"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = InertialTimeSeriesClassifier(
        model_name=args.model,
        input_dim=model_input_channels(args),
        n_classes=N_CLASSES,
        channels=args.channels,
        adaptive_size=args.adaptive_size,
        kernel_size=args.kernel_size,
        depth=args.depth,
        residual=not args.disable_residual,
        bottleneck=not args.disable_bottleneck,
        dropout=args.dropout,
        input_proj_dim=args.input_proj_dim,
        specialized_head=args.specialized_head,
    ).to(device)
    class_weights = balanced_class_weights(train_meta['y_true'].to_numpy(dtype=np.int64)).to(device) if args.class_weight == 'balanced' else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
    ) if args.lr_scheduler == "plateau" else None

    best_score = -1.0
    best_state = None
    best_epoch = None
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.log_every)
        _, val_prob, y_val, val_loss = evaluate_probabilities_and_loss(model, val_loader, criterion, device)
        val_pred = val_prob.argmax(axis=1)
        val_metrics = compute_metrics(y_val, val_pred)
        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, **val_metrics})
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"val_acc={val_metrics['acc']:.6f} val_macro_f1={val_metrics['macro_f1']:.6f} "
            f"val_weighted_f1={val_metrics['weighted_f1']:.6f}"
        )

        if val_metrics['macro_f1'] > best_score:
            best_score = val_metrics['macro_f1']
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
            torch.save({
                'model_state_dict': best_state,
                'args': vars(args),
                'best_epoch': best_epoch,
                'best_macro_f1': best_score,
            }, args.output_dir / checkpoint_name(args.model))
            print(f"saved best epoch={epoch} macro_f1={best_score:.6f}")
        else:
            bad_epochs += 1
            print(f"bad_epochs={bad_epochs}/{args.early_stopping_rounds}")

        if bad_epochs >= args.early_stopping_rounds:
            print('early stopping')
            break

    if best_state is None:
        raise RuntimeError('Training did not produce a best checkpoint')

    model.load_state_dict(best_state)
    _, train_prob, y_train = predict_probabilities(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda"), device)
    _, val_prob, y_val = predict_probabilities(model, val_loader, device)
    train_pred = train_prob.argmax(axis=1)
    val_pred = val_prob.argmax(axis=1)
    train_metrics = compute_metrics(y_train, train_pred)
    val_metrics = compute_metrics(y_val, val_pred)
    train_group_metrics = group_metrics(y_train, train_pred, train_meta['sensor_group'].to_numpy(dtype=np.int64)) if args.specialized_head == 'arm_leg' and args.sensor_fusion_mode == 'single' else None
    val_group_metrics = group_metrics(y_val, val_pred, val_meta['sensor_group'].to_numpy(dtype=np.int64)) if args.specialized_head == 'arm_leg' and args.sensor_fusion_mode == 'single' else None
    print(f"final_best epoch={best_epoch} train_metrics={train_metrics} val_metrics={val_metrics}")
    if train_group_metrics is not None:
        print(f"train_group_metrics={train_group_metrics}")
        print(f"val_group_metrics={val_group_metrics}")

    save_eval_outputs(args.output_dir, 'train', train_meta, train_prob)
    save_eval_outputs(args.output_dir, 'val', val_meta, val_prob)
    pd.DataFrame(history).to_csv(args.output_dir / 'history.csv', index=False)
    (args.output_dir / 'summary.json').write_text(json.dumps({
        'best_epoch': best_epoch,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'train_group_metrics': train_group_metrics,
        'val_group_metrics': val_group_metrics,
    }, indent=2))

    if args.predict_test:
        if args.sensor_fusion_mode != "single":
            raise ValueError("predict-test is not supported with sensor_fusion_mode=parallel4")
        validate_test_configuration(args.root, args)
        test_ds = TestInertialDataset(args.root, args, normalization_stats)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        test_ids, test_prob, _ = predict_probabilities(model, test_loader, device)
        np.save(args.output_dir / 'test_probabilities.npy', test_prob.astype(np.float32))
        test_ds.meta.to_csv(args.output_dir / 'test_meta.csv', index=False)
        output_path = args.output_dir / submission_name(args.model)
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
            "checkpoint": args.output_dir / checkpoint_name(args.model),
            "test_probabilities": args.output_dir / "test_probabilities.npy" if args.predict_test else None,
            "test_meta": args.output_dir / "test_meta.csv" if args.predict_test else None,
            "submission": output_path if args.predict_test else None,
        },
        metrics={"train": train_metrics, "val": val_metrics},
        extra={"best_epoch": best_epoch, "train_group_metrics": train_group_metrics, "val_group_metrics": val_group_metrics},
    )
    print(f"saved outputs: {args.output_dir}")


if __name__ == '__main__':
    main()
