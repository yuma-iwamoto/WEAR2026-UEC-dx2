import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from approach_base.src.collect_cv_oof import resolve_output_file


N_CLASSES = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Expand video test probabilities to sensor-level using inertial test metadata as reference.')
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--video-meta-name', type=str, default='cv_test_meta.csv')
    parser.add_argument('--video-prob-name', type=str, default='cv_test_probabilities.npy')
    parser.add_argument('--reference-meta-name', type=str, default='cv_test_meta.csv')
    return parser.parse_args()


def _load_test_bundle(base_dir: Path, meta_name: str, prob_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    meta_path = resolve_output_file(base_dir, meta_name)
    prob_path = resolve_output_file(base_dir, prob_name)
    meta = pd.read_csv(meta_path)
    prob = np.load(prob_path).astype(np.float32)
    if prob.ndim != 2 or prob.shape[1] != N_CLASSES:
        raise ValueError(f'Expected probability shape (N, {N_CLASSES}) for {prob_path}, got {prob.shape}')
    if len(meta) != prob.shape[0]:
        raise ValueError(f'Row/probability length mismatch for {base_dir}: len(meta)={len(meta)} prob_rows={prob.shape[0]}')
    return meta, prob


def _fallback_test_bundle(base_dir: Path, preferred_meta_name: str, preferred_prob_name: str) -> tuple[pd.DataFrame, np.ndarray, str, str]:
    candidates = [
        (preferred_meta_name, preferred_prob_name),
        ('test_meta.csv', 'test_probabilities.npy'),
    ]
    last_error = None
    for meta_name, prob_name in candidates:
        try:
            meta, prob = _load_test_bundle(base_dir, meta_name, prob_name)
            return meta, prob, meta_name, prob_name
        except Exception as exc:
            last_error = exc
    raise last_error  # type: ignore[misc]


def expand_video_test_to_sensor_level(video_dir: Path, reference_dir: Path, output_dir: Path | None = None, video_meta_name: str = 'cv_test_meta.csv', video_prob_name: str = 'cv_test_probabilities.npy', reference_meta_name: str = 'cv_test_meta.csv') -> tuple[Path, Path]:
    output_dir = output_dir or video_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_meta, video_prob, _, _ = _fallback_test_bundle(video_dir, video_meta_name, video_prob_name)
    ref_meta, _, ref_meta_name_used, _ = _fallback_test_bundle(reference_dir, reference_meta_name, 'cv_test_probabilities.npy')

    if 'id' not in video_meta.columns or 'id' not in ref_meta.columns:
        raise ValueError('Both video and reference test meta must contain an id column')

    video_keys = video_meta[['id']].copy()
    video_keys['video_row_idx'] = np.arange(len(video_meta), dtype=np.int64)
    merged = ref_meta.copy().merge(video_keys, on='id', how='left', sort=False)
    if merged['video_row_idx'].isna().any():
        missing = merged.loc[merged['video_row_idx'].isna(), ['id']].head(10)
        raise ValueError('Missing matching video test rows for reference rows, examples:\n' + missing.to_string(index=False))

    row_idx = merged['video_row_idx'].to_numpy(dtype=np.int64)
    expanded_prob = video_prob[row_idx].astype(np.float32, copy=False)

    out_meta = ref_meta.copy()
    out_meta['source'] = 'video'
    out_meta['source_row_idx'] = row_idx

    meta_path = output_dir / ('sensor_level_cv_test_meta.csv' if ref_meta_name_used == 'cv_test_meta.csv' else 'sensor_level_test_meta.csv')
    prob_path = output_dir / ('sensor_level_cv_test_probabilities.npy' if ref_meta_name_used == 'cv_test_meta.csv' else 'sensor_level_test_probabilities.npy')
    out_meta.to_csv(meta_path, index=False)
    np.save(prob_path, expanded_prob)
    return meta_path, prob_path


def main() -> None:
    args = parse_args()
    meta_path, prob_path = expand_video_test_to_sensor_level(
        args.video_dir,
        args.reference_dir,
        args.output_dir,
        video_meta_name=args.video_meta_name,
        video_prob_name=args.video_prob_name,
        reference_meta_name=args.reference_meta_name,
    )
    print(f'saved: {meta_path}')
    print(f'saved: {prob_path}')


if __name__ == '__main__':
    main()
