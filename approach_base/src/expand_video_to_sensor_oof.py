import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from approach_base.src.collect_cv_oof import resolve_output_file


N_CLASSES = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Expand video OOF probabilities to sensor-level using inertial OOF metadata as reference.')
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    return parser.parse_args()


def _load_predictions(base_dir: Path, pred_name: str, prob_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    pred_path = resolve_output_file(base_dir, pred_name)
    prob_path = resolve_output_file(base_dir, prob_name)
    df = pd.read_csv(pred_path)
    prob = np.load(prob_path).astype(np.float32)
    if prob.ndim != 2 or prob.shape[1] != N_CLASSES:
        raise ValueError(f'Expected probability shape (N, {N_CLASSES}) for {prob_path}, got {prob.shape}')
    if len(df) != prob.shape[0]:
        raise ValueError(f'Row/probability length mismatch for {base_dir}: len(df)={len(df)} prob_rows={prob.shape[0]}')
    return df, prob


def expand_video_oof_to_sensor_level(video_dir: Path, reference_dir: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = output_dir or video_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    video_df, video_prob = _load_predictions(video_dir, 'oof_predictions.csv', 'oof_probabilities.npy')
    ref_df, _ = _load_predictions(reference_dir, 'oof_predictions.csv', 'oof_probabilities.npy')

    required_video_cols = {'file_id', 'start', 'y_true'}
    required_ref_cols = {'file_id', 'start', 'sensor', 'y_true'}
    if not required_video_cols.issubset(video_df.columns):
        raise ValueError(f'Video OOF is missing columns: {sorted(required_video_cols - set(video_df.columns))}')
    if not required_ref_cols.issubset(ref_df.columns):
        raise ValueError(f'Reference OOF is missing columns: {sorted(required_ref_cols - set(ref_df.columns))}')

    video_keys = video_df[['file_id', 'start']].copy()
    video_keys['video_row_idx'] = np.arange(len(video_df), dtype=np.int64)
    video_keys['video_y_true'] = video_df['y_true'].to_numpy(dtype=np.int64)

    merged = ref_df.copy().merge(video_keys, on=['file_id', 'start'], how='left', sort=False)
    if merged['video_row_idx'].isna().any():
        missing = merged.loc[merged['video_row_idx'].isna(), ['file_id', 'start', 'sensor']].head(10)
        raise ValueError('Missing matching video OOF rows for reference rows, examples:\n' + missing.to_string(index=False))

    video_y_true = merged['video_y_true'].to_numpy(dtype=np.int64)
    ref_y_true = merged['y_true'].to_numpy(dtype=np.int64)
    if not np.array_equal(video_y_true, ref_y_true):
        mismatch = merged.loc[video_y_true != ref_y_true, ['file_id', 'start', 'sensor', 'y_true', 'video_y_true']].head(10)
        raise ValueError('y_true mismatch between video and reference OOF, examples:\n' + mismatch.to_string(index=False))

    row_idx = merged['video_row_idx'].to_numpy(dtype=np.int64)
    expanded_prob = video_prob[row_idx].astype(np.float32, copy=False)
    expanded_pred = expanded_prob.argmax(axis=1).astype(np.int64)

    out_df = ref_df.copy()
    out_df['source'] = 'video'
    out_df['source_row_idx'] = row_idx
    out_df['y_pred'] = expanded_pred
    if 'y_pred_label' in video_df.columns:
        label_lookup = dict(zip(np.arange(len(video_df), dtype=np.int64), video_df['y_pred_label'].astype(str)))
        out_df['y_pred_label'] = [label_lookup[idx] for idx in row_idx]
    if 'y_true_label' in video_df.columns:
        true_label_lookup = dict(zip(np.arange(len(video_df), dtype=np.int64), video_df['y_true_label'].astype(str)))
        out_df['y_true_label'] = [true_label_lookup[idx] for idx in row_idx]

    pred_path = output_dir / 'sensor_level_oof_predictions.csv'
    prob_path = output_dir / 'sensor_level_oof_probabilities.npy'
    out_df.to_csv(pred_path, index=False)
    np.save(prob_path, expanded_prob)
    return pred_path, prob_path


def main() -> None:
    args = parse_args()
    pred_path, prob_path = expand_video_oof_to_sensor_level(args.video_dir, args.reference_dir, args.output_dir)
    print(f'saved: {pred_path}')
    print(f'saved: {prob_path}')


if __name__ == '__main__':
    main()
