import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from approach_base.src.collect_cv_oof import resolve_output_file


N_CLASSES = 19
KEY_COLUMNS = ['file_id', 'start', 'sensor']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Align OOF predictions/probabilities to a reference key set.')
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--pred-name', type=str, default='oof_predictions.csv')
    parser.add_argument('--prob-name', type=str, default='oof_probabilities.npy')
    parser.add_argument('--output-prefix', type=str, default='aligned')
    return parser.parse_args()


def _load_bundle(base_dir: Path, pred_name: str, prob_name: str) -> tuple[pd.DataFrame, np.ndarray]:
    pred_path = resolve_output_file(base_dir, pred_name)
    prob_path = resolve_output_file(base_dir, prob_name)
    df = pd.read_csv(pred_path)
    prob = np.load(prob_path).astype(np.float32)
    if prob.ndim != 2 or prob.shape[1] != N_CLASSES:
        raise ValueError(f'Expected probability shape (N, {N_CLASSES}) for {prob_path}, got {prob.shape}')
    if len(df) != prob.shape[0]:
        raise ValueError(f'Length mismatch for {base_dir}: len(df)={len(df)} prob_rows={prob.shape[0]}')
    missing_keys = [col for col in KEY_COLUMNS if col not in df.columns]
    if missing_keys:
        raise ValueError(f'Missing key columns in {pred_path}: {missing_keys}')
    return df, prob


def align_oof_to_reference(source_dir: Path, reference_dir: Path, output_dir: Path | None = None, pred_name: str = 'oof_predictions.csv', prob_name: str = 'oof_probabilities.npy', output_prefix: str = 'aligned') -> tuple[Path, Path]:
    output_dir = output_dir or source_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    source_df, source_prob = _load_bundle(source_dir, pred_name, prob_name)
    ref_df, _ = _load_bundle(reference_dir, pred_name, prob_name)

    source_keys = source_df[KEY_COLUMNS].copy()
    source_keys['source_row_idx'] = np.arange(len(source_df), dtype=np.int64)
    if 'y_true' in source_df.columns:
        source_keys['source_y_true'] = source_df['y_true'].to_numpy(dtype=np.int64)

    merged = ref_df.copy().merge(source_keys, on=KEY_COLUMNS, how='left', sort=False)
    if merged['source_row_idx'].isna().any():
        missing = merged.loc[merged['source_row_idx'].isna(), KEY_COLUMNS].head(10)
        raise ValueError('Missing source rows for reference keys, examples:\n' + missing.to_string(index=False))

    if 'y_true' in ref_df.columns and 'source_y_true' in merged.columns:
        ref_y = merged['y_true'].to_numpy(dtype=np.int64)
        src_y = merged['source_y_true'].to_numpy(dtype=np.int64)
        if not np.array_equal(ref_y, src_y):
            mismatch = merged.loc[ref_y != src_y, KEY_COLUMNS + ['y_true', 'source_y_true']].head(10)
            raise ValueError('y_true mismatch after alignment, examples:\n' + mismatch.to_string(index=False))

    row_idx = merged['source_row_idx'].to_numpy(dtype=np.int64)
    aligned_prob = source_prob[row_idx].astype(np.float32, copy=False)
    aligned_pred = aligned_prob.argmax(axis=1).astype(np.int64)

    out_df = ref_df.copy()
    out_df['source_row_idx'] = row_idx
    out_df['y_pred'] = aligned_pred
    if 'y_pred_label' in source_df.columns:
        lookup = dict(zip(np.arange(len(source_df), dtype=np.int64), source_df['y_pred_label'].astype(str)))
        out_df['y_pred_label'] = [lookup[idx] for idx in row_idx]
    if 'y_true_label' in source_df.columns:
        lookup = dict(zip(np.arange(len(source_df), dtype=np.int64), source_df['y_true_label'].astype(str)))
        out_df['y_true_label'] = [lookup[idx] for idx in row_idx]

    pred_out = output_dir / f'{output_prefix}_oof_predictions.csv'
    prob_out = output_dir / f'{output_prefix}_oof_probabilities.npy'
    out_df.to_csv(pred_out, index=False)
    np.save(prob_out, aligned_prob)
    return pred_out, prob_out


def main() -> None:
    args = parse_args()
    pred_out, prob_out = align_oof_to_reference(
        args.source_dir,
        args.reference_dir,
        args.output_dir,
        pred_name=args.pred_name,
        prob_name=args.prob_name,
        output_prefix=args.output_prefix,
    )
    print(f'saved: {pred_out}')
    print(f'saved: {prob_out}')


if __name__ == '__main__':
    main()
