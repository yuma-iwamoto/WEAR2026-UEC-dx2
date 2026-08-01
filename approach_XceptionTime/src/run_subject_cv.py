import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from approach_base.src.export_cv_eval_from_oof import save_eval_bundle
from approach_base.src.output_utils import make_experiment_dir
from approach_base.src.output_utils import save_run_metadata
from approach_base.src.train_inertial_gbdt import load_inertial_records
from approach_XceptionTime.src.predict_cv_test import main as predict_cv_test_main


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run subject-based cross validation for approach_XceptionTime training scripts.")
    parser.add_argument("--root", type=Path, default=Path("/workspace/input/3rd-wear-dataset-challenge-hasca-2026"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="*", default=None)
    parser.add_argument("--fold-size", type=int, default=1, help="Number of subjects per validation fold. 1 means LOSO.")
    parser.add_argument("--num-folds", type=int, default=None, help="Number of validation folds. Subjects are split into roughly equal contiguous folds.")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--exclude-file-id-suffix-2", action="store_true")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()
    if args.num_folds is not None and args.num_folds <= 1:
        parser.error("--num-folds must be greater than 1")
    return args, extra_args


def chunk_subjects(subjects: list[int], fold_size: int) -> list[list[int]]:
    if fold_size <= 0:
        raise ValueError("fold_size must be positive")
    return [subjects[idx:idx + fold_size] for idx in range(0, len(subjects), fold_size)]


def split_subjects_into_folds(subjects: list[int], num_folds: int) -> list[list[int]]:
    if num_folds <= 1:
        raise ValueError("num_folds must be greater than 1")
    if num_folds > len(subjects):
        raise ValueError(f"num_folds={num_folds} is greater than number of subjects={len(subjects)}")

    base_size, remainder = divmod(len(subjects), num_folds)
    folds = []
    start = 0
    for fold_idx in range(num_folds):
        fold_size = base_size + (1 if fold_idx < remainder else 0)
        end = start + fold_size
        folds.append(subjects[start:end])
        start = end
    return folds


def discover_subjects(root: Path, exclude_file_id_suffix_2: bool) -> list[int]:
    records = load_inertial_records(root, exclude_file_id_suffix_2=exclude_file_id_suffix_2)
    return sorted({int(record["sbj_id"]) for record in records})


def resolve_output_file(base_dir: Path, filename: str) -> Path:
    direct_path = base_dir / filename
    if direct_path.exists():
        return direct_path

    candidates = sorted(
        [path for path in base_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for child_dir in candidates:
        candidate = child_dir / filename
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find {filename} under {base_dir}")


def save_oof_outputs(cv_output_dir: Path, fold_names: list[str]) -> None:
    oof_frames = []
    oof_probabilities = []

    for fold_name in fold_names:
        fold_dir = cv_output_dir / fold_name
        val_pred_path = resolve_output_file(fold_dir, 'val_predictions.csv')
        val_prob_path = resolve_output_file(fold_dir, 'val_probabilities.npy')
        fold_df = pd.read_csv(val_pred_path)
        fold_df['fold'] = fold_name
        oof_frames.append(fold_df)
        oof_probabilities.append(np.load(val_prob_path))

    if not oof_frames:
        return

    oof_df = pd.concat(oof_frames, axis=0, ignore_index=True)
    order = np.lexsort((
        oof_df['sensor'].to_numpy(),
        oof_df['start'].to_numpy(),
        oof_df['file_id'].to_numpy(),
    ))
    oof_df = oof_df.iloc[order].reset_index(drop=True)
    oof_prob = np.concatenate(oof_probabilities, axis=0)[order].astype(np.float32)
    oof_df.to_csv(cv_output_dir / 'oof_predictions.csv', index=False)
    np.save(cv_output_dir / 'oof_probabilities.npy', oof_prob)


def main() -> None:
    args, extra_args = parse_args()
    args.output_dir = make_experiment_dir(args.output_dir, args.exp_name)

    subjects = args.subjects if args.subjects else discover_subjects(args.root, args.exclude_file_id_suffix_2)
    if not subjects:
        raise ValueError("No subjects found for cross validation")
    if args.num_folds is not None:
        folds = split_subjects_into_folds(subjects, args.num_folds)
    else:
        folds = chunk_subjects(subjects, args.fold_size)
    if args.max_folds is not None:
        folds = folds[:args.max_folds]

    print(f"subjects={subjects}")
    print(f"fold_size={args.fold_size} num_folds={args.num_folds} n_folds={len(folds)}")
    print(f"exclude_file_id_suffix_2={args.exclude_file_id_suffix_2}")
    if extra_args:
        print(f"forwarded_args={shlex.join(extra_args)}")

    rows = []
    completed_folds = []
    for fold_idx, val_subjects in enumerate(folds):
        fold_name = f"fold_{fold_idx:02d}"
        fold_output_root = args.output_dir / fold_name
        cmd = [
            args.python_bin,
            "-m",
            "approach_XceptionTime.src.train_inertial_xceptiontime",
            "--root",
            str(args.root),
            "--output-dir",
            str(fold_output_root),
            "--val-subjects",
            *[str(subject) for subject in val_subjects],
            *extra_args,
        ]
        if args.exclude_file_id_suffix_2:
            cmd.append("--exclude-file-id-suffix-2")

        print(f"[{fold_name}] val_subjects={val_subjects}")
        print(f"[{fold_name}] cmd={shlex.join(cmd)}")

        if args.dry_run:
            rows.append({"fold": fold_name, "val_subjects": " ".join(map(str, val_subjects)), "status": "dry_run"})
            continue

        subprocess.run(cmd, check=True)
        summary = json.loads(resolve_output_file(fold_output_root, "summary.json").read_text())
        val_metrics = summary["val_metrics"]
        completed_folds.append(fold_name)
        rows.append({
            "fold": fold_name,
            "val_subjects": " ".join(map(str, val_subjects)),
            "val_acc": float(val_metrics["acc"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_weighted_f1": float(val_metrics["weighted_f1"]),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.output_dir / "cv_summary.csv", index=False)
    if not args.dry_run and {"val_acc", "val_macro_f1", "val_weighted_f1"}.issubset(summary_df.columns):
        save_oof_outputs(args.output_dir, completed_folds)
        oof_pred_path = args.output_dir / 'oof_predictions.csv'
        oof_prob_path = args.output_dir / 'oof_probabilities.npy'
        oof_reference_df = pd.read_csv(oof_pred_path)
        oof_probabilities = np.load(oof_prob_path).astype(np.float32)
        save_eval_bundle(
            args.output_dir / 'oof_eval',
            'oof',
            oof_reference_df,
            oof_reference_df['y_true'].to_numpy(dtype=np.int64),
            oof_probabilities,
            normalize='true',
            annot='both',
            dpi=150,
        )
        print(f"saved oof eval bundle: {args.output_dir / 'oof_eval'}")
        original_argv = sys.argv[:]
        try:
            sys.argv = [
                'predict_cv_test.py',
                '--cv-output-dir',
                str(args.output_dir),
                '--root',
                str(args.root),
            ]
            predict_cv_test_main()
        except Exception as exc:
            print(f"cv test aggregation skipped: {exc}")
        finally:
            sys.argv = original_argv
        mean_row = {
            "fold": "mean",
            "val_subjects": "-",
            "val_acc": float(summary_df["val_acc"].mean()),
            "val_macro_f1": float(summary_df["val_macro_f1"].mean()),
            "val_weighted_f1": float(summary_df["val_weighted_f1"].mean()),
        }
        pd.DataFrame([mean_row]).to_csv(args.output_dir / "cv_mean.csv", index=False)
        print(f"cv_mean={mean_row}")
    metadata_outputs = {
        'cv_summary': args.output_dir / 'cv_summary.csv',
        'cv_mean': args.output_dir / 'cv_mean.csv' if (args.output_dir / 'cv_mean.csv').exists() else None,
        'oof_predictions': args.output_dir / 'oof_predictions.csv' if (args.output_dir / 'oof_predictions.csv').exists() else None,
        'oof_probabilities': args.output_dir / 'oof_probabilities.npy' if (args.output_dir / 'oof_probabilities.npy').exists() else None,
        'oof_eval_dir': args.output_dir / 'oof_eval' if (args.output_dir / 'oof_eval').exists() else None,
        'cv_test_probabilities': args.output_dir / 'cv_test_probabilities.npy' if (args.output_dir / 'cv_test_probabilities.npy').exists() else None,
        'cv_test_meta': args.output_dir / 'cv_test_meta.csv' if (args.output_dir / 'cv_test_meta.csv').exists() else None,
    }
    save_run_metadata(
        args.output_dir,
        args=args,
        inputs={'root': args.root, 'subjects': subjects, 'extra_args': extra_args},
        outputs=metadata_outputs,
    )
    print(f"saved outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
