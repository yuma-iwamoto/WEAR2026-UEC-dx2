import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from approach_base.src.export_cv_eval_from_oof import save_eval_bundle, save_sensor_group_eval_bundles
from approach_base.src.output_utils import make_experiment_dir, save_run_metadata


N_CLASSES = 19


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble model probabilities from compatible validation/test outputs.")
    parser.add_argument("--root", type=Path, default=Path("/workspace/input/3rd-wear-dataset-challenge-hasca-2026"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/approach_base/output/ensemble"))
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--val-prob-files", type=Path, nargs="+", required=True)
    parser.add_argument("--test-prob-files", type=Path, nargs="+", default=None)
    parser.add_argument("--val-target-file", type=Path, required=True)
    parser.add_argument("--test-id-file", type=Path, default=None)
    parser.add_argument("--weight-mode", choices=["manual", "auto"], default="manual")
    parser.add_argument("--auto-search-mode", choices=["dirichlet", "greedy", "hybrid", "beam"], default="dirichlet")
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--n-trials", type=int, default=5000)
    parser.add_argument("--dirichlet-alpha", type=float, default=1.0)
    parser.add_argument("--hybrid-top-k", type=int, default=20)
    parser.add_argument("--hybrid-greedy-steps", type=int, default=500)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--beam-steps", type=int, default=500)
    parser.add_argument("--dirichlet-alpha-list", type=float, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("weights must be 1D")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value")
    return weights / total


def load_probability_stack(paths: list[Path]) -> list[np.ndarray]:
    probs = [np.load(path).astype(np.float32) for path in paths]
    first_shape = probs[0].shape
    for path, arr in zip(paths, probs):
        if arr.shape != first_shape:
            raise ValueError(f"Probability shape mismatch for {path}: expected {first_shape}, got {arr.shape}")
        if arr.ndim != 2 or arr.shape[1] != N_CLASSES:
            raise ValueError(f"Expected probability shape (N, {N_CLASSES}) for {path}, got {arr.shape}")
    return probs


def weighted_average(probabilities: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probabilities[0], dtype=np.float64)
    for weight, prob in zip(weights, probabilities):
        out += float(weight) * prob
    return out.astype(np.float32)


def macro_f1_from_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y_pred = probabilities.argmax(axis=1)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def resolve_dirichlet_alphas(dirichlet_alpha: float, dirichlet_alpha_list: list[float] | None) -> list[float]:
    alphas = dirichlet_alpha_list if dirichlet_alpha_list is not None else [dirichlet_alpha]
    if not alphas:
        raise ValueError("At least one Dirichlet alpha is required")
    resolved = []
    for alpha in alphas:
        alpha = float(alpha)
        if alpha <= 0.0:
            raise ValueError(f"Dirichlet alpha must be positive, got {alpha}")
        resolved.append(alpha)
    return resolved


def sample_dirichlet_candidates(rng: np.random.Generator, n_models: int, n_trials: int, alphas: list[float]) -> tuple[list[np.ndarray], list[str]]:
    candidates: list[np.ndarray] = []
    labels: list[str] = []
    if n_trials <= 0:
        return candidates, labels

    base, remainder = divmod(n_trials, len(alphas))
    for alpha_idx, alpha in enumerate(alphas):
        count = base + (1 if alpha_idx < remainder else 0)
        if count <= 0:
            continue
        sampled = rng.dirichlet(np.full(n_models, alpha, dtype=np.float64), size=count)
        candidates.extend(sampled)
        labels.extend([f"dirichlet(alpha={alpha})"] * count)
    return candidates, labels


def score_dirichlet_candidates(
    y_true: np.ndarray,
    val_probs: list[np.ndarray],
    n_trials: int,
    seed: int,
    dirichlet_alphas: list[float],
    *,
    log_prefix: str = "search_progress",
) -> list[tuple[float, np.ndarray, str]]:
    n_models = len(val_probs)
    rng = np.random.default_rng(seed)

    candidates = [normalize_weights(np.ones(n_models, dtype=np.float64))]
    candidate_sources = ["uniform"]
    for idx in range(n_models):
        one_hot = np.zeros(n_models, dtype=np.float64)
        one_hot[idx] = 1.0
        candidates.append(one_hot)
        candidate_sources.append(f"one_hot(model={idx})")

    random_candidates, random_sources = sample_dirichlet_candidates(rng, n_models, n_trials, dirichlet_alphas)
    candidates.extend(random_candidates)
    candidate_sources.extend(random_sources)

    best_score = -1.0
    total_candidates = len(candidates)
    progress_every = max(1, min(500, total_candidates // 20))
    scored: list[tuple[float, np.ndarray, str]] = []
    for idx, (weights, source) in enumerate(zip(candidates, candidate_sources), start=1):
        normalized_weights = normalize_weights(weights)
        ensemble_prob = weighted_average(val_probs, normalized_weights)
        score = macro_f1_from_probabilities(y_true, ensemble_prob)
        scored.append((score, normalized_weights, source))
        if score > best_score:
            best_score = score
            print(f"{log_prefix} candidate={idx}/{total_candidates} source={source} best_macro_f1={best_score:.6f}")
        elif idx == 1 or idx % progress_every == 0 or idx == total_candidates:
            print(f"{log_prefix} candidate={idx}/{total_candidates} source={source} current_best_macro_f1={best_score:.6f}")

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def auto_search_weights_dirichlet(
    y_true: np.ndarray,
    val_probs: list[np.ndarray],
    n_trials: int,
    seed: int,
    dirichlet_alphas: list[float],
) -> tuple[np.ndarray, float]:
    scored = score_dirichlet_candidates(y_true, val_probs, n_trials, seed, dirichlet_alphas)
    best_score, best_weights, _ = scored[0]
    return best_weights, best_score


def auto_search_weights_greedy(
    y_true: np.ndarray,
    val_probs: list[np.ndarray],
    n_trials: int,
    *,
    init_weights: np.ndarray | None = None,
    log_prefix: str = "greedy_progress",
) -> tuple[np.ndarray, float]:
    n_models = len(val_probs)
    if n_trials <= 0:
        raise ValueError("n_trials must be positive for greedy search")

    if init_weights is None:
        current_counts = np.zeros(n_models, dtype=np.int64)
        running_sum = np.zeros_like(val_probs[0], dtype=np.float64)
        best_weights = normalize_weights(np.ones(n_models, dtype=np.float64))
        best_score = macro_f1_from_probabilities(y_true, weighted_average(val_probs, best_weights))
        print(f"{log_prefix} step=0/{n_trials} init=uniform best_macro_f1={best_score:.6f}")
    else:
        best_weights = normalize_weights(init_weights)
        current_counts = np.maximum(1, np.rint(best_weights * n_trials).astype(np.int64))
        running_sum = np.zeros_like(val_probs[0], dtype=np.float64)
        for model_idx, count in enumerate(current_counts):
            if count > 0:
                running_sum += float(count) * val_probs[model_idx]
        current_weights = normalize_weights(current_counts.astype(np.float64))
        best_score = macro_f1_from_probabilities(y_true, weighted_average(val_probs, current_weights))
        best_weights = current_weights.copy()
        print(f"{log_prefix} step=0/{n_trials} init=seeded best_macro_f1={best_score:.6f} weights={best_weights.tolist()}")

    progress_every = max(1, min(100, n_trials // 20))
    total_count = int(current_counts.sum())
    for step in range(1, n_trials + 1):
        step_best_score = -1.0
        step_best_idx = 0
        for model_idx, prob in enumerate(val_probs):
            candidate_prob = (running_sum + prob) / float(total_count + 1)
            score = macro_f1_from_probabilities(y_true, candidate_prob.astype(np.float32))
            if score > step_best_score:
                step_best_score = score
                step_best_idx = model_idx

        current_counts[step_best_idx] += 1
        running_sum += val_probs[step_best_idx]
        total_count += 1
        current_weights = normalize_weights(current_counts.astype(np.float64))

        if step_best_score > best_score:
            best_score = step_best_score
            best_weights = current_weights.copy()
            print(
                f"{log_prefix} step={step}/{n_trials} added_model={step_best_idx} "
                f"best_macro_f1={best_score:.6f} weights={best_weights.tolist()}"
            )
        elif step == 1 or step == n_trials or step % progress_every == 0:
            print(
                f"{log_prefix} step={step}/{n_trials} added_model={step_best_idx} "
                f"current_best_macro_f1={best_score:.6f}"
            )

    return best_weights, best_score



def auto_search_weights_beam(
    y_true: np.ndarray,
    val_probs: list[np.ndarray],
    beam_width: int,
    beam_steps: int,
) -> tuple[np.ndarray, float]:
    n_models = len(val_probs)
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    if beam_steps <= 0:
        raise ValueError("beam_steps must be positive")

    uniform_weights = normalize_weights(np.ones(n_models, dtype=np.float64))
    uniform_score = macro_f1_from_probabilities(y_true, weighted_average(val_probs, uniform_weights))
    print(f"beam_progress step=0/{beam_steps} init=uniform best_macro_f1={uniform_score:.6f}")

    beam: list[tuple[float, np.ndarray]] = [(uniform_score, np.zeros(n_models, dtype=np.int64))]
    best_score = uniform_score
    best_weights = uniform_weights
    progress_every = max(1, min(100, beam_steps // 20))

    for step in range(1, beam_steps + 1):
        candidate_counts_list: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()
        for _, counts in beam:
            for model_idx in range(n_models):
                candidate_counts = counts.copy()
                candidate_counts[model_idx] += 1
                key = tuple(int(v) for v in candidate_counts.tolist())
                if key in seen:
                    continue
                seen.add(key)
                candidate_counts_list.append(candidate_counts)

        candidates = []
        for candidate_counts in candidate_counts_list:
            candidate_weights = normalize_weights(candidate_counts.astype(np.float64))
            score = macro_f1_from_probabilities(y_true, weighted_average(val_probs, candidate_weights))
            candidates.append((score, candidate_counts))

        candidates.sort(key=lambda item: item[0], reverse=True)
        beam = candidates[: min(beam_width, len(candidates))]
        step_best_score, step_best_counts = beam[0]
        step_best_weights = normalize_weights(step_best_counts.astype(np.float64))

        if step_best_score > best_score:
            best_score = step_best_score
            best_weights = step_best_weights.copy()
            print(
                f"beam_progress step={step}/{beam_steps} beam_width={beam_width} "
                f"best_macro_f1={best_score:.6f} weights={best_weights.tolist()}"
            )
        elif step == 1 or step == beam_steps or step % progress_every == 0:
            print(
                f"beam_progress step={step}/{beam_steps} beam_width={beam_width} "
                f"current_best_macro_f1={best_score:.6f}"
            )

    return best_weights, best_score


def auto_search_weights_hybrid(
    y_true: np.ndarray,
    val_probs: list[np.ndarray],
    n_trials: int,
    seed: int,
    dirichlet_alphas: list[float],
    hybrid_top_k: int,
    hybrid_greedy_steps: int,
) -> tuple[np.ndarray, float]:
    if hybrid_top_k <= 0:
        raise ValueError("hybrid_top_k must be positive")
    if hybrid_greedy_steps <= 0:
        raise ValueError("hybrid_greedy_steps must be positive")

    scored = score_dirichlet_candidates(
        y_true,
        val_probs,
        n_trials,
        seed,
        dirichlet_alphas,
        log_prefix="hybrid_dirichlet_progress",
    )
    top_candidates = scored[: min(hybrid_top_k, len(scored))]
    best_score, best_weights, best_source = top_candidates[0]
    print(
        f"hybrid_seed_selection top_k={len(top_candidates)} initial_best_source={best_source} "
        f"initial_best_macro_f1={best_score:.6f}"
    )

    for rank, (seed_score, seed_weights, seed_source) in enumerate(top_candidates, start=1):
        print(
            f"hybrid_refine_start seed_rank={rank}/{len(top_candidates)} source={seed_source} "
            f"seed_macro_f1={seed_score:.6f}"
        )
        refined_weights, refined_score = auto_search_weights_greedy(
            y_true,
            val_probs,
            hybrid_greedy_steps,
            init_weights=seed_weights,
            log_prefix=f"hybrid_greedy_progress(rank={rank})",
        )
        if refined_score > best_score:
            best_score = refined_score
            best_weights = refined_weights
            best_source = f"{seed_source}->greedy"
            print(
                f"hybrid_refine_best seed_rank={rank}/{len(top_candidates)} best_source={best_source} "
                f"best_macro_f1={best_score:.6f}"
            )

    return best_weights, best_score


def save_val_predictions(output_dir: Path, reference_df: pd.DataFrame, probabilities: np.ndarray) -> None:
    y_pred = probabilities.argmax(axis=1).astype(int)
    df = reference_df.copy()
    df["ensemble_pred"] = y_pred
    df.to_csv(output_dir / "val_ensemble_predictions.csv", index=False)
    np.save(output_dir / "val_ensemble_probabilities.npy", probabilities.astype(np.float32))


def save_submission(output_dir: Path, root: Path, test_id_file: Path | None, probabilities: np.ndarray) -> None:
    if test_id_file is None:
        ids_df = pd.read_csv(root / "sample_submission.csv")
    else:
        ids_df = pd.read_csv(test_id_file)
    if "id" not in ids_df.columns:
        raise ValueError(f"id column not found in {test_id_file or (root / 'sample_submission.csv')}")

    preds = probabilities.argmax(axis=1).astype(int)
    pred_df = pd.DataFrame({"id": ids_df["id"].astype(int), "target_value": preds})
    sub = pd.read_csv(root / "sample_submission.csv")
    sub = sub[["id"]].merge(pred_df, on="id", how="left")
    if sub["target_value"].isna().any():
        missing = sub[sub["target_value"].isna()]["id"].tolist()[:10]
        raise ValueError(f"Missing predictions for ids: {missing}")
    sub["target_value"] = sub["target_value"].astype(int)
    sub.to_csv(output_dir / "submission_ensemble.csv", index=False)
    np.save(output_dir / "test_ensemble_probabilities.npy", probabilities.astype(np.float32))


def main() -> None:
    args = parse_args()
    args.output_dir = make_experiment_dir(args.output_dir, args.exp_name)
    print(f"loading_validation_probabilities n_files={len(args.val_prob_files)}")

    val_probs = load_probability_stack(args.val_prob_files)
    reference_df = pd.read_csv(args.val_target_file)
    if "y_true" not in reference_df.columns:
        raise ValueError(f"y_true column not found in {args.val_target_file}")
    y_true = reference_df["y_true"].to_numpy(dtype=np.int64)
    if len(y_true) != val_probs[0].shape[0]:
        raise ValueError("Validation labels and validation probabilities have different lengths")

    dirichlet_alphas = resolve_dirichlet_alphas(args.dirichlet_alpha, args.dirichlet_alpha_list)

    if args.weight_mode == "manual":
        if args.weights is None:
            raise ValueError("--weights is required when --weight-mode manual")
        if len(args.weights) != len(val_probs):
            raise ValueError("Number of weights must match number of val probability files")
        weights = normalize_weights(np.asarray(args.weights, dtype=np.float64))
        best_score = macro_f1_from_probabilities(y_true, weighted_average(val_probs, weights))
        print("manual_weights_applied")
    else:
        print(
            f"starting_auto_weight_search mode={args.auto_search_mode} n_models={len(val_probs)} n_trials={args.n_trials} "
            f"dirichlet_alphas={dirichlet_alphas}"
        )
        if args.auto_search_mode == "dirichlet":
            weights, best_score = auto_search_weights_dirichlet(
                y_true,
                val_probs,
                args.n_trials,
                args.seed,
                dirichlet_alphas,
            )
        elif args.auto_search_mode == "greedy":
            weights, best_score = auto_search_weights_greedy(
                y_true,
                val_probs,
                args.n_trials,
            )
        elif args.auto_search_mode == "hybrid":
            weights, best_score = auto_search_weights_hybrid(
                y_true,
                val_probs,
                args.n_trials,
                args.seed,
                dirichlet_alphas,
                args.hybrid_top_k,
                args.hybrid_greedy_steps,
            )
        else:
            weights, best_score = auto_search_weights_beam(
                y_true,
                val_probs,
                args.beam_width,
                args.beam_steps,
            )

    val_ensemble_prob = weighted_average(val_probs, weights)
    save_val_predictions(args.output_dir, reference_df, val_ensemble_prob)
    save_eval_bundle(
        args.output_dir / "val_eval",
        "val_ensemble",
        reference_df,
        y_true,
        val_ensemble_prob,
        normalize="true",
        annot="both",
        dpi=150,
    )
    created_group_evals = save_sensor_group_eval_bundles(
        args.output_dir,
        "val_ensemble",
        reference_df,
        y_true,
        val_ensemble_prob,
        normalize="true",
        annot="both",
        dpi=150,
    )


    summary = {
        "weight_mode": args.weight_mode,
        "weights": weights.tolist(),
        "val_macro_f1": float(best_score),
        "val_prob_files": [str(path) for path in args.val_prob_files],
        "test_prob_files": [str(path) for path in args.test_prob_files] if args.test_prob_files is not None else None,
        "auto_search_mode": args.auto_search_mode,
        "dirichlet_alpha": float(args.dirichlet_alpha),
        "dirichlet_alpha_list": None if args.dirichlet_alpha_list is None else [float(v) for v in args.dirichlet_alpha_list],
        "hybrid_top_k": int(args.hybrid_top_k),
        "hybrid_greedy_steps": int(args.hybrid_greedy_steps),
        "beam_width": int(args.beam_width),
        "beam_steps": int(args.beam_steps),
    }
    (args.output_dir / "ensemble_summary.json").write_text(json.dumps(summary, indent=2))
    output_paths = {
        "val_predictions": args.output_dir / "val_ensemble_predictions.csv",
        "val_probabilities": args.output_dir / "val_ensemble_probabilities.npy",
        "val_eval_dir": args.output_dir / "val_eval",
        "summary": args.output_dir / "ensemble_summary.json",
    }
    for group_name in created_group_evals:
        output_paths[f"val_{group_name}_eval_dir"] = args.output_dir / f"val_ensemble_{group_name}_eval"
    print(f"val_macro_f1={best_score:.6f}")
    print(f"weights={weights.tolist()}")

    if args.test_prob_files is not None:
        if len(args.test_prob_files) != len(val_probs):
            raise ValueError("Number of test probability files must match number of val probability files")
        test_probs = load_probability_stack(args.test_prob_files)
        test_ensemble_prob = weighted_average(test_probs, weights)
        save_submission(args.output_dir, args.root, args.test_id_file, test_ensemble_prob)
        output_paths["submission"] = args.output_dir / "submission_ensemble.csv"
        output_paths["test_probabilities"] = args.output_dir / "test_ensemble_probabilities.npy"
        print(f"saved submission: {args.output_dir / 'submission_ensemble.csv'}")

    save_run_metadata(
        args.output_dir,
        args=args,
        inputs={
            "val_prob_files": args.val_prob_files,
            "test_prob_files": args.test_prob_files,
            "val_target_file": args.val_target_file,
            "test_id_file": args.test_id_file,
        },
        outputs=output_paths,
        metrics={"val_macro_f1": float(best_score)},
        extra={
            "weights": weights.tolist(),
            "n_models": len(val_probs),
            "auto_search_mode": args.auto_search_mode,
            "dirichlet_alphas_used": dirichlet_alphas,
            "hybrid_top_k": int(args.hybrid_top_k),
            "hybrid_greedy_steps": int(args.hybrid_greedy_steps),
            "beam_width": int(args.beam_width),
            "beam_steps": int(args.beam_steps),
            "created_group_evals": created_group_evals,
            },
    )


if __name__ == "__main__":
    main()
