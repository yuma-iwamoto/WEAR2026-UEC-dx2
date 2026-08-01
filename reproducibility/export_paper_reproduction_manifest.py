import argparse
import json
import shlex
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def command_from_argv(argv: list[str]) -> str:
    if not argv:
        return ""
    cmd = ["python"]
    first = Path(argv[0])
    if first.name.endswith(".py") and "src" in first.parts:
        src_idx = first.parts.index("src")
        package_parts = first.parts[src_idx - 1 : -1] + (first.stem,)
        cmd.extend(["-m", ".".join(package_parts)])
        cmd.extend(argv[1:])
    else:
        cmd.extend(argv)
    return " ".join(shlex.quote(str(x)) for x in cmd)


def base_run_dir_from_prob_file(path: str) -> Path:
    p = Path(path)
    return p.parent


def infer_model_name(run_dir: Path, prob_file: str) -> str:
    text = f"{run_dir}/{prob_file}"
    if "cv_inertial_gbdt_lgbm" in text:
        return "Inertial-LightGBM"
    if "cv_xceptiontime" in text:
        return "Inertial-XceptionTime"
    if "cv_video_cnn" in text:
        return "Video-CNN"
    if "cv_video_mlp" in text:
        return "Video-MLP"
    if "exp024_xceptiontime" in text:
        return "Inertial-XceptionTime+Video-MLP"
    if "exp024_reg_mid" in text:
        return "Inertial-CNN+Video-MLP"
    return run_dir.name


def build_align_oof_command(source_dir: Path, reference_dir: Path, output_prefix: str) -> str:
    cmd = [
        "python",
        "-m",
        "approach_base.src.align_oof_to_reference",
        "--source-dir",
        str(source_dir),
        "--reference-dir",
        str(reference_dir),
        "--output-dir",
        str(source_dir),
        "--output-prefix",
        output_prefix,
    ]
    return " ".join(shlex.quote(x) for x in cmd)


def build_expand_video_oof_command(video_dir: Path, reference_dir: Path) -> str:
    cmd = [
        "python",
        "-m",
        "approach_base.src.expand_video_to_sensor_oof",
        "--video-dir",
        str(video_dir),
        "--reference-dir",
        str(reference_dir),
        "--output-dir",
        str(video_dir),
    ]
    return " ".join(shlex.quote(x) for x in cmd)


def build_expand_video_test_command(video_dir: Path, reference_dir: Path) -> str:
    cmd = [
        "python",
        "-m",
        "approach_base.src.expand_video_to_sensor_test",
        "--video-dir",
        str(video_dir),
        "--reference-dir",
        str(reference_dir),
        "--output-dir",
        str(video_dir),
    ]
    return " ".join(shlex.quote(x) for x in cmd)


def build_postprocess_steps(ensemble_args: dict) -> list[dict]:
    val_target_dir = Path(ensemble_args["val_target_file"]).parent
    test_id_dir = Path(ensemble_args["test_id_file"]).parent
    steps = []
    seen = set()

    for prob_file in ensemble_args["val_prob_files"]:
        prob_path = Path(prob_file)
        run_dir = prob_path.parent
        if prob_path.name == "exp024ref_oof_probabilities.npy":
            key = ("align", str(run_dir), "exp024ref")
            if key not in seen:
                steps.append(
                    {
                        "name": f"align_oof_to_exp024_reference:{run_dir.name}",
                        "purpose": "Align OOF rows to the multimodal reference OOF row order used by the ensemble.",
                        "command": build_align_oof_command(run_dir, val_target_dir, "exp024ref"),
                        "outputs": [
                            str(run_dir / "exp024ref_oof_predictions.csv"),
                            str(run_dir / "exp024ref_oof_probabilities.npy"),
                        ],
                    }
                )
                seen.add(key)
        if prob_path.name == "sensor_level_oof_probabilities.npy":
            key = ("expand_video_oof", str(run_dir))
            if key not in seen:
                steps.append(
                    {
                        "name": f"expand_video_oof_to_sensor_level:{run_dir.name}",
                        "purpose": "Assign video-only OOF probabilities to the corresponding sensor-level rows.",
                        "command": build_expand_video_oof_command(run_dir, val_target_dir),
                        "outputs": [
                            str(run_dir / "sensor_level_oof_predictions.csv"),
                            str(run_dir / "sensor_level_oof_probabilities.npy"),
                        ],
                    }
                )
                seen.add(key)

    for prob_file in ensemble_args["test_prob_files"]:
        prob_path = Path(prob_file)
        run_dir = prob_path.parent
        if prob_path.name == "sensor_level_cv_test_probabilities.npy":
            key = ("expand_video_test", str(run_dir))
            if key not in seen:
                steps.append(
                    {
                        "name": f"expand_video_test_to_sensor_level:{run_dir.name}",
                        "purpose": "Assign video-only test probabilities to the corresponding sensor-level rows.",
                        "command": build_expand_video_test_command(run_dir, test_id_dir),
                        "outputs": [
                            str(run_dir / "sensor_level_cv_test_meta.csv"),
                            str(run_dir / "sensor_level_cv_test_probabilities.npy"),
                        ],
                    }
                )
                seen.add(key)
    return steps


def build_manifest(ensemble_dir: Path) -> dict:
    run_metadata = read_json(ensemble_dir / "run_metadata.json")
    ensemble_args = run_metadata["args"]
    val_prob_files = ensemble_args["val_prob_files"]
    test_prob_files = ensemble_args["test_prob_files"]

    base_steps = []
    seen_dirs = set()
    for idx, prob_file in enumerate(val_prob_files):
        run_dir = base_run_dir_from_prob_file(prob_file)
        if run_dir in seen_dirs:
            continue
        meta_path = run_dir / "run_metadata.json"
        meta = read_json(meta_path)
        base_steps.append(
            {
                "index": idx + 1,
                "model_name": infer_model_name(run_dir, prob_file),
                "run_dir": str(run_dir),
                "command": command_from_argv(meta.get("argv", [])),
                "run_metadata": str(meta_path),
                "outputs": meta.get("outputs", {}),
            }
        )
        seen_dirs.add(run_dir)

    ensemble_command = command_from_argv(run_metadata.get("argv", []))
    return {
        "ensemble_dir": str(ensemble_dir),
        "objective": "Reproduce the base-model artifacts and ensemble output used for the paper.",
        "notes": [
            "Base training commands create timestamped output directories; downstream paths must be updated to the newly created directories when retraining from scratch.",
            "The ensemble command below exactly reproduces the reported ensemble from the saved base-model artifacts.",
            "OOF evaluation is computed over concatenated subject-wise CV predictions, not as a mean of fold scores.",
        ],
        "base_model_training_steps": base_steps,
        "postprocess_steps": build_postprocess_steps(ensemble_args),
        "ensemble_step": {
            "command": ensemble_command,
            "inputs": run_metadata.get("inputs", {}),
            "outputs": run_metadata.get("outputs", {}),
            "metrics": run_metadata.get("metrics", {}),
            "weights": run_metadata.get("extra", {}).get("weights"),
        },
        "source_run_metadata": str(ensemble_dir / "run_metadata.json"),
    }


def write_markdown(manifest: dict, path: Path) -> None:
    lines = [
        "# Paper Reproduction Manifest",
        "",
        "This file records the commands and artifacts used to reproduce the paper ensemble output.",
        "",
        "## Notes",
        "",
    ]
    for note in manifest["notes"]:
        lines.append(f"- {note}")
    lines.extend(["", "## Base Model Training", ""])
    for step in manifest["base_model_training_steps"]:
        lines.extend(
            [
                f"### Ensemble input {step['index']}: {step['model_name']}",
                "",
                f"- Run directory: `{step['run_dir']}`",
                f"- Metadata: `{step['run_metadata']}`",
                "",
                "```bash",
                step["command"],
                "```",
                "",
            ]
        )
    lines.extend(["## Post-processing", ""])
    for step in manifest["postprocess_steps"]:
        lines.extend(
            [
                f"### {step['name']}",
                "",
                step["purpose"],
                "",
                "```bash",
                step["command"],
                "```",
                "",
            ]
        )
    lines.extend(["## Ensemble", "", "```bash", manifest["ensemble_step"]["command"], "```", ""])
    path.write_text("\n".join(lines))


def write_shell(manifest: dict, path: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /workspace",
        "",
        "# This script reproduces the reported ensemble from saved base-model artifacts.",
        "# To retrain from scratch, run the base model commands in paper_reproduction_manifest.md first,",
        "# then update the downstream artifact paths to the newly timestamped output directories.",
        "",
        manifest["ensemble_step"]["command"],
        "",
    ]
    path.write_text("\n".join(lines))
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    ensemble_dir = args.ensemble_dir
    output_dir = args.output_dir or ensemble_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(ensemble_dir)
    json_path = output_dir / "paper_reproduction_manifest.json"
    md_path = output_dir / "paper_reproduction_manifest.md"
    sh_path = output_dir / "reproduce_paper_ensemble_from_artifacts.sh"

    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_markdown(manifest, md_path)
    write_shell(manifest, sh_path)

    print(f"saved {json_path}")
    print(f"saved {md_path}")
    print(f"saved {sh_path}")


if __name__ == "__main__":
    main()
