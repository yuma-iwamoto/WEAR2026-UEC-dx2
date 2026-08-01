import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from approach_base.src.train_inertial_gbdt import N_CLASSES, SENSOR_COLS
from approach_base.src.video_only_common import WindowedVideoDataset, TestVideoDataset, balanced_class_weights, build_video_temporal_features, compute_metrics, load_video_records, predict_probabilities, save_eval_outputs, save_submission, seed_everything, train_one_epoch, evaluate_probabilities_and_loss
from approach_base.src.output_utils import make_experiment_dir
from approach_base.src.output_utils import save_run_metadata


class VideoCNN(nn.Module):
    def __init__(self, input_feature_dim: int, proj_dim: int, channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Conv1d(input_feature_dim, proj_dim, kernel_size=1)
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(proj_dim, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, N_CLASSES),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.dim() != 3:
            raise ValueError(f'Expected 3D video tensor, got shape={tuple(video.shape)}')
        x = video.transpose(1, 2)
        x = self.input_proj(x)
        x = self.encoder(x).squeeze(-1)
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a simple video-only CNN baseline.')
    parser.add_argument('--root', type=Path, default=Path('/workspace/input/3rd-wear-dataset-challenge-hasca-2026'))
    parser.add_argument('--output-dir', type=Path, default=Path('/workspace/approach_base/output/video_cnn'))
    parser.add_argument('--val-subjects', type=int, nargs='+', default=[18, 19, 20, 21])
    parser.add_argument('--window-size', type=int, default=50)
    parser.add_argument('--stride', type=int, default=25)
    parser.add_argument('--window-label-mode', choices=['purity', 'majority', 'strict'], default='purity')
    parser.add_argument('--min-label-purity', type=float, default=0.8)
    parser.add_argument('--sensor-keys', type=str, nargs='+', choices=list(SENSOR_COLS), default=['ra'])
    parser.add_argument('--max-windows-per-record', type=int, default=None)
    parser.add_argument('--video-window-size', type=int, default=15)
    parser.add_argument('--video-hz', type=int, default=30)
    parser.add_argument('--video-frame-selection', choices=['all', 'first_mid_last'], default='all')
    parser.add_argument('--video-feature-mode', choices=['raw', 'delta_concat'], default='raw')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--proj-dim', type=int, default=256)
    parser.add_argument('--channels', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--early-stopping-rounds', type=int, default=5)
    parser.add_argument("--lr-scheduler", choices=["none", "plateau"], default="plateau")
    parser.add_argument("--lr-scheduler-patience", type=int, default=3)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='balanced')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--predict-test', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--exclude-file-id-suffix-2', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir = make_experiment_dir(args.output_dir, args.exp_name)
    device = torch.device('cuda' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    print(f'start video_cnn device={device.type} window_label_mode={args.window_label_mode} window_size={args.window_size} stride={args.stride} video_window_size={args.video_window_size} video_frame_selection={args.video_frame_selection} video_feature_mode={args.video_feature_mode}')

    records = load_video_records(args.root, exclude_file_id_suffix_2=args.exclude_file_id_suffix_2)
    val_subjects = set(args.val_subjects)
    train_records = [record for record in records if record['sbj_id'] not in val_subjects]
    val_records = [record for record in records if record['sbj_id'] in val_subjects]
    print(f"train_subjects={sorted({r['sbj_id'] for r in train_records})}")
    print(f"val_subjects={sorted({r['sbj_id'] for r in val_records})}")

    train_ds = WindowedVideoDataset(train_records, args, 'train')
    val_ds = WindowedVideoDataset(val_records, args, 'val')
    for sample in train_ds.samples:
        sample['video'] = build_video_temporal_features(sample['video'], args.video_feature_mode)
    for sample in val_ds.samples:
        sample['video'] = build_video_temporal_features(sample['video'], args.video_feature_mode)
    train_meta = train_ds.metadata_frame()
    val_meta = val_ds.metadata_frame()
    print(f"dataset train_samples={len(train_ds)} val_samples={len(val_ds)} train_classes={np.bincount(train_meta['y_true'].to_numpy(dtype=np.int64), minlength=N_CLASSES).tolist()} val_classes={np.bincount(val_meta['y_true'].to_numpy(dtype=np.int64), minlength=N_CLASSES).tolist()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')

    input_feature_dim = train_ds.samples[0]['video'].shape[1]
    model = VideoCNN(input_feature_dim=input_feature_dim, proj_dim=args.proj_dim, channels=args.channels, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
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
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr, **val_metrics})
        print(f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_metrics['acc']:.6f} val_macro_f1={val_metrics['macro_f1']:.6f} val_weighted_f1={val_metrics['weighted_f1']:.6f}")
        if val_metrics['macro_f1'] > best_score:
            best_score = val_metrics['macro_f1']
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
            torch.save({'model_state_dict': best_state, 'args': vars(args), 'best_epoch': best_epoch, 'best_macro_f1': best_score}, args.output_dir / 'video_cnn_best.pt')
            print(f'saved best epoch={epoch} macro_f1={best_score:.6f}')
        else:
            bad_epochs += 1
            print(f'bad_epochs={bad_epochs}/{args.early_stopping_rounds}')
        if bad_epochs >= args.early_stopping_rounds:
            print('early stopping')
            break

    if best_state is None:
        raise RuntimeError('Training did not produce a best checkpoint')

    model.load_state_dict(best_state)
    _, train_prob, y_train = predict_probabilities(model, DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda'), device)
    _, val_prob, y_val = predict_probabilities(model, val_loader, device)
    train_pred = train_prob.argmax(axis=1)
    val_pred = val_prob.argmax(axis=1)
    train_metrics = compute_metrics(y_train, train_pred)
    val_metrics = compute_metrics(y_val, val_pred)
    print(f'final_best epoch={best_epoch} train_metrics={train_metrics} val_metrics={val_metrics}')

    save_eval_outputs(args.output_dir, 'train', train_meta, train_prob)
    save_eval_outputs(args.output_dir, 'val', val_meta, val_prob)
    pd.DataFrame(history).to_csv(args.output_dir / 'history.csv', index=False)
    (args.output_dir / 'summary.json').write_text(json.dumps({'best_epoch': best_epoch, 'train_metrics': train_metrics, 'val_metrics': val_metrics}, indent=2))

    if args.predict_test:
        test_ds = TestVideoDataset(
            args.root,
            video_frame_selection=args.video_frame_selection,
            video_feature_mode=args.video_feature_mode,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
        test_ids, test_prob, _ = predict_probabilities(model, test_loader, device)
        np.save(args.output_dir / 'test_probabilities.npy', test_prob.astype(np.float32))
        test_ds.meta.to_csv(args.output_dir / 'test_meta.csv', index=False)
        output_path = args.output_dir / 'submission_cnn_video.csv'
        save_submission(args.root, output_path, test_ids, test_prob)
        print(f'saved submission: {output_path}')
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
            "checkpoint": args.output_dir / "video_cnn_best.pt",
            "test_probabilities": args.output_dir / "test_probabilities.npy" if args.predict_test else None,
            "test_meta": args.output_dir / "test_meta.csv" if args.predict_test else None,
            "submission": output_path if args.predict_test else None,
        },
        metrics={"train": train_metrics, "val": val_metrics},
        extra={"best_epoch": best_epoch},
    )
    print(f'saved outputs: {args.output_dir}')


if __name__ == '__main__':
    main()
