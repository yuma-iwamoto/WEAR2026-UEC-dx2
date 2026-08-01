# Paper Reproduction Manifest

This file records the commands and artifacts used to reproduce the paper ensemble output.

## Notes

- Base training commands create timestamped output directories; downstream paths must be updated to the newly created directories when retraining from scratch.
- The ensemble command below exactly reproduces the reported ensemble from the saved base-model artifacts.
- OOF evaluation is computed over concatenated subject-wise CV predictions, not as a mean of fold scores.

## Base Model Training

### Ensemble input 1: Inertial-XceptionTime+Video-MLP

- Run directory: `/workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_130412_exp024_xceptiontime_nf16_video_drop055_sqrt_cv5`
- Metadata: `/workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_130412_exp024_xceptiontime_nf16_video_drop055_sqrt_cv5/run_metadata.json`

```bash
python -m approach_base.src.run_subject_cv --module approach_base.src.train_exp024_cnn8_videomae_aux --root /workspace/input/3rd-wear-dataset-challenge-hasca-2026 --output-dir /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux --num-folds 5 --exclude-file-id-suffix-2 --exp-name exp024_xceptiontime_nf16_video_drop055_sqrt_cv5 --window-size 50 --stride 25 --window-label-mode purity --min-label-purity 0.8 --sensor-keys ra rl ll la --precompute-features --video-hz 30 --video-window-size 15 --sensor-emb-dim 8 --inertial-backbone xceptiontime --xception-nf 16 --xception-kernel-size 40 --xception-adaptive-size 8 --cnn-base-channels 64 --cnn-dropout 0.25 --video-hidden-dim 128 --video-out-dim 128 --video-dropout 0.55 --classifier-hidden 256 --classifier-dropout 0.50 --epochs 50 --batch-size 1024 --learning-rate 5e-4 --weight-decay 2e-3 --early-stopping-rounds 5 --class-weight sqrt --label-smoothing 0.07 --scheduler-patience 2 --scheduler-factor 0.5 --device gpu --num-workers 4 --log-every 50 --predict-test
```

### Ensemble input 2: Inertial-CNN+Video-MLP

- Run directory: `/workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5`
- Metadata: `/workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5/run_metadata.json`

```bash
python -m approach_base.src.run_subject_cv --module approach_base.src.train_exp024_cnn8_videomae_aux --output-dir /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux --exp-name exp024_reg_mid_video_drop055_cv5 --num-folds 5 --exclude-file-id-suffix-2 --root /workspace/input/3rd-wear-dataset-challenge-hasca-2026 --weight-decay 2e-3 --classifier-dropout 0.50 --video-dropout 0.55 --cnn-dropout 0.25 --label-smoothing 0.07 --early-stopping-rounds 5 --scheduler-patience 2 --scheduler-factor 0.5
```

### Ensemble input 3: Inertial-LightGBM

- Run directory: `/workspace/approach_base/output/cv_inertial_gbdt_lgbm/20260701_181019_lgbm_imu_all_features_sensor_balanced_cv5`
- Metadata: `/workspace/approach_base/output/cv_inertial_gbdt_lgbm/20260701_181019_lgbm_imu_all_features_sensor_balanced_cv5/run_metadata.json`

```bash
python -m approach_base.src.run_subject_cv --module approach_base.src.train_inertial_gbdt --output-dir /workspace/approach_base/output/cv_inertial_gbdt_lgbm --exp-name lgbm_imu_all_features_sensor_balanced_cv5 --num-folds 5 --model lightgbm --sensor-embedding-mode sensor --sensor-keys ra rl ll la --window-size 50 --stride 25 --window-label-mode purity --min-label-purity 0.8 --normalization-mode none --smoothing-window 5 --iterations 3000 --learning-rate 0.03 --depth 5 --num-leaves 31 --class-weight balanced --device gpu --build-workers 4 --exclude-file-id-suffix-2
```

### Ensemble input 4: Video-CNN

- Run directory: `/workspace/approach_base/output/cv_video_cnn/20260625_202639_first_mid_last_delta_concat`
- Metadata: `/workspace/approach_base/output/cv_video_cnn/20260625_202639_first_mid_last_delta_concat/run_metadata.json`

```bash
python -m approach_base.src.run_subject_cv --module approach_base.src.train_video_cnn --output-dir /workspace/approach_base/output/cv_video_cnn --exp-name first_mid_last_delta_concat --num-folds 5 --exclude-file-id-suffix-2 --video-frame-selection first_mid_last --video-feature-mode delta_concat
```

### Ensemble input 5: Video-MLP

- Run directory: `/workspace/approach_base/output/cv_video_mlp/20260625_204407_first_mid_last_delta_concat`
- Metadata: `/workspace/approach_base/output/cv_video_mlp/20260625_204407_first_mid_last_delta_concat/run_metadata.json`

```bash
python -m approach_base.src.run_subject_cv --module approach_base.src.train_video_mlp --output-dir /workspace/approach_base/output/cv_video_mlp --exp-name first_mid_last_delta_concat --num-folds 5 --exclude-file-id-suffix-2 --video-frame-selection first_mid_last --video-feature-mode delta_concat
```

### Ensemble input 6: Inertial-XceptionTime

- Run directory: `/workspace/approach_XceptionTime/output/cv_xceptiontime/20260701_155306_reproduce_20260623_095214_xceptiontime_cv5`
- Metadata: `/workspace/approach_XceptionTime/output/cv_xceptiontime/20260701_155306_reproduce_20260623_095214_xceptiontime_cv5/run_metadata.json`

```bash
python -m approach_XceptionTime.src.run_subject_cv --output-dir /workspace/approach_XceptionTime/output/cv_xceptiontime --exp-name reproduce_20260623_095214_xceptiontime_cv5 --num-folds 5 --model xceptiontime --sensor-keys ra rl ll la --window-size 50 --stride 25 --window-label-mode purity --min-label-purity 0.8 --normalization-mode global --add-magnitude --add-diff --add-diff-magnitude --add-diff-5 --add-diff-5-magnitude --add-diff-10 --add-diff-10-magnitude --channels 48 --kernel-size 41 --adaptive-size 8 --epochs 80 --batch-size 128 --learning-rate 0.001 --weight-decay 0.0001 --dropout 0.0 --early-stopping-rounds 8 --class-weight none --device gpu --num-workers 8 --seed 42 --exclude-file-id-suffix-2
```

## Post-processing

### align_oof_to_exp024_reference:20260701_181019_lgbm_imu_all_features_sensor_balanced_cv5

Align OOF rows to the multimodal reference OOF row order used by the ensemble.

```bash
python -m approach_base.src.align_oof_to_reference --source-dir /workspace/approach_base/output/cv_inertial_gbdt_lgbm/20260701_181019_lgbm_imu_all_features_sensor_balanced_cv5 --reference-dir /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5 --output-dir /workspace/approach_base/output/cv_inertial_gbdt_lgbm/20260701_181019_lgbm_imu_all_features_sensor_balanced_cv5 --output-prefix exp024ref
```

### expand_video_oof_to_sensor_level:20260625_202639_first_mid_last_delta_concat

Assign video-only OOF probabilities to the corresponding sensor-level rows.

```bash
python -m approach_base.src.expand_video_to_sensor_oof --video-dir /workspace/approach_base/output/cv_video_cnn/20260625_202639_first_mid_last_delta_concat --reference-dir /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5 --output-dir /workspace/approach_base/output/cv_video_cnn/20260625_202639_first_mid_last_delta_concat
```

### expand_video_oof_to_sensor_level:20260625_204407_first_mid_last_delta_concat

Assign video-only OOF probabilities to the corresponding sensor-level rows.

```bash
python -m approach_base.src.expand_video_to_sensor_oof --video-dir /workspace/approach_base/output/cv_video_mlp/20260625_204407_first_mid_last_delta_concat --reference-dir /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5 --output-dir /workspace/approach_base/output/cv_video_mlp/20260625_204407_first_mid_last_delta_concat
```

### align_oof_to_exp024_reference:20260701_155306_reproduce_20260623_095214_xceptiontime_cv5

Align OOF rows to the multimodal reference OOF row order used by the ensemble.

```bash
python -m approach_base.src.align_oof_to_reference --source-dir /workspace/approach_XceptionTime/output/cv_xceptiontime/20260701_155306_reproduce_20260623_095214_xceptiontime_cv5 --reference-dir /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5 --output-dir /workspace/approach_XceptionTime/output/cv_xceptiontime/20260701_155306_reproduce_20260623_095214_xceptiontime_cv5 --output-prefix exp024ref
```

### expand_video_test_to_sensor_level:20260625_202639_first_mid_last_delta_concat

Assign video-only test probabilities to the corresponding sensor-level rows.

```bash
python -m approach_base.src.expand_video_to_sensor_test --video-dir /workspace/approach_base/output/cv_video_cnn/20260625_202639_first_mid_last_delta_concat --reference-dir /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5 --output-dir /workspace/approach_base/output/cv_video_cnn/20260625_202639_first_mid_last_delta_concat
```

### expand_video_test_to_sensor_level:20260625_204407_first_mid_last_delta_concat

Assign video-only test probabilities to the corresponding sensor-level rows.

```bash
python -m approach_base.src.expand_video_to_sensor_test --video-dir /workspace/approach_base/output/cv_video_mlp/20260625_204407_first_mid_last_delta_concat --reference-dir /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5 --output-dir /workspace/approach_base/output/cv_video_mlp/20260625_204407_first_mid_last_delta_concat
```

## Ensemble

```bash
python -m approach_base.src.ensemble --output-dir /workspace/approach_base/output/ensemble --exp-name exp024_lgbm181019_video_xceptiontime155306_6model_beam_cv5 --val-prob-files /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_130412_exp024_xceptiontime_nf16_video_drop055_sqrt_cv5/oof_probabilities.npy /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5/oof_probabilities.npy /workspace/approach_base/output/cv_inertial_gbdt_lgbm/20260701_181019_lgbm_imu_all_features_sensor_balanced_cv5/exp024ref_oof_probabilities.npy /workspace/approach_base/output/cv_video_cnn/20260625_202639_first_mid_last_delta_concat/sensor_level_oof_probabilities.npy /workspace/approach_base/output/cv_video_mlp/20260625_204407_first_mid_last_delta_concat/sensor_level_oof_probabilities.npy /workspace/approach_XceptionTime/output/cv_xceptiontime/20260701_155306_reproduce_20260623_095214_xceptiontime_cv5/exp024ref_oof_probabilities.npy --test-prob-files /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_130412_exp024_xceptiontime_nf16_video_drop055_sqrt_cv5/cv_test_probabilities.npy /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5/cv_test_probabilities.npy /workspace/approach_base/output/cv_inertial_gbdt_lgbm/20260701_181019_lgbm_imu_all_features_sensor_balanced_cv5/cv_test_probabilities.npy /workspace/approach_base/output/cv_video_cnn/20260625_202639_first_mid_last_delta_concat/sensor_level_cv_test_probabilities.npy /workspace/approach_base/output/cv_video_mlp/20260625_204407_first_mid_last_delta_concat/sensor_level_cv_test_probabilities.npy /workspace/approach_XceptionTime/output/cv_xceptiontime/20260701_155306_reproduce_20260623_095214_xceptiontime_cv5/cv_test_probabilities.npy --val-target-file /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5/oof_predictions.csv --test-id-file /workspace/approach_base/output/cv_exp024_cnn8_videomae_aux/20260701_121613_exp024_reg_mid_video_drop055_cv5/cv_test_meta.csv --weight-mode auto --auto-search-mode beam --beam-width 8 --beam-steps 250
```
