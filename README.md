# WEAR2026-UEC-dx2

## Reproducing the Final Ensemble

The commands and paths used to reproduce the final ensemble are documented in:

```text
reproducibility/20260703_171425_exp024_lgbm181019_video_xceptiontime155306_6model_beam_cv5/paper_reproduction_manifest.md
```

This manifest contains the commands for training the six base models, running the required post-processing steps, and constructing the final ensemble.

To regenerate the final ensemble using the saved base-model artifacts, run:

```bash
bash reproducibility/20260703_171425_exp024_lgbm181019_video_xceptiontime155306_6model_beam_cv5/reproduce_paper_ensemble_from_artifacts.sh
```

To reproduce the full pipeline from scratch, first run the base-model training commands listed in `paper_reproduction_manifest.md`. Then update the downstream post-processing and ensemble paths to point to the newly generated timestamped output directories.
