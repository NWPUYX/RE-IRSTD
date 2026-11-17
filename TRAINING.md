# Training provenance and research source

The paper's final checkpoint was obtained through a multi-stage source-domain development pipeline. `training_recipe.json` describes the last stage only. It is **not** a standalone from-scratch recipe that regenerates the released detector in two epochs.

`training_sources.zip` contains the project-authored architecture, loss, augmentation, training and ablation programs in their historical relative directory structure. The `source/` entries preserve original source bytes; `additional_source/` includes stage dependencies and analysis programs. It also contains the full checkpoint configuration/parent graph in `provenance/training_lineage.json`, and portable split records. The original root prefix in lineage metadata is replaced by `${WORKSPACE}`. Historical source may still contain local paths; these are provenance, not a portable inference interface. Use the top-level release scripts for inference and evaluation.

To inspect the research implementation:

```bash
python -m zipfile -e training_sources.zip research
```

The original layout has sibling `IRSTD` and `Scenario_adaptability` directories. Merge the `source/` and `additional_source/` trees under a fresh workspace, retaining relative paths. Install the core package from `IRSTD` and add the workspace's `Scenario_adaptability` and `Scenario_adaptability/experiments/IRSTD_research_20261107` to the Python module search path. Recreate `Scenario_adaptability/splits` using the released content manifest and paths to your own legally obtained images. Restore stage configurations from the lineage records and relocate dataset, initialization, teacher and output paths to this fresh workspace. Do not run historical scripts against an existing experiment directory, because some entries resume or overwrite outputs.

## Main detector lineage

The exact stage entry points, executed epochs/updates, and selected descendant inputs are in `training_entry_points.json`. The order is:

1. Initial detector (`train_irstd.py`); a selected early checkpoint initializes the next stage.
2. Geometry repair (`irstd_geometry_repair.run`).
3. Balanced geometry (`irstd_geometry_balance.run`).
4. Target-size routing (`irstd_t_size.run`).
5. Native precision (`irstd_native_precision.run`).
6. Guarded background (`irstd_background_center.run`).
7. Head interpolation (`background_interpolation.py --refine`); no optimization updates.
8. Robust target training (`irstd_robust_t.run`).
9. Source-population calibration (`population_T_source_screen.py`); no optimization updates.
10. Target correspondence (`irstd_population_t.run`).
11. Scene-scale training (`irstd_scene_scale.run`).
12. Integral statistics joint training (`integral_joint_training.py`).
13. Peak-calibrated training (`peak_calibrated_training.py`).
14. Consistent-teacher training (`consistent_teacher_training.py`).

The final two-epoch stage executed 4,182 updates; the selected checkpoint has stored epoch index 1. Full-stage budgets and the distinction between executed and retained training are recorded in the lineage and technical supplement. Teacher checkpoints are dependencies, not extra inference models. The released inference uses one model.

Historical intermediate/teacher weights and large optimizer states are not bundled. Rebuilding the entire curriculum requires producing the upstream checkpoints and source-calibration statistics described in the lineage before running downstream entries. The release provides source provenance and final inference equivalence; it does not assert an independently replayed, one-command full training reproduction.

## Structural ablations

The research archive includes the structural ablation runner and configurations. These use the teacher-free matched 12-epoch recipe with paired seeds 3407, 3408 and 3409. Do not initialize them from the final multi-stage weights or compare their seed variance as if it were variance of the complete main-detector curriculum. Refer to the paper's ablation table and supplement for the route/statistics/supervision settings.

The same-cell supervision policy is explicit: frozen annotations are retained, and colliding labels participate in shared regression. Loss denominators must retain the original target counts. The original standalone v1 rejection policy is not the final benchmark policy.

## Release validation

Five CPU tests cover checkpoint loading and finite outputs, constant-image evidence, deterministic plateau representatives, shared main-head anchors, and perfect synthetic box metrics. Separate release preparation compared all final-model tensors, image preprocessing, output maps and decoded proposals against the frozen implementation, on one source validation frame and one non-square synthetic input. Both cases had zero maximum absolute output difference. `release_validation.json` records this result. These checks do not constitute an additional benchmark evaluation or full training replay.
