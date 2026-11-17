# RE-IRSTD

Code and pretrained weights for **Single-Frame Cross-Scene Infrared Small-Target Detection via Local Relative Evidence and Background-Conditioned Filtering**.

RE-IRSTD uses annular relative evidence, background-conditioned zero-sum filtering, and a full-resolution target pathway. Source-population moments are fixed in the target pathway; context and fusion retain online group normalization. The released detector has **808,952 learned parameters**. Integral statistics accumulate in FP64 and feed FP32 network features. Main-head peak locations are retained when the main and auxiliary logits are combined with weights 0.75 and 0.25.

## Install

Python 3.11 or newer is required. Install matching PyTorch and torchvision builds for your hardware, then:

```bash
git clone https://github.com/NWPUYX/RE-IRSTD.git
cd RE-IRSTD
python -m pip install -r requirements.txt
python -m unittest -v test_release.py
```

The release was validated with PyTorch 2.14.0+cu132. CPU smoke tests and CUDA comparisons against the frozen research implementation passed. The full training curriculum was not replayed during release preparation.

## Single-image inference

```bash
python predict.py --image /path/to/image.png --device cpu --output outputs/prediction.json
```

For CUDA use `--device cuda`. The default confidence threshold, **0.5791078209877014**, was selected on the source validation set. Images follow the frozen RGB bicubic 512×512 letterbox with fill 128, followed by grayscale conversion and division by 255. Boxes use a 16×16 source reference, score floor 1e-5, top 1,000 candidates, IoU-0.5 NMS, and at most 100 detections. Output coordinates are original-image `xyxy`. `--threshold` changes only the output operating point; do not tune it on target test labels when reproducing the paper.

`re_irstd.pt` is a weights-only state dictionary. All tensors equal those in the frozen paper checkpoint. `checkpoint_provenance.json` records both file hashes; the different serialization excludes historical non-model metadata.

## Data and evaluation

Third-party images are obtained from their owners, including the reconstructed collections distributed through [MoPKL](https://github.com/UESTC-nnLab/MoPKL). Images are not redistributed here. `dataset_file_manifest.csv.gz` specifies the exact relative identities, splits, groups, boxes, and file/pixel/annotation hashes used by the paper. Arrange your legally obtained images under a common root according to its `relative_id` column.

```bash
python prepare_manifest.py --domain IRDST-H --split test --data-root /path/to/data --verify-hashes --output outputs/IRDST-H_test.json
python evaluate.py --manifest outputs/IRDST-H_test.json --data-root /path/to/data --device cuda --output outputs/IRDST-H_metrics.json
```

Evaluation uses greedy one-to-one box matching at ten IoU thresholds from 0.50 to 0.95, 101-point interpolated AP, the frozen source F1 threshold, and 3-pixel center matching. The optional group bootstrap uses 1,000 draws and seed 3407. AP/F1 in JSON and CSV are fractions; multiply by 100 for percentages. Empty prediction lists are valid. The supplied `domain_detection_metrics.csv` contains the frozen main-paper counts and metrics, not new release-time benchmark runs.

## Reported external-domain results

| Domain | AP50 (%) | F1 (%) | Box FP/frame |
|---|---:|---:|---:|
| IRDST-H | 15.41 | 17.02 | 0.163 |
| IRSTD-UAV | 1.31 | 3.80 | 0.694 |
| ITSDT-15K | 0.04 | 0.03 | 0.318 |
| Macro average | 5.59 | 6.95 | 0.392 |

These are the development benchmark results reported in the manuscript. The main multi-stage detector and the teacher-free three-seed structural ablations use different training recipes; they must not be conflated. See `TRAINING.md` and the paper's technical supplement for the stage dependencies and selection rules.

## Files

```text
RE-IRSTD/
├── model.py, model_config.json       # Final paper detector
├── integral_evidence.py              # FP64 annular integral statistics
├── population.py                    # Target source-population normalization
├── backbone.py, blocks.py, heads.py  # Learned architecture
├── relative_evidence.py              # Evidence buffers / reference implementation
├── decode.py, plateaus.py            # Main-anchor-preserving readout
├── predict.py, evaluate.py           # Portable inference and evaluation
├── evaluation.py                    # Frozen metric and postprocessing rules
├── prepare_manifest.py              # Portable split preparation
├── re_irstd.pt                       # Final pretrained state dictionary
├── training_sources.zip             # Research training / ablation source archive
├── training_recipe.json              # Final-stage recipe
├── training_entry_points.json        # Actual 14-stage lineage entry points
├── dataset_file_manifest.csv.gz      # Dataset identities and frozen annotations
├── domain_detection_metrics.csv      # Frozen results and counts
├── test_release.py                   # CPU release checks
└── release_validation.json           # Original/release numerical comparison
```

## License

Project-authored code and released model weights are provided under the MIT license. Third-party datasets, packages, and baseline implementations retain their respective licenses; no ownership of those materials is claimed. See `THIRD_PARTY.md`.
