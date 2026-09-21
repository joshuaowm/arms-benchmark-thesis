# ARMS Segmentation Benchmark

This repository contains the code for the master's thesis *Benchmarking Interactive and
Automated Segmentation Models on Autonomous Reef Monitoring Structures Images* (University of
Salzburg and University of South Brittany, 2026).

**Joshua Owen Mangotang**

Supervisors: Dr. Minh-Tan Pham and Dr. Hoàng-Ân Lê (University of South Brittany), Gella
Getachew Workineh (University of Salzburg)

## Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Reproducing the Thesis Results](#reproducing-the-thesis-results)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

Autonomous Reef Monitoring Structures (ARMS) are stacks of settlement plates used to monitor
hard-bottom marine biodiversity. After retrieval, every plate is photographed and the organisms
on it are annotated by experts. This benchmark measures how much of that annotation existing
segmentation models can take over, in two settings:

- **Interactive segmentation.** The annotator prompts one organism with clicks or a box, and
  the model returns its mask. Prompts are simulated from the ground-truth masks.
- **Automatic instance segmentation.** A detector finds, classifies and segments all
  organisms on a tile without any prompt.

Every model is trained under three regimes (Belgium, Crete, and both sites combined) and
evaluated on the test plates of both sites, which gives in-domain and cross-domain results for
each model. The train, validation and test splits are plate-disjoint and are the same for all
experiments.

### Dataset

The images and annotations are from ARMSDS (Hadjipieris et al., 2026), which covers two
ARMS-MBON observatories: Belgium (North Sea) and Crete (Mediterranean). Each plate image is
divided into 1024 x 1024 tiles.

|                     | Belgium | Crete  | Total  |
| ------------------- | ------: | -----: | -----: |
| Plates              | 20      | 18     | 38     |
| Tiles (1024 x 1024) | 400     | 432    | 832    |
| Annotations         | 5,362   | 11,276 | 16,638 |
| Taxa                | 26      | 45     | 57     |

The benchmark uses the five most frequent classes at each site. Serpulidae and Spirobranchus
occur at both sites, so the shared label space has eight classes.

| Site    | Classes                                                                      | Plates (train / val / test) |
| ------- | ---------------------------------------------------------------------------- | --------------------------- |
| Belgium | Serpulidae, Spirobranchus, Membraniporoidea, *Crepidula fornicata*, Hydrozoa | 12 / 3 / 5                  |
| Crete   | Spirorbinae, Serpulidae, Rhodophyta, Spirobranchus, *Ciona*                  | 11 / 3 / 4                  |

A second label set groups the taxa by phylum (WoRMS) and is used with YOLO11s-seg to study
the long tail of rare classes. Five phyla are kept at each site. Annelida, Bryozoa and
Mollusca occur at both sites, and the combined regime uses all seven phyla.

| Site    | Phyla                                             |
| ------- | ------------------------------------------------- |
| Belgium | Annelida, Arthropoda, Bryozoa, Cnidaria, Mollusca  |
| Crete   | Annelida, Bryozoa, Chordata, Mollusca, Rhodophyta  |

### Models

Interactive models (all at base size):

| Model       | Type       | Input size | Fine-tuned part      | Trainable params |
| ----------- | ---------- | ---------: | -------------------- | ---------------: |
| SAM-1-B     | generalist | 1024       | mask decoder         | 4.06M            |
| HQ-SAM-B    | generalist | 1024       | HQ output module     | 1.07M            |
| SAM2-B+     | generalist | 1024       | mask decoder         | 4.22M            |
| OSISeg-B    | generalist | 512        | adapters and decoder | 14.75M           |
| SegNext     | specialist | 1024       | neck, fusion, head   | 25.0M            |
| SimpleClick | specialist | 448        | neck, head           | 11.0M            |

Automatic models: Mask R-CNN with a ResNet-50 FPN backbone (detectron2), YOLO11s-seg and
YOLO26s-seg (Ultralytics). Each detector is trained at its native input size (640 for YOLO,
a 640 to 800 pixel short side for Mask R-CNN) and at 1024.

### Evaluation Protocol

- **One-pass prompts:** a single click (optionally with one negative click), a box with
  simulated annotator noise, and K clicks (K = 2 to 6) placed along the object skeleton or at
  random. Scores are macro IoU averaged over ten seeded prompt draws.
- **Iterative correction:** up to 20 clicks, each placed in the largest error region of the
  previous prediction, following the protocol of RITM (Sofiiuk et al., 2022). Scores are
  micro IoU at 1, 10 and 20 clicks, NoC@85 and NoF@85.
- **Automatic models:** macro matched-mask IoU (m-IoU), mAP and AP50, with predictions kept
  above a confidence of 0.5.

## Repository Structure

```
arms_benchmark/
├── arms/               shared library: prompt samplers, datasets, early stopping, taxonomy
├── configs/            OSISeg model and training configuration
├── env/                conda environment file and pip freeze
├── eval/               evaluation of the interactive models (multieval.py), latency benchmark
├── experiments/        frozen plate-disjoint splits (manifests) used for all results
├── notebooks/          results.ipynb (result tables) and visuals.ipynb (figures)
├── scripts/            data preparation, job launchers and figure scripts
│   ├── slurm/          SLURM jobs for training and evaluation
│   └── taskB/          automatic models: dataset conversion, training and scoring
├── third_party/        script that fetches the model repositories at the commits used, and
│                       the patched SimpleClick file
├── train/              one training script per interactive model
│   └── specialist/     fine-tuning code for SegNext and SimpleClick
└── LICENSE
```

Datasets, model weights and experiment outputs are not included in this repository.

## Installation

### Requirements

The experiments were run on a Linux SLURM cluster with NVIDIA GPUs, using:

- Python 3.10
- PyTorch 2.1.2 with CUDA 12.1, torchvision 0.16.2
- Ultralytics 8.4.60
- detectron2 (commit e0ec4e1)

### Setup

The scripts expect this repository in a folder named `arms_benchmark` inside a project folder
(`BASE` in the scripts), which also holds the data, the model code and the outputs:

```bash
mkdir arms && cd arms
git clone https://github.com/joshuaowm/arms-benchmark-thesis.git arms_benchmark
conda env create -f arms_benchmark/env/environment_torch-arms.yml
conda activate torch-arms
```

`env/pip_freeze_torch-arms.txt` lists the exact package versions. The environment file pins
CUDA builds; remove those packages on a machine without an NVIDIA GPU. SegNext and
SimpleClick run in their own environments (`rclicks` with mmcv 2.1.0 and `simpleclick` with
mmcv 1.7.2), as set in `scripts/slurm/eval_specialist.sbatch` and
`scripts/slurm/train_specialist.sbatch`.

Fetch the model code into `code/` in the project folder and install the fine-tuning configs
for SegNext and SimpleClick:

```bash
bash arms_benchmark/third_party/setup_third_party.sh
BASE=$PWD bash arms_benchmark/scripts/install_specialist_ft.sh
```

`setup_third_party.sh` clones OSISeg, HQ-SAM (which contains SAM 2 under `sam-hq2/`), SegNext
and SimpleClick at the commits used for the thesis, then applies two small compatibility
fixes: `np.int` becomes `np.int64` in OSISeg, and SimpleClick's albumentations import is
updated (`third_party/diffs/SimpleClick.diff`).

### Pretrained Weights

Download the official checkpoints to the following paths, relative to the project folder:

| Model                  | Path                                                        |
| ---------------------- | ----------------------------------------------------------- |
| SAM ViT-B (SAM, OSISeg) | `code/OSISeg/pre_weights/sam_vit_b_01ec64.pth`             |
| HQ-SAM ViT-B           | `code/sam-hq/pretrained_checkpoint/sam_hq_vit_b.pth`        |
| SAM 2.1 Hiera-B+       | `code/sam-hq/sam-hq2/checkpoints/sam2.1_hiera_base_plus.pt` |
| SegNext                | `code/SegNext/weights/vitb_sa2_cocolvis_hq44k_epoch_0.pth`  |
| SimpleClick            | `code/SimpleClick/weights/cocolvis_vit_base.pth`            |
| YOLO11s-seg            | `yolo11s-seg.pt`                                            |
| YOLO26s-seg            | `yolo26s-seg.pt`                                            |

The MD5 checksums of the YOLO weights used for the thesis are given in
`scripts/taskB/train_yolo.py`. Mask R-CNN is initialised from the detectron2 model zoo.

## Data Preparation

ARMSDS is not redistributed here. Place it in the project folder as follows:

```
arms/                                       project folder (BASE)
├── arms_benchmark/                         this repository
├── code/                                   model code (setup_third_party.sh)
├── dataset/deliverable_dataset/
│   ├── Belgium/full_plate/
│   │   ├── annotations.coco.json           plate-level annotations
│   │   ├── train_stitched/                 stitched plate images
│   │   └── grid_1024/                      1024 x 1024 tiles (build_plate_grid.py)
│   └── Crete/full_plate/                   same layout as Belgium
├── yolo11s-seg.pt
└── yolo26s-seg.pt
```

Tile the plates and build the combined-site COCO files:

```bash
python arms_benchmark/scripts/build_plate_grid.py --site Belgium
python arms_benchmark/scripts/build_plate_grid.py --site Crete
python arms_benchmark/scripts/build_manifests.py
python arms_benchmark/scripts/build_combined_allclass_coco.py
```

The splits used for all results are in `experiments/manifests_v11/` (species labels) and
`experiments/manifests_phylum/` (phylum labels), and all jobs read them from there.
`build_manifests.py` and `build_phylum_manifest.py` rebuild the splits into
`<project>/experiments/`, so the copies in this repository are never overwritten.
`build_manifests.py` is still needed because it also writes
`dataset/deliverable_dataset/grid_1024_combined.coco.json`.

The manifests store absolute paths to their COCO file (`coco`) and image folder
(`images_dir`). Update both fields to your data location before running any job.

The automatic models use their own copies of the splits in COCO and YOLO formats:

```bash
bash arms_benchmark/scripts/taskB/submit_taskB.sh build species
bash arms_benchmark/scripts/taskB/submit_taskB.sh build phylum
```

## Training and Evaluation

All thesis experiments are submitted to SLURM through `scripts/submit_experiments.sh`.
Without arguments it only prints the job list:

```bash
bash arms_benchmark/scripts/submit_experiments.sh                          # print jobs
bash arms_benchmark/scripts/submit_experiments.sh --go                     # submit all blocks
EXPERIMENTS="E5" bash arms_benchmark/scripts/submit_experiments.sh --go    # one block
```

The jobs are grouped into three blocks. The block names are internal experiment numbers and
do not follow the thesis chapters.

| Block | Experiments                                                                                                                                                                                            |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| E1234 | SAM-1-B, HQ-SAM-B, SAM2-B+ and OSISeg-B trained under each regime and evaluated on both sites; zero-shot evaluation of these four models and of SegNext and SimpleClick                                 |
| E5    | Mask R-CNN, YOLO11s-seg and YOLO26s-seg with species labels at both input sizes; YOLO11s-seg with phylum labels                                                                                         |
| E267  | One-pass prompt variants and 20-click iterative correction, with and without negative clicks, for HQ-SAM-B and SegNext zero-shot and for HQ-SAM-B fine-tuned in-domain (submit together with E1234) |

Each block uses the following jobs:

- `scripts/slurm/train_cell.sbatch` runs `scripts/run.sh`, which prepares the training crops
  (`scripts/prepare_crops.py`) and runs the training script in `train/`.
- `scripts/slurm/eval_cell.sbatch` prepares the test crops and prompts
  (`scripts/prepare_crops.py`, `scripts/prepare_test_prompts.py`) and runs
  `eval/multieval.py`.
- `scripts/slurm/eval_specialist.sbatch` and `scripts/slurm/train_specialist.sbatch` evaluate
  and fine-tune SegNext and SimpleClick.
- `scripts/taskB/submit_maskrcnn.sbatch` and `scripts/taskB/submit_yolo.sbatch` train and
  score the automatic models.

Checkpoints and results are written to `<project>/outputs/v15/`, prepared crops to
`<project>/prepared_v15/` and logs to `<project>/logs/v15/`.

Some results come from jobs that are submitted separately:

| Result                                         | Command                                                                                                                                  |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| K sweep (Table 4.3)                            | `sbatch --export=ALL,FAMILY=hqsam_vit_b,TRAIN_REGIME=<regime>,EVAL_SITE=<site>,ZEROSHOT=0,PROMPTS=ksweep scripts/slurm/eval_cell.sbatch` for Belgium to Belgium, Crete to Crete, combined to Belgium and combined to Crete |
| Fine-tuned SegNext and SimpleClick (Table 4.1) | `scripts/slurm/train_specialist.sbatch`, then `scripts/slurm/eval_specialist.sbatch` with `TRAIN_REGIME` set                               |
| YOLO at confidence 0.5 (Tables 4.5, 4.6)       | `scripts/taskB/submit_yolo.sbatch` with `EVAL_ONLY=1 SCORE_THR=0.5`                                                                       |
| Phylum m-IoU of the v15 runs (Table 4.7)       | `scripts/taskB/submit_yolo.sbatch` with `EVAL_ONLY=1 REMAP=1`                                                                             |
| Phylum mAP and AP50 (Table 4.7)                | E5 phylum jobs with `VTAG=v16tb`                                                                                                          |

The original YOLO runs were scored at the Ultralytics default confidence of 0.25 and
re-scored at 0.5. The default is now 0.5, so new E5 runs need no re-scoring. The two phylum
rows are explained under [Reproducibility Notes](#reproducibility-notes).

## Reproducing the Thesis Results

`notebooks/results.ipynb` reads the result files and produces the tables;
`notebooks/visuals.ipynb` and the scripts below produce the figures.

| Thesis                              | Notebook section                          | Source                                                                      |
| ----------------------------------- | ----------------------------------------- | --------------------------------------------------------------------------- |
| Tables 3.1, 3.2                     |                                           | tile annotations and `experiments/manifests_v11/`                           |
| Table 3.3                           | results.ipynb, Methods                    | training logs                                                               |
| Table 4.1                           | results.ipynb, 1 and 1b                   | E1234, fine-tuned SegNext and SimpleClick                                   |
| Table 4.2                           | results.ipynb, 2                          | E1234                                                                       |
| Table 4.3                           | results.ipynb, 3.1b                       | K sweep                                                                     |
| Table 4.4, Figure 4.3               | results.ipynb, 3.3 and 3.4                | E267; figure by `scripts/export_iter_curves.py`                             |
| Tables 4.5, 4.6                     | results.ipynb, 5, 5b and 5.1              | E5                                                                          |
| Table 4.7                           | results.ipynb, 6, 6b and 6.1              | m-IoU from the v15 phylum runs re-scored with `REMAP=1`; mAP and AP50 from `outputs/v16tb` |
| Figures 3.1 to 3.4, B.1, B.2        | visuals.ipynb                             | Figure 3.3 by `scripts/make_top10_stacked.py`                               |
| Figures 4.1, 4.2                    |                                           | `notebooks/regen_qual_figs.py` with `scripts/qual_targets.py` and `scripts/qual_masks.py` |

## Reproducibility Notes

**Cluster paths.** The code still contains the paths of the cluster it was run on:

- The launchers set `BASE=/share/castor/home/e2406747/axolotl`. `scripts/run.sh` reads
  `BASE` and `REPO` from the environment; the `.sbatch` files read `REPO` but set `BASE`
  themselves.
- Most Python scripts define `ROOT` near the top with the same path.
- `arms/paths.py` reads the model code folder from `ARMS_THIRD_PARTY`.
- The `.sbatch` headers, conda environments and log paths are specific to that cluster.

Replace these with your project folder before running.

**Notebooks.** The notebooks set `BASE = Path("..")` and were run from
`<project>/notebooks/`. Set `BASE` to the project folder in the first code cell when running
them from this repository. The first code cell under "Methods" in `results.ipynb` needs
`prepared_v15/`; the other cells only need the result files. Figures are written to
`<project>/notebooks/visuals_out/` and `<project>/thesis/figures/`.

**SimpleClick box prompt.** Table 4.1 reports SimpleClick box results from before a fix to
its zoom-in step: with a box and no clicks, the step had no region of interest and resized the
whole tile to 448 pixels. This code includes the fix. `results.ipynb` section 1b shows both
versions (macro IoU):

| Trained on | Belgium, thesis | Belgium, this code | Crete, thesis | Crete, this code |
| ---------- | --------------: | -----------------: | ------------: | ---------------: |
| zero-shot  | 36.1            | 48.1               | 30.4          | 50.8             |
| Belgium    | 36.2            | 50.1               | 31.0          | 53.5             |
| Crete      | 32.5            | 48.5               | 28.0          | 50.2             |
| combined   | 35.4            | 49.7               | 30.2          | 52.7             |

The click, skeleton and random prompt results are identical in both versions.

**Phylum class ids.** Each phylum manifest lists the site's own five phyla (`phyla`) and the
three phyla shared by both sites (`shared_phyla`). Class ids index the seven phyla of the
combined regime (`vocabulary`), so a shared phylum has the same id at both sites. The v15
phylum runs numbered each site's five phyla from 0 to 4 instead, with the same classes,
plates and annotations. For Table 4.7, their matched-mask IoU was therefore re-scored with
`REMAP=1`, which matches classes by name. The mAP and AP50 come from a retrain with the
shared ids (`outputs/v16tb/`), because COCO AP needs the same class ids at training and
evaluation. New runs with the provided manifests need no re-scoring.

**OSISeg training prompts.** The OSISeg dataset seeds each sample's prompt with Python's
`hash()` of the file name, which changes between runs unless `PYTHONHASHSEED` is set. The
scripts do not set it, so OSISeg training prompts are not exactly repeatable.

**Run-to-run variation.** Differences below 0.59 macro IoU for the interactive models and
below 5.78 matched-mask IoU for the automatic models are within run-to-run variation.

**Naming.** `manifests_v11` is the split created in development round v11 and used unchanged
since. Output folders keep their `v11gc_`, `v11_t1_` and `v12me_` prefixes because the
notebooks look up results by those names. All thesis results are from the v15 runs
(`VTAG=v15`).

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{mangotang2026arms,
  title  = {Benchmarking Interactive and Automated Segmentation Models on Autonomous Reef
            Monitoring Structures Images},
  author = {Mangotang, Joshua Owen},
  school = {University of Salzburg and University of South Brittany},
  type   = {Master's thesis},
  year   = {2026}
}
```

and the dataset:

```bibtex
@misc{hadjipieris2026armsds,
  title     = {{ARMSDS}},
  author    = {Hadjipieris, Andreas and Dimitriou, Neofytos and Conings, Bram and
               Abihssira Garc{\'i}a, Isabel Sof{\'i}a and Gkoulia, Andromachi},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.22042300}
}
```

## Acknowledgements

This work was carried out within the Erasmus Mundus Joint Master "Copernicus Master in
Digital Earth", co-funded by the European Union. The ARMS imagery comes from the ARMS-MBON
network, annotated in ARMSDS.

This code builds on
[Segment Anything](https://github.com/facebookresearch/segment-anything),
[HQ-SAM](https://github.com/SysCV/sam-hq),
[SAM 2](https://github.com/facebookresearch/sam2),
[OSISeg](https://github.com/zhilyzhang/OSISeg),
[SegNext](https://github.com/uncbiag/SegNext),
[SimpleClick](https://github.com/uncbiag/SimpleClick),
[detectron2](https://github.com/facebookresearch/detectron2) and
[Ultralytics](https://github.com/ultralytics/ultralytics). The iterative evaluation follows
RITM (Sofiiuk et al., *Reviving Iterative Training with Mask Guidance for Interactive
Segmentation*, ICIP 2022).

## License

This code is released under the [MIT License](LICENSE). The patched SimpleClick file in
`third_party/overlay/` remains under SimpleClick's MIT License, included as
`third_party/overlay/SimpleClick/LICENSE`. The model repositories fetched by
`setup_third_party.sh` and the other dependencies are not part of this repository and keep
their own licenses; note that Ultralytics, used for the YOLO models, is licensed under
AGPL-3.0.
