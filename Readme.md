# Adaptive Class Weighting for Imbalanced Classification
### Work in Progress

## Gemini Suggested General Folder Structure?
```text
project-name/
├── configs/                    # Declarative YAML configs (Hydra / OmegaConf)
│   ├── config.yaml             # Main entry-point config defaults
│   ├── data/                   # Dataset & DataLoader configs (e.g., fraud_tabular.yaml)
│   ├── model/                  # Architecture hyperparams (e.g., resnet.yaml, mlp.yaml)
│   ├── optim/                  # Optimizer & LR scheduler settings
│   └── experiment/             # Full reproducible ablation overrides (e.g., exp01_focal_loss.yaml)
│
├── data/                       # Git-ignored (except README/DVC files)
│   ├── raw/                    # Immutable original downloads (read-only)
│   ├── interim/                # Cleaned, joined, or deduplicated intermediates
│   └── processed/              # Tensor-ready formats (Parquet, HDF5, WebDataset, memmap)
│
├── notebooks/                  # Exploratory Data Analysis (EDA) & visualization
│   └── 01_eda_distributions.ipynb
│
├── src/
│   └── my_project/             # Importable Python package
│       ├── __init__.py
│       ├── data/
│       │   ├── datasets.py     # Custom torch.utils.data.Dataset classes
│       │   ├── transforms.py   # Feature engineering & tensor augmentations
│       │   └── datamodule.py   # Train/val/test split & DataLoader factory
│       ├── models/
│       │   ├── backbones/      # Reusable encoders / feature extractors
│       │   ├── heads/          # Task-specific classification/regression heads
│       │   └── losses.py       # Custom loss functions (e.g., FocalLoss)
│       ├── engine/
│       │   ├── trainer.py      # Training & validation loops, AMP, gradient clipping
│       │   ├── evaluator.py    # Inference, threshold tuning, & test metrics
│       │   └── callbacks.py    # Early stopping, checkpointing, W&B / MLflow logging
│       └── utils/
│           ├── seed.py         # Deterministic seeding utilities
│           ├── registry.py     # Component decorator registry
│           └── io.py           # Safe checkpoint saving/loading
│
├── scripts/                    # Thin CLI entry points (no heavy logic here)
│   ├── preprocess.py           # Converts data/raw -> data/processed
│   ├── train.py                # Parses config, builds components, runs Trainer
│   └── evaluate.py             # Loads checkpoint, generates final paper tables/plots
│
├── tests/                      # Fast unit & shape-verification tests (pytest)
│   ├── test_models.py          # Forward-pass tensor shape assertions
│   └── test_datasets.py        # Data leakage & schema checks
│
├── outputs/                    # Git-ignored artifacts: checkpoints, logs, hydra dumps
├── pyproject.toml              # Dependencies, ruff/mypy config, package metadata
└── README.md                   # Setup instructions, dataset attribution, reproduction commands
```

## Overview
This project implements and evaluates machine learning pipelines for classification with imbalanced data across three benchmark tabular datasets:
1. **Bank Account Fraud (BAF) Dataset Suite (NeurIPS 2022)** — Synthetic bank account opening fraud detection under temporal shift and controlled bias.
2. **ULB Credit Card Fraud Detection Dataset** — Anonymized PCA-transformed European credit card transaction fraud detection under extreme class imbalance.
3. **UCI Default of Credit Card Clients Dataset** — Credit card default prediction from demographic, billing, and repayment history records.

---

## Repository Structure & Data Setup

To respect upstream dataset licenses and avoid redistributing large or modified database files, **raw and preprocessed dataset files are not stored in this repository.**

```text
├── data/                   # Ignored by Git (place downloaded CSVs here)
├── src/                    # Preprocessing, training, and evaluation scripts
├── notebooks/              # Exploratory data analysis and model experiments
├── models/                 # Serialized model checkpoints (see licensing note below)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation, attributions, and licenses
```

### Downloading the Datasets
You can download the required datasets directly via the Kaggle CLI or `kagglehub`:

```bash
# 1. Bank Account Fraud Dataset Suite (NeurIPS 2022)
kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/baf --unzip

# 2. ULB Credit Card Fraud Detection
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ulb --unzip

# 3. UCI Default of Credit Card Clients Dataset
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset -p data/uci --unzip
```

---

## Dataset Attributions, Licenses & Disclaimers

This project uses three third-party datasets. Each dataset remains the intellectual property of its respective creators and is governed by its own license terms. **No original dataset creator or institution endorses this project or its findings.**

### 1. Bank Account Fraud (BAF) Dataset Suite (NeurIPS 2022)
* **Source:** [Kaggle (`sgpjesus/bank-account-fraud-dataset-neurips-2022`)](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) | [Official GitHub (`feedzai/bank-account-fraud`)](https://github.com/feedzai/bank-account-fraud)
* **Creators:** Sérgio Jesus, José Pombal, Duarte Alves, André Cruz, Pedro Saleiro, Rita P. Ribeiro, João Gama, and Pedro Bizarro (Feedzai Research).
* **License:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) *(Note: OpenReview supplementary documentation also references [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)).*
* **Usage & Modification Notice (CC 4.0 Compliance):**
  * Used strictly for **non-commercial academic and research purposes**.
  * Raw data files are not redistributed. During local execution, our pipeline applies the following transformations to the original data: `[e.g., train/validation/test temporal splitting by month, one-hot encoding of categorical variables, feature scaling, and class-weight adjustments / resampling]`.
  * Any derivative artifacts, evaluation reports, or model weights trained on the BAF dataset in this repository are shared exclusively under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

### 2. Credit Card Fraud Detection Dataset (MLG-ULB)
* **Source:** [Kaggle (`mlg-ulb/creditcardfraud`)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Creators:** Machine Learning Group (MLG) at Université Libre de Bruxelles (ULB) and Worldline (Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, Yann-Aël Le Borgne, Gianluca Bontempi, et al.).
* **License:** [Open Data Commons Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/) (with database structure governed by [ODbL v1.0](https://opendatacommons.org/licenses/odbl/1-0/) conventions and explicit academic citation requirements).
* **Usage & Modification Notice:**
  * Raw data files are not redistributed. Our local preprocessing pipeline applies `[e.g., RobustScaler/StandardScaler transformation on 'Time' and 'Amount', stratified train/test splitting, and undersampling/oversampling on training folds]`.

### 3. Default of Credit Card Clients Dataset (UCI / Kaggle)
* **Source:** [Kaggle (`uciml/default-of-credit-card-clients-dataset`)](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) | [UCI Machine Learning Repository (DOI: `10.24432/C55S3H`)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
* **Creators:** I-Cheng Yeh and Che-hui Lien (Department of Information Management, Chung Hua University & Department of International Business, Ming Chuan University / Tamkang University, Taiwan).
* **License:** Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) by the upstream UCI Machine Learning Repository (and mirrored on Kaggle under [CC0 1.0 Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)). We comply with the upstream **CC BY 4.0** attribution requirements.
* **Usage & Modification Notice:**
  * Raw data files are not redistributed. Our local preprocessing pipeline applies `[e.g., column renaming, consolidation of undocumented categorical levels in EDUCATION and MARRIAGE, feature scaling, and stratified splitting]`.

---

## Academic Citations (BibTeX)

If you reference or build upon this work, please cite the original dataset publications as required by their creators:

### Bank Account Fraud (BAF) Dataset Suite
```bibtex
@article{jesusTurningTablesBiased2022,
  title={Turning the {{Tables}}: {{Biased}}, {{Imbalanced}}, {{Dynamic Tabular Datasets}} for {{ML Evaluation}}},
  author={Jesus, S{\'e}rgio and Pombal, Jos{\'e} and Alves, Duarte and Cruz, Andr{\'e} and Saleiro, Pedro and Ribeiro, Rita P. and Gama, Jo{\~a}o and Bizarro, Pedro},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  volume={35},
  pages={33563--33575},
  year={2022}
}
```

### ULB Credit Card Fraud Detection Dataset
```bibtex
@inproceedings{dal2015calibrating,
  title={Calibrating Probability with Undersampling for Unbalanced Classification},
  author={Dal Pozzolo, Andrea and Caelen, Olivier and Johnson, Reid A. and Bontempi, Gianluca},
  booktitle={Symposium on Computational Intelligence and Data Mining (CIDM)},
  pages={159--166},
  year={2015},
  organization={IEEE}
}

@article{dal2014learned,
  title={Learned lessons in credit card fraud detection from a practitioner perspective},
  author={Dal Pozzolo, Andrea and Caelen, Olivier and Le Borgne, Yann-Ael and Waterschoot, Serge and Bontempi, Gianluca},
  journal={Expert Systems with Applications},
  volume={41},
  number={10},
  pages={4915--4928},
  year={2014},
  publisher={Pergamon}
}

@article{dal2018credit,
  title={Credit card fraud detection: a realistic modeling and a novel learning strategy},
  author={Dal Pozzolo, Andrea and Boracchi, Giacomo and Caelen, Olivier and Alippi, Cesare and Bontempi, Gianluca},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  volume={29},
  number={8},
  pages={3784--3797},
  year={2018},
  publisher={IEEE}
}

@article{carcillo2018scarff,
  title={SCARFF: a scalable framework for streaming credit card fraud detection with Spark},
  author={Carcillo, Fabrizio and Dal Pozzolo, Andrea and Le Borgne, Yann-A{\"e}l and Caelen, Olivier and Mazzer, Yannis and Bontempi, Gianluca},
  journal={Information Fusion},
  volume={41},
  pages={182--194},
  year={2018},
  publisher={Elsevier}
}

@article{carcillo2018streaming,
  title={Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization},
  author={Carcillo, Fabrizio and Le Borgne, Yann-A{\"e}l and Caelen, Olivier and Bontempi, Gianluca},
  journal={International Journal of Data Science and Analytics},
  volume={5},
  number={4},
  pages={285--300},
  year={2018},
  publisher={Springer}
}

@inproceedings{lebichot2019deep,
  title={Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection},
  author={Lebichot, Bertrand and Le Borgne, Yann-A{\"e}l and He, Liyun and Obl{\'e}, Frederic and Bontempi, Gianluca},
  booktitle={INNS Big Data and Deep Learning Conference},
  pages={78--88},
  year={2019},
  organization={Springer}
}

@article{carcillo2019combining,
  title={Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection},
  author={Carcillo, Fabrizio and Le Borgne, Yann-A{\"e}l and Caelen, Olivier and Obl{\'e}, Frederic and Bontempi, Gianluca},
  journal={Information Sciences},
  volume={557},
  pages={317--331},
  year={2019},
  publisher={Elsevier}
}

@book{leborgne2022fraud,
  title={Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook},
  author={Le Borgne, Yann-A{\"e}l and Siblini, Wissam and Lebichot, Bertrand and Bontempi, Gianluca},
  url={[https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook)},
  year={2022},
  publisher={Universit{\'e} Libre de Bruxelles}
}
```

### UCI Default of Credit Card Clients Dataset
```bibtex
@article{yeh2009comparisons,
  title={The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients},
  author={Yeh, I-Cheng and Lien, Che-hui},
  journal={Expert Systems with Applications},
  volume={36},
  number={2},
  pages={2473--2480},
  year={2009},
  publisher={Elsevier}
}

@misc{yeh2016default,
  author={Yeh, I-Cheng},
  title={{Default of Credit Card Clients}},
  year={2016},
  howpublished={UCI Machine Learning Repository},
  note={{DOI}: \url{[https://doi.org/10.24432/C55S3H](https://doi.org/10.24432/C55S3H)}}
}
```

---

## Project Code License

* **Source Code:** All original Python scripts and notebooks written for this are All Rights Reserved.
* **Datasets & BAF-Derived Artifacts:** The underlying datasets are **not** covered by the source code license and remain subject to their respective licenses listed above (`CC BY-NC-SA 4.0`, `ODC DbCL v1.0`, and `CC BY 4.0`). Any model weights or evaluation outputs derived from the Bank Account Fraud (BAF) dataset are shared strictly for non-commercial use under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).