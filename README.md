# Heart Attack XAI Research Project

Topic:
Explainable Machine Learning for Heart Attack Risk Prediction Using Real-World Health Data

Dataset:
CDC BRFSS 2024 official real-world public health survey data.

## What you need to do first

1. Download the official CDC BRFSS 2024 SAS Transport ZIP file.
2. Extract it.
3. Put `LLCP2024.XPT` inside:

```text
data/raw/LLCP2024.XPT
```

Then run:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_dataset.py
python scripts/train_baseline_models.py
```

## Main target variable

```text
CVDINFR4
```

Meaning:
Ever told you had a heart attack, also called myocardial infarction?

Target mapping:

```text
1 = Heart attack Yes
2 = Heart attack No
7/9/missing = removed
```

## Columns avoided due to leakage

Do not use these as input features:

```text
CVDINFR4
_MICHD
CVDCRHD4
```

## Output files

After running the scripts, you will get:

```text
data/processed/heart_attack_cleaned.csv
outputs/tables/dataset_summary.csv
outputs/tables/model_results.csv
outputs/figures/class_distribution.png
outputs/figures/confusion_matrix_logistic_regression.png
```
