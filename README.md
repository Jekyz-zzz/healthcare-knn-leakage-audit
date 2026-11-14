# KNN on the Kaggle Healthcare Dataset: A Signal and Leakage Audit

This project evaluates K-Nearest Neighbors (KNN), with XGBoost as a comparison, on the [Kaggle Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) to predict `Test Results` (Normal / Abnormal / Inconclusive).

## Key findings

- **No learnable signal.** After cleaning, all models perform at chance level (~33% accuracy, F1 macro ~0.33). A permutation test confirms that neither KNN (p = 0.18) nor XGBoost (p = 0.89) performs better than models trained on shuffled labels.
- **5,500 duplicate rows cause data leakage.** The raw data contains 534 exact duplicates and 4,966 near-duplicates (identical except for `Age`, shifted by 1–5 years), all with identical labels.
- **Leakage inflates accuracy from 33% to 44%.** Without cleaning, KNN (k = 1) reaches 43.9% accuracy, exactly matching a simple theoretical prediction based on the duplicate rate.
- **Leakage also fools tuning and statistical tests.** On raw data, hyperparameter tuning always selects k = 1, and the permutation test reports p = 0.0099, making a model that learned nothing appear significant.

## Repository contents

| File | Description |
|---|---|
| `KNN_Kelompok2.ipynb` | Full analysis: data audit, EDA, preprocessing, training, evaluation, and leakage experiments |
| `requirements.txt` | Library versions used to produce the results |

## How to run

1. Download `healthcare_dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/prasad22/healthcare-dataset) and place it in the same folder as the notebook.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run all cells in `KNN_Kelompok2.ipynb`.

Results were produced with scikit-learn 1.6.1, pandas 2.2.3, and XGBoost 3.4.1. Note that scikit-learn 1.6.1 has a bug where KNN with uniform weights and Manhattan distance fails on string labels; the notebook uses integer-encoded labels to avoid it.

## Context

Final project for the Machine Learning course, Teknik Informatika, Universitas Sriwijaya.
