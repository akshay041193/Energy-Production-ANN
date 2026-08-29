# Combined Cycle Power Plant — Energy Output Prediction (PyTorch ANN)

A regression project that predicts the net hourly electrical energy output (`PE`) of a combined cycle power plant using a feedforward artificial neural network (ANN) built in PyTorch.

## Dataset

Dataset

The Combined Cycle Power Plant dataset — originally from the UCI Machine Learning Repository — contains 9,568 hourly measurements collected over 6 years (2006–2011) from a power plant operating at full load.
https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant

| Column | Renamed to | Description |
|---|---|---|
| `AT` | `Temperature` | Ambient temperature (°C) |
| `V` | `Velocity` | Exhaust vacuum (cm Hg) |
| `AP` | `Pressure` | Ambient pressure (millibar) |
| `RH` | `Humidity` | Relative humidity (%) |
| `PE` | `Produced_Energy` | Net hourly electrical energy output (MW) — **target variable** |

## Approach

1. **Data loading & cleaning** — load the CSV, rename columns for readability, check for missing values.
2. **Train/test split** — 80/20 split (`random_state=42`) performed *before* any scaling, to avoid data leakage.
3. **Feature scaling** — `StandardScaler` fit on the training set only, then applied to both train and test sets.
4. **PyTorch data pipeline** — features and targets converted to tensors, wrapped in `TensorDataset` and `DataLoader` (batch size 32).
5. **Model architecture** — a simple feedforward ANN:
   ```
   Input (4 features) → Linear(4, 6) → ReLU → Linear(6, 6) → ReLU → Linear(6, 1)
   ```
6. **Training** — 100 epochs, `Adam` optimizer, `MSELoss`. The best model (lowest validation loss) is checkpointed to `best_model.pt` during training.
7. **Evaluation** — MSE, RMSE, MAE, and R² computed on the held-out test set; training vs. validation loss curves plotted to check for overfitting.

## Results

The trained model is evaluated using standard regression metrics:

- **MSE** / **RMSE** — average squared/root-squared prediction error, in the same units as `PE` (MW) for RMSE.
- **MAE** — average absolute prediction error.
- **R²** — proportion of variance in `PE` explained by the model (closer to 1.0 is better).

## Results

| Metric | Training | Test |
|---|---|---|
| MSE  | 20.7806 | 19.1012 |
| RMSE | 4.5585 | 4.3704 |
| MAE  | - | 3.4761 |
| R²   | - | 0.9332 |


## Project structure

```
.
├── powerplant_data.csv     # Raw dataset
├── powerplant.ipynb        # Full notebook: data prep, training, evaluation
├── best_model.pt           # Saved weights of the best-performing model (generated on run)
└── README.md
```

## Requirements

```
torch
pandas
numpy
scikit-learn
matplotlib
```

Install with:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## Usage

1. Clone this repository and ensure `powerplant_data.csv` is in the project root.
2. Open and run `powerplant.ipynb` sequentially — each section is organized with markdown headers (Train-Test Split → Scaling → PyTorch Tensors → Model → Training → Evaluation).
3. The best model checkpoint is saved automatically to `best_model.pt` during training and reloaded for final evaluation.
