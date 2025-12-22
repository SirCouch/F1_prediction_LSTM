# F1 Race Outcome Predictor

A machine learning pipeline using **LSTM (Long Short-Term Memory)** neural networks to predict Formula 1 race outcomes. This project leverages the **FastF1** API for high-resolution telemetry and timing data, incorporating tire performance and management metrics for more accurate forecasting.

## Features

- **Sequential Learning**: Uses LSTM to capture driver and team performance trends over time.
- **Tire Performance Integration**: Incorporates tire compound usage and tire life (management) from previous races.
- **Professional CLI**: Unified interface for data fetching, model training, and race prediction.
- **Future-Ready**: Support for predicting future races (2025 and beyond) as schedule data becomes available.
- **Robust Preprocessing**: Automatic handling of missing data (DNFs/DNQs) and feature normalization.

## Installation

1. **Clone the repository** (or navigate to the project directory).
2. **Install dependencies**:
   ```bash
   pip install fastf1 torch pandas scikit-learn matplotlib tqdm
   ```
3. **Cache Setup**: The app creates an `f1_cache` directory to speed up subsequent data fetches.

## Usage

The project is managed via the `f1_cli.py` entry point.

### 1. Update & Train
Fetch historical data and train the model. You can specify which years to include.
```bash
python f1_cli.py update --years 2021 2022 2023 2024
```

### 2. Predict Race Results
Run a prediction for a specific race. The model uses the "Grid Position" (from qualifying) and the driver's recent history to predict their finishing rank.
```bash
python f1_cli.py predict --year 2024 --round 1
```

### 3. Check Status
View the health of your local model and the size of your dataset.
```bash
python f1_cli.py status
```

## Technical Architecture

### Data Pipeline (`data_loader.py`)
- **Extraction**: Fetches race results, starting grid, and lap-by-lap tire data via FastF1.
- **Engineering**: 
  - Tracks majority tire compound and max tire life per race.
  - Generates sequences of the **5 most recent races** for each driver.
- **Normalization**: Scales inputs (Grid, TeamID, EventID, Position, Compound, TyreLife) to a `[0, 1]` range for neural network stability.

### The Model (`model.py`)
- **Type**: 2-Layer LSTM.
- **Structure**:
  - Input: `(batch_size, sequence_length, features)`
  - Hidden Layers: 64 units with Dropout (0.2).
  - Head: Fully connected layer with ReLU activation.
- **Loss Function**: Mean Squared Error (MSE), optimized using Adam.

### Prediction Logic (`predict.py`)
For a given race, the model:
1. Loads the current starting grid.
2. Retrieves the historical "form" of each driver from the local dataset.
3. Simulates the outcome by predicting a normalized finishing position for every driver.
4. Ranks drivers based on these predictions to produce a final table.

## Results Example
A prediction on the 2024 Bahrain GP correctly identified **Verstappen** as the winner, followed by **Perez** and **Leclerc**, closely mirroring the actual race finish.

---

*Note: F1 is inherently unpredictable due to mechanical failures, weather, and crashes. This model provides a statistical baseline based on historical performance.*
