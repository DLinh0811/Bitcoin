Here's a refined version of your `README.md` file with improved grammar, formatting consistency, and clarity—ready to copy and paste:


# 📈 Bitcoin Price Prediction Using Deep Learning RNN Models

This project implements and compares three Recurrent Neural Network (RNN) architectures—Vanilla RNN, LSTM, and GRU—for predicting Bitcoin prices using technical indicators and engineered features.

## 🚀 Project Overview

This repository contains a complete pipeline for Bitcoin price prediction using deep learning. It covers feature engineering, hyperparameter tuning, model training with early stopping, and comprehensive evaluation of RNN-based architectures.

## 📁 Project Structure
```
bitcoin-price-prediction/
├── EDA.ipynb                  # Exploratory Data Analysis
├── BTCUSDT\_1h.csv             # Original raw data
├── BTCUSDT\_1h\_engineered.csv  # Feature-engineered data
├── feature\_engineer.py        # Feature engineering script
├── finetune.py                # Hyperparameter tuning script
├── train.py                   # Model training and evaluation
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🛠️ System Requirements

- **GPU**: NVIDIA RTX 4070 or equivalent
- **RAM**: 16GB or more (recommended)
- **Python**: Version 3.8 or higher

## 📦 Installation

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n bitcoin-rnn python=3.9

# Activate the environment
conda activate bitcoin-rnn

# Install PyTorch with CUDA support (adjust the CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
````

### Using pip

```bash
# Create a virtual environment
python -m venv bitcoin-rnn-env

# Activate the environment
# On Windows:
bitcoin-rnn-env\Scripts\activate
# On macOS/Linux:
source bitcoin-rnn-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚦 Usage

1. **Feature Engineering**
   Generate the feature-engineered dataset:

   ```bash
   python feature_engineer.py
   ```

2. **Hyperparameter Tuning** *(Optional)*
   Run the grid search:

   ```bash
   python finetune.py
   ```

3. **Model Training and Evaluation**
   Train and evaluate the selected model:

   ```bash
   python train.py
   ```

4. **Exploratory Data Analysis**
   Open `EDA.ipynb` using Jupyter Notebook to explore the dataset.

## 🧪 Best Hyperparameters

After extensive tuning, the following hyperparameters yielded the best results:

| Model       | Optimizer | Batch Size | Learning Rate | Hidden Units |
| ----------- | --------- | ---------- | ------------- | ------------ |
| Vanilla RNN | AdamW     | 32         | 0.0005        | 64           |
| LSTM        | AdamW     | 64         | 0.0005        | 64           |
| GRU         | AdamW     | 64         | 0.0005        | 64           |

## 📊 Model Performance

### Validation Set

| Model       | RMSE (\$)  | MAE (\$)   | MAPE (%) | R² Score  |
| ----------- | ---------- | ---------- | -------- | --------- |
| Vanilla RNN | 987.35     | 714.22     | 3.89     | 0.881     |
| LSTM        | 924.71     | 658.30     | 3.21     | 0.899     |
| GRU         | **905.12** | **644.15** | **3.08** | **0.913** |

### Final Test Set (Best GRU Model)

| Metric    | Value  |
| --------- | ------ |
| RMSE (\$) | 893.14 |
| MAE (\$)  | 629.55 |
| MAPE (%)  | 3.01   |
| R² Score  | 0.921  |

## 🔍 Key Features

* **📈 Technical Indicators**: Includes Moving Averages, Bollinger Bands, RSI, and MACD
* **🧠 Deep Learning Models**: RNN, LSTM, and GRU implementations
* **⚙️ Hyperparameter Tuning**: Grid search over batch size, learning rate, hidden units, and optimizer
* **🛡️ Early Stopping**: Prevents overfitting by monitoring validation loss
* **📊 Multi-Metric Evaluation**: Uses RMSE, MAE, MAPE, and R²
* **⚡ GPU Acceleration**: Optimized for modern GPUs (e.g., NVIDIA RTX 4070)

## 🧠 Model Architecture

The RNN-based models are configured as follows:

* **Input Window**: 20-day historical window to predict the 21st day's closing price
* **Input Features**: 15 features (5 raw + 10 engineered)
* **Architecture**: Single hidden layer with configurable units
* **Optimizer**: AdamW
* **Early Stopping**: Patience of 10 epochs

## ✅ Results Summary

The GRU model outperformed RNN and LSTM with:

* **92.1% R²** score on the test set
* **\$893.14 RMSE**
* **3.01% MAPE**

---

<div align="center">
  <strong>📊 Built with Python, PyTorch, and ❤️ for Financial Deep Learning 📊</strong>
</div>
```

