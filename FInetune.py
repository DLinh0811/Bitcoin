import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import itertools
from tqdm import tqdm

# ----------------------------
# 1. Load and Prepare Data
# ----------------------------

df = pd.read_csv('BTCUSDT_1h_engineered.csv')

# Select all 15 features (5 basic + 10 engineered)
features = ['open', 'high', 'low', 'close', 'volume',
            'MA20', 'MA10', 'DIFF-MA20-CLOSE', 'DIFF-MA10-CLOSE',
            'MA14_low', 'MA14_high', 'Bollinger_Upper', 'Bollinger_Lower',
            'RSI', 'MACD']

target = 'close'

# Convert to NumPy arrays
data = df[features].values
target_data = df[target].values

# Normalize the data
scaler_data = RobustScaler()
scaler_target = RobustScaler()

data_scaled = scaler_data.fit_transform(data)
target_scaled = scaler_target.fit_transform(target_data.reshape(-1, 1)).flatten()

# Create sequences
def create_sequences(data, target, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(target[i+window_size])
    return np.array(X), np.array(y)

window_size = 20
X_seq, y_seq = create_sequences(data_scaled, target_scaled, window_size)

# Split into train/val/test (60/20/20)
split1 = int(0.6 * len(X_seq))
split2 = int(0.8 * len(X_seq))

X_train, X_val, X_test = X_seq[:split1], X_seq[split1:split2], X_seq[split2:]
y_train, y_val, y_test = y_seq[:split1], y_seq[split1:split2], y_seq[split2:]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ----------------------------
# 2. Define Models
# ----------------------------

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# ----------------------------
# 3. Hyperparameter Grid Search
# ----------------------------

optimizers = {
    'Adam': torch.optim.Adam,
    'AdamW': torch.optim.AdamW,
    'RMSprop': torch.optim.RMSprop
}

batch_sizes = [32, 64, 128]
learning_rates = [0.001, 0.0005, 0.0001]
hidden_units = [32, 64, 128]

hyperparameter_grid = list(itertools.product(optimizers.keys(), batch_sizes, learning_rates, hidden_units))

input_size = X_train.shape[2]
num_epochs = 20

results = []

def evaluate_model(model_class, opt_name, batch_size, lr, hidden_size):
    model = model_class(input_size, hidden_size)
    optimizer = optimizers[opt_name](model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

    model.train()
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch, _ in val_loader:
            pred = model(x_batch).squeeze()
            preds.append(pred)
    preds = torch.cat(preds).numpy()
    mse = mean_squared_error(y_val.numpy(), preds)
    return mse

# ----------------------------
# 4. Run Grid Search
# ----------------------------

models = {
    'RNN': RNNModel,
    'LSTM': LSTMModel,
    'GRU': GRUModel
}

best_configs = {}

for model_name, model_class in models.items():
    print(f"\n🔍 Tuning {model_name}...")
    best_mse = float('inf')
    best_config = None

    for opt, bs, lr, hu in tqdm(hyperparameter_grid, desc=f"Tuning {model_name}"):
        try:
            mse = evaluate_model(model_class, opt, bs, lr, hu)
            results.append((model_name, opt, bs, lr, hu, mse))
            if mse < best_mse:
                best_mse = mse
                best_config = (opt, bs, lr, hu)
        except Exception as e:
            print(f"[Error] {model_name} failed with config: {opt}, {bs}, {lr}, {hu}")
            continue

    best_configs[model_name] = (best_config, best_mse)
    print(f"✅ Best config for {model_name}: {best_config}, MSE: {best_mse:.6f}")

# ----------------------------
# 5. Print Best Configurations
# ----------------------------

print("\n🏆 Final Best Hyperparameters:")
for model, (config, mse) in best_configs.items():
    print(f"{model}: Optimizer={config[0]}, Batch Size={config[1]}, LR={config[2]}, Hidden Units={config[3]} | MSE={mse:.6f}")