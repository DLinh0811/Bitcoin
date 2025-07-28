import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
import os

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

# Use RobustScaler instead of StandardScaler
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
# 3. Early Stopping Class
# ----------------------------

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ----------------------------
# 4. Training Function
# ----------------------------

def train_model(model, model_name, train_loader, val_loader, optimizer, criterion, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_rmse = float('inf')
    best_checkpoint_path = f'best_{model_name}_model.pth'
    
    early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch).squeeze()
                val_preds.append(pred)
                val_targets.append(y_batch)
        
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_rmse = torch.sqrt(criterion(val_preds, val_targets)).item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.6f}, Val RMSE: {val_rmse:.6f}')
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"💾 Saved Best {model_name} Model with Val RMSE: {val_rmse:.6f}")
        
        # Early stopping
        early_stopping(val_rmse)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return best_checkpoint_path, best_val_rmse

# ----------------------------
# 5. Evaluation Function
# ----------------------------

def evaluate_model(model, model_path, test_loader, scaler_target):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    preds = []
    targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch).squeeze()
            preds.append(pred.cpu())
            targets.append(y_batch.cpu())
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    
    # Inverse transform to original scale
    preds_original = scaler_target.inverse_transform(preds.reshape(-1, 1)).flatten()
    targets_original = scaler_target.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(targets_original, preds_original))
    r2 = r2_score(targets_original, preds_original)
    mae = mean_absolute_error(targets_original, preds_original)
    mape = np.mean(np.abs((targets_original - preds_original) / targets_original)) * 100
    
    return rmse, r2, mae, mape, preds_original, targets_original

# ----------------------------
# 6. Train All Models with Best Hyperparameters
# ----------------------------

# Best hyperparameters
best_params = {
    'RNN': {'optimizer': 'AdamW', 'batch_size': 32, 'lr': 0.0005, 'hidden_units': 64},
    'LSTM': {'optimizer': 'AdamW', 'batch_size': 64, 'lr': 0.0005, 'hidden_units': 64},
    'GRU': {'optimizer': 'AdamW', 'batch_size': 64, 'lr': 0.0005, 'hidden_units': 64}
}

input_size = X_train.shape[2]
num_epochs = 100
criterion = nn.MSELoss()

model_results = {}

for model_name, params in best_params.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name} Model")
    print(f"{'='*50}")
    
    # Create model
    if model_name == 'RNN':
        model = RNNModel(input_size, params['hidden_units'])
    elif model_name == 'LSTM':
        model = LSTMModel(input_size, params['hidden_units'])
    else:  # GRU
        model = GRUModel(input_size, params['hidden_units'])
    
    # Create optimizer
    if params['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
    else:  # RMSprop
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'])
    
    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=params['batch_size'])
    
    # Train model
    best_model_path, best_val_rmse = train_model(
        model, model_name, train_loader, val_loader, optimizer, criterion, num_epochs
    )
    
    # Store results
    model_results[model_name] = {
        'best_model_path': best_model_path,
        'best_val_rmse': best_val_rmse
    }

# ----------------------------
# 7. Evaluate Best Models on Validation Set
# ----------------------------

print(f"\n{'='*50}")
print("Validation Results")
print(f"{'='*50}")

val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

for model_name, results in model_results.items():
    # Load model
    if model_name == 'RNN':
        model = RNNModel(input_size, best_params[model_name]['hidden_units'])
    elif model_name == 'LSTM':
        model = LSTMModel(input_size, best_params[model_name]['hidden_units'])
    else:  # GRU
        model = GRUModel(input_size, best_params[model_name]['hidden_units'])
    
    # Evaluate on validation set
    val_preds = []
    val_targets = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(results['best_model_path'], map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch).squeeze()
            val_preds.append(pred.cpu())
            val_targets.append(y_batch.cpu())
    
    val_preds = torch.cat(val_preds).numpy()
    val_targets = torch.cat(val_targets).numpy()
    
    # Inverse transform
    val_preds_original = scaler_target.inverse_transform(val_preds.reshape(-1, 1)).flatten()
    val_targets_original = scaler_target.inverse_transform(val_targets.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    rmse = math.sqrt(mean_squared_error(val_targets_original, val_preds_original))
    r2 = r2_score(val_targets_original, val_preds_original)
    mae = mean_absolute_error(val_targets_original, val_preds_original)
    mape = np.mean(np.abs((val_targets_original - val_preds_original) / val_targets_original)) * 100
    
    print(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")

# ----------------------------
# 8. Test Best GRU Model
# ----------------------------

print(f"\n{'='*50}")
print("Testing Best GRU Model")
print(f"{'='*50}")

# Load best GRU model
best_gru_model = GRUModel(input_size, best_params['GRU']['hidden_units'])
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

rmse, r2, mae, mape, preds, targets = evaluate_model(
    best_gru_model, model_results['GRU']['best_model_path'], test_loader, scaler_target
)

print(f"Test Results:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# Save predictions for further analysis if needed
np.save('test_predictions.npy', preds)
np.save('test_targets.npy', targets)

print("\n✅ Training and evaluation completed!")