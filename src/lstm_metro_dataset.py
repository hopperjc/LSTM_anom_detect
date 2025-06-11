import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 1. Fetch dataset (Metro Interstate Traffic Volume, id=492)
dataset = fetch_ucirepo(id=492)
X_df = dataset.data.features.copy()
y_series = dataset.data.targets.astype(int)

# 2. Combine features and target
X_df['traffic_volume'] = y_series

# 3. Parse datetime and sort
X_df['date_time'] = pd.to_datetime(X_df['date_time'])
X_df.sort_values('date_time', inplace=True)
X_df.reset_index(drop=True, inplace=True)

# 4. EDA
print("Dataset shape:", X_df.shape)
print(X_df.head())
print(X_df.describe())

plt.figure(figsize=(12,4))
plt.plot(X_df['date_time'], X_df['traffic_volume'])
plt.title('Traffic Volume ao Longo do Tempo')
plt.xlabel('Date Time')
plt.ylabel('Volume')
plt.show()

# 5. Univariate anomaly detection via Z-score
traffic = X_df['traffic_volume'].values
z_scores = np.abs(stats.zscore(traffic))
thresh = 2
idx_z = np.where(z_scores > thresh)[0]
print(f"Z-score anomalias (>={thresh}): {len(idx_z)}")

# 6. Multivariate preprocessing
def preprocess(df):
    features = ['holiday','temp','rain_1h','snow_1h','clouds_all','weather_main']
    transformer = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['holiday','weather_main']),
        ('num', MinMaxScaler(), ['temp','rain_1h','snow_1h','clouds_all'])
    ])
    X = transformer.fit_transform(df[features])
    y = df['traffic_volume'].values
    return X, y, transformer

X_all, y_all, preproc = preprocess(X_df)

# 7. Create sequences and train/val split
def create_sequences(X, y, window):
    seq_X, seq_y = [], []
    for i in range(len(X) - window):
        seq_X.append(X[i:i+window])
        seq_y.append(y[i+window])
    return np.array(seq_X), np.array(seq_y)

window = 24
n_features = X_all.shape[1]
X_seq, y_seq = create_sequences(X_all, y_all, window)
split = int(len(X_seq) * 0.7)
X_train, y_train = X_seq[:split], y_seq[:split]
X_val, y_val = X_seq[split:], y_seq[split:]

# Convert to PyTorch tensors and loaders
torch.manual_seed(42)
X_train_t = torch.from_numpy(X_train).float()
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
X_val_t = torch.from_numpy(X_val).float()
y_val_t = torch.from_numpy(y_val).float().unsqueeze(1)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# 8. Define LSTM model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMAnomalyDetect(nn.Module):
    def __init__(self, n_features, n_hidden=64, n_layers=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers, dropout=drop,
                            batch_first=True)
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # take last time step
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMAnomalyDetect(n_features=n_features).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 9. Train with early stopping
best_loss, patience, trials = np.inf, 5, 0
for epoch in range(1, 51):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_losses.append(criterion(model(xb), yb).item())
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch} -> train: {np.mean(train_losses):.4f}, val: {val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        trials = 0
        torch.save(model.state_dict(), 'best_lstm.pt')
    else:
        trials += 1
        if trials >= patience:
            print("Interrompendo por early stopping")
            break

# Load best model
model.load_state_dict(torch.load('best_lstm.pt'))
model.eval()

# 10. Predict and detect anomalies
with torch.no_grad():
    y_pred = model(X_val_t.to(device)).cpu().numpy().flatten()
errors = np.abs(y_val - y_pred)
mean_e, std_e = errors.mean(), errors.std()
th_e = mean_e + 2 * std_e
idx_anom = np.where(errors > th_e)[0]
print(f"Erros: mean={mean_e:.2f}, std={std_e:.2f}, threshold={th_e:.2f}")
print(f"Anomalias detectadas: {len(idx_anom)}")

# 11. Plot results
dates = X_df['date_time'].iloc[window+split:].reset_index(drop=True)

plt.figure(figsize=(12,4))
plt.plot(dates, y_val, label='True')
plt.plot(dates, y_pred, label='Pred')
plt.scatter(dates[idx_anom], y_val[idx_anom], color='red', s=10, label='Anomalias')
plt.title('Anomalias via PyTorch LSTM')
plt.xlabel('Date Time')
plt.ylabel('Volume')
plt.legend()
plt.show()