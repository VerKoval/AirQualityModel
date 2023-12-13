import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from pathlib import Path


# Load air quality data and convert dates
script_dir = Path(__file__).parent.absolute()
data = pd.read_csv(script_dir/'../../data/processed/reorganized_air_quality_with_aqi.csv', parse_dates=['Start_Date'])


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out

# Convert date to datetime and sort
data.sort_values(by=['Start_Date', 'UHF34 Zone'], inplace=True)

# Check for NaN values and handle them
data.dropna(inplace=True)

# Select relevant columns for training (excluding date and UHF zone)
columns_to_use = ['Fine particles (PM 2.5)', 'Nitrogen dioxide (NO2)', 'Ozone (O3)', 'AQI']
data_for_model = data[columns_to_use]

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_for_model)

# Function to create dataset with sliding window
def create_dataset(dataset, look_back=7):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, -1])
    return np.array(X), np.array(Y)

look_back = 7
X, Y = create_dataset(scaled_data, look_back)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.astype(np.float32))
Y_train_tensor = torch.tensor(Y_train.astype(np.float32))
X_test_tensor = torch.tensor(X_test.astype(np.float32))
Y_test_tensor = torch.tensor(Y_test.astype(np.float32))

# Create data loaders
train_data = TensorDataset(X_train_tensor, Y_train_tensor)
test_data = TensorDataset(X_test_tensor, Y_test_tensor)

batch_size = 32
train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

input_dim = X_train.shape[2]
hidden_dim = 50
num_layers = 2
output_dim = 1

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        Y_pred = model(X_batch).squeeze()
        loss = criterion(Y_pred, Y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1} Loss: {loss.item()}')

model.eval()
total_loss = 0
with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        Y_pred = model(X_batch).squeeze()
        total_loss += criterion(Y_pred, Y_batch).item()

test_loss = total_loss / len(test_loader)
print(f'Test Loss: {test_loss}')

torch.save(model, script_dir/'../../models/lstm_model.pth')
torch.save(model.state_dict(), script_dir/'../../models/lstm_state_dict.pth')
joblib.dump(scaler, script_dir/'../../models/lstm_scaler.save')

model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
model = torch.load(script_dir/'../../models/lstm_model.pth')
model.eval()


def predict_aqi_for_date_and_zones(model, data, date, scaler, look_back=7):
    predictions = {}
    date = pd.to_datetime(date)
    
    # Iterate over each unique zone
    for zone in data['UHF34 Zone'].unique():
        # Filter data for the current zone and get the last 'look_back' days data
        zone_data = data[data['UHF34 Zone'] == zone]
        zone_data_before_date = zone_data[zone_data['Start_Date'] < date]

        if zone_data_before_date.shape[0] >= look_back:
            input_data = zone_data_before_date.iloc[-look_back:][columns_to_use].values

            # Normalize the data
            input_data_normalized = scaler.fit_transform(input_data)

            # Convert to PyTorch tensor
            input_tensor = torch.tensor(input_data_normalized[np.newaxis, :, :].astype(np.float32))

            # Predict
            with torch.no_grad():
                prediction = model(input_tensor).squeeze().numpy()

            # Inverse transform to get actual AQI value
            dummy_array = np.zeros((1, len(columns_to_use)))
            dummy_array[0, -1] = prediction  # Assuming AQI is the last feature
            prediction_actual = scaler.inverse_transform(dummy_array)[0, -1]
            predictions[zone] = prediction_actual
        else:
            predictions[zone] = None  # Not enough data to predict

    return predictions


# Assuming you have 'data' DataFrame loaded and 'scaler' used during training
# date_to_predict = '2018-12-13'
# predicted_aqis = predict_aqi_for_date_and_zones(model, data, date_to_predict, scaler)

# for zone, aqi in predicted_aqis.items():
#     if aqi is not None:
#         print(f"Predicted AQI for Zone {zone} on {date_to_predict}: {int(aqi.round())}")
#     else:
#         print(f"Not enough data to predict for Zone {zone} on {date_to_predict}")


data['Predicted AQI'] = None

# Define the date range
start_date = pd.to_datetime('2008-12-08')
end_date = pd.to_datetime('2022-01-01')
date_range = pd.date_range(start_date, end_date)

# Iterate over the date range and update predictions in the DataFrame
for current_date in date_range:
    predictions = predict_aqi_for_date_and_zones(model, data, current_date, scaler)
    for zone, aqi in predictions.items():
        if aqi is not None:
            # Update the DataFrame - Set 'Predicted AQI' where the date and zone match
            data.loc[(data['Start_Date'] == current_date) & (data['UHF34 Zone'] == zone), 'Predicted AQI'] = aqi

# Save the updated DataFrame to a new CSV
data.to_csv(script_dir/'../../data/final/predicted_aqi.csv', index=False)