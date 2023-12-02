import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Transformer
from torch_geometric.nn import ChebConv
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# Load air quality data and convert dates
script_dir = Path(__file__).parent.absolute()
air_quality_df = pd.read_csv(script_dir/'../../data/processed/normalized_air_quality.csv')
air_quality_df['Start_Date'] = pd.to_datetime(air_quality_df['Start_Date'])

# Normalizing the feature columns
scaler = StandardScaler()
feature_columns = ['Nitrogen dioxide (NO2)', 'Ozone (O3)', 'Fine particles (PM 2.5)']
target_column = ['Fine particles (PM 2.5)']
#air_quality_df[feature_columns] = scaler.fit_transform(air_quality_df[feature_columns])
features_data = air_quality_df[feature_columns].values


# Function to prepare data for model input
def prepare_data(data):
    features = torch.tensor(data[feature_columns].values, dtype=torch.float)
    targets = torch.tensor(data['Fine particles (PM 2.5)'].values, dtype=torch.float).view(-1, 1)
    return Data(x=features, edge_index=edge_index_tensor, y=targets)

# Splitting the dataset based on date for time series data
train_end = air_quality_df['Start_Date'].min() + pd.DateOffset(months=16)
valid_end = train_end + pd.DateOffset(months=2)

train_data = air_quality_df[air_quality_df['Start_Date'] < train_end]
valid_data = air_quality_df[(air_quality_df['Start_Date'] >= train_end) & (air_quality_df['Start_Date'] < valid_end)]
test_data = air_quality_df[air_quality_df['Start_Date'] >= valid_end]

# Convert features to PyTorch tensor
features_tensor = torch.tensor(features_data, dtype=torch.float)

# Load adjacency matrix, skipping the first row (header) and the first column (index)
adjacency_df = pd.read_csv(script_dir/'../../data/processed/adjacency_matrix_from_gexf.csv', index_col=0)

# Ensure that all values are numeric. If the matrix is correct, all values should be 0 or 1 (integers)
adjacency_df = adjacency_df.apply(pd.to_numeric)

# Now convert to edge_index list
edge_index = []
for src, row in enumerate(adjacency_df.values):
    for dest, val in enumerate(row):
        if val == 1:
            edge_index.append([src, dest])

print(f"First few edges: {edge_index[:5]}")  # Debugging: Print the first few edges

# Convert to PyTorch tensor
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

print(f"Edge index tensor shape: {edge_index_tensor.shape}")  # Debugging: Print the shape of the edge index tensor

# Create PyTorch Geometric data object
graph_data = Data(x=features_tensor, edge_index=edge_index_tensor)

# Assuming you have 3 features per node and a certain number of nodes (zones)
num_features = 3  # NO2, O3, PM2.5
num_nodes = len(air_quality_df['UHF34 Zone'].unique())

#MLP Layer
class MLP(nn.Module):
    # Output size should match GCN input size
    def __init__(self, num_input_features, num_output_features):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_input_features, 100),
            nn.ReLU(),
            nn.Linear(100, num_output_features)  # Adjust the output size to match GCN input
        )

    def forward(self, x):
        return self.layers(x)

#GCN Layer
class GCN(nn.Module):
    def __init__(self, num_input_features, gcn_output_features, transformer_feature_size):
        super(GCN, self).__init__()
        self.conv1 = ChebConv(num_input_features, 100, K=2)
        self.conv2 = ChebConv(100, gcn_output_features, K=2)
        self.linear_transform = nn.Linear(gcn_output_features, transformer_feature_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.linear_transform(x)  # Transform to the transformer feature size
        return x
    

# Transformer Layer as a Simplified Alternative to Informer
class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super(TransformerLayer, self).__init__()
        self.transformer = Transformer(
            d_model=d_model,  # Use the passed d_model
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=1,
            dim_feedforward=2048,
            dropout=0.05
        )

    def forward(self, src, tgt):
        return self.transformer(src, tgt)

# GCNInformer Model Integration
class GCNInformer(nn.Module):
    def __init__(self, num_features, num_nodes, transformer_feature_size):
        super(GCNInformer, self).__init__()
        self.mlp = MLP(num_input_features=num_features, num_output_features=num_features)
        self.gcn = GCN(num_input_features=num_features, gcn_output_features=16, transformer_feature_size=transformer_feature_size)
        self.transformer = TransformerLayer(d_model=transformer_feature_size)

    def forward(self, x, edge_index):
        x_mlp = self.mlp(x)
        x_gcn = self.gcn(x_mlp, edge_index)
        x_transformer = self.transformer(x_gcn, x_gcn)  # Using the same tensor for src and tgt as an example
        return x_transformer


# Define the transformer feature size (d_model) which is expected by the Transformer layer
transformer_feature_size = 512

# Define the model, loss function, optimizer
model = GCNInformer(num_features=3, num_nodes=num_nodes, transformer_feature_size=512)
loss_function = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.0001)

# Prepare datasets
train_dataset = prepare_data(train_data)
valid_dataset = prepare_data(valid_data)
test_dataset = prepare_data(test_data)

# Training loop with early stopping
best_loss = float('inf')
patience = 5
counter = 0

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    output = model(train_dataset.x, train_dataset.edge_index)
    loss = loss_function(output, train_dataset.y)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(valid_dataset.x, valid_dataset.edge_index)
        val_loss = loss_function(val_output, valid_dataset.y)
    
    print(f'Epoch {epoch+1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    # Early stopping criteria
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter == patience:
            print('Early stopping!')
            break

# Save the trained model
torch.save(model.state_dict(), script_dir/'../../models/experimental_model.pth')

# Test the model on the test set
model.eval()
with torch.no_grad():
    test_output = model(test_dataset.x, test_dataset.edge_index)
    test_loss = loss_function(test_output, test_dataset.y)
print(f'Test Loss: {test_loss.item()}')