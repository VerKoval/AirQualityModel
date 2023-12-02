import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from pathlib import Path


num_nodes = 35  # Update this based on your actual data
num_features = 3  # NO2, O3, PM2.5
batch_size = 32  # This can be set based on how you want to batch your data

# Load the merged air quality and geographic data
script_dir = Path(__file__).parent.absolute()
merged_csv_path = script_dir/'../../data/processed/merged_air_quality_geographic.csv'
merged_data = pd.read_csv(merged_csv_path)
merged_data['Start_Date'] = pd.to_datetime(merged_data['Start_Date'])

# Extract year, month, and day as separate features
merged_data['Year'] = merged_data['Start_Date'].dt.year
merged_data['Month'] = merged_data['Start_Date'].dt.month
merged_data['Day'] = merged_data['Start_Date'].dt.day

# Load the graph
graph_path = script_dir/'../../data/processed/zone_graph_from_topojson2.gexf'
G = nx.read_gexf(graph_path)

# Convert GEOCODE node labels to integer indices
node_mapping = {node: i for i, node in enumerate(G.nodes())}
G = nx.relabel_nodes(G, node_mapping)

edge_index = torch.tensor(list(G.edges())).t().contiguous()

# Assign 'data_value' to each node in G based on average 'Data Value'
avg_data_values = merged_data.groupby('UHF34 Zone')['Data Value'].mean().to_dict()
for node in G.nodes():
    original_geocode = list(node_mapping.keys())[list(node_mapping.values()).index(node)]
    G.nodes[node]['data_value'] = avg_data_values.get(int(original_geocode), 0)

# Function to prepare data for a given segment, including time features
def prepare_data_for_segment(segment_data, G, node_mapping, edge_index):    # Group data by 'UHF34 Zone' and calculate mean 'Data Value'
    segment_avg_values = segment_data.groupby('UHF34 Zone')['Data Value'].mean().to_dict()
    segment_avg_year = segment_data['Year'].mean()
    segment_avg_month = segment_data['Month'].mean()
    segment_avg_day = segment_data['Day'].mean()

    # Update node features in the graph based on the segment data
    features = []
    for node in G.nodes():
        original_geocode = list(node_mapping.keys())[list(node_mapping.values()).index(node)]
        feature_value = segment_avg_values.get(int(original_geocode), 0)
        features.append([feature_value, segment_avg_year, segment_avg_month, segment_avg_day])
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    # Convert to PyTorch tensors
    x = torch.tensor(normalized_features, dtype=torch.float)
    y = torch.tensor(normalized_features[:, 0], dtype=torch.float).unsqueeze(1)  # Assuming target is the first column

    return Data(edge_index=edge_index, x=x, y=y)

# STGNN Model Definition
class STGNN(torch.nn.Module):
    def __init__(self, node_features, out_features):
        super(STGNN, self).__init__()
        # Adjust the input size based on the number of features (1 original + 3 time features)
        self.conv1 = GCNConv(4, 32)  # Adjusted for 4 input features
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# Define the number of time segments for validation
num_segments = 5
segment_length = len(merged_data) // num_segments

# Initialize metrics collection
metrics = []

# Training and validation loop
for i in range(num_segments - 1):
    # Define train and test segments
    train_data_segment = merged_data.iloc[i * segment_length : (i + 1) * segment_length]
    test_data_segment = merged_data.iloc[(i + 1) * segment_length : (i + 2) * segment_length]

    # Prepare data for STGNN model for each segment
    train_graph_data = prepare_data_for_segment(train_data_segment, G, node_mapping, edge_index)
    test_graph_data = prepare_data_for_segment(test_data_segment, G, node_mapping, edge_index)
    # Initialize model and optimizer
    model = STGNN(node_features=4, out_features=1)  # Adjusted for 4 input features
    optimizer = Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    # Training function
    def train(data):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        return loss.item()

    # Training loop
    for epoch in range(400):
        loss = train(train_graph_data)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

    # Validate the model on the test segment
    model.eval()
    with torch.no_grad():
        test_out = model(test_graph_data)
        test_loss = criterion(test_out, test_graph_data.y.view(-1, 1))
        metrics.append(test_loss.item())

# Calculate and print average performance
average_performance = np.mean(metrics)
print("Average Performance:", average_performance)

# Save the trained model
torch.save(model.state_dict(), script_dir/'../../models/stgnn_model.pth')

# Prediction function incorporating time features
def predict_air_quality(model, zone, time_features, G, node_mapping):
    if zone in node_mapping:
        node_index = node_mapping[zone]
        node_feature = G.nodes[node_index]['data_value']
        combined_features = torch.tensor([[node_feature] + time_features], dtype=torch.float)

        # Create a dummy edge_index for a single node (self-loop)
        dummy_edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(Data(x=combined_features, edge_index=dummy_edge_index)).item()
        return prediction
    else:
        return "Zone not found in graph"

# Example usage of prediction function
zone_to_predict = '203'  # Replace with actual zone ID
time_features_to_predict = [2023, 10, 22]  # Replace with actual year, month, day
prediction = predict_air_quality(model, zone_to_predict, time_features_to_predict, G, node_mapping)
print("Predicted Air Quality for Zone:", zone_to_predict, "is", prediction)