import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from torch.nn import Transformer
import pandas as pd
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import numpy as np

# Load air quality data and convert dates
air_quality_df = pd.read_csv('/Users/pana/projects/github/AirQualityModel/AirQualityModel/Data/Reorganized_air_quality.csv')
air_quality_df['Start_Date'] = pd.to_datetime(air_quality_df['Start_Date'])

#DEBUG PRINT
#print(air_quality_df.head())


# Removing rows where 'UHF34 Zone' is NaN
air_quality_df.isnull().sum()
air_quality_df = air_quality_df.dropna()
air_quality_df.isnull().sum()

def remap_nodes(adjacency_df):
    # Create a mapping from the original identifiers to new sequential integers
    unique_identifiers = list(set(map(str, adjacency_df.columns)) | set(map(str, adjacency_df.index)))
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_identifiers)}

    # Apply this mapping to the rows and columns of the adjacency matrix
    remapped_adjacency_df = adjacency_df.rename(index=node_mapping, columns=node_mapping)

    return remapped_adjacency_df, node_mapping

# Load adjacency matrix, skipping the first row (header) and the first column (index)
adjacency_df = pd.read_csv('/Users/pana/projects/github/AirQualityModel/AirQualityModel/Data/adjacency_matrix_from_gexf.csv', index_col=0)

# Ensure that all values are numeric. If the matrix is correct, all values should be 0 or 1 (integers)
adjacency_df = adjacency_df.apply(pd.to_numeric)

# Apply the remap_nodes function to the adjacency matrix
remapped_adjacency_df, node_mapping = remap_nodes(adjacency_df)

# Convert 'UHF34 Zone' in the air quality dataframe from string float to integer
# First convert to float, then to int, bypassing NaN values
air_quality_df['UHF34 Zone'] = air_quality_df['UHF34 Zone'].apply(
    lambda x: int(float(x)) if pd.notnull(x) and x != 'NaN' else x
)

# Creating a mapping from 'UHF34 Zone' to remapped indices using node_mapping
zone_to_node_mapping = {zone: node_mapping.get(zone, None)
                        for zone in air_quality_df['UHF34 Zone'].unique() if pd.notnull(zone)}

# Applying the mapping to the 'UHF34 Zone' in air quality data
air_quality_df['Mapped_Node'] = air_quality_df['UHF34 Zone'].map(zone_to_node_mapping)


#Defining features and target variable
features = air_quality_df[['Nitrogen dioxide (NO2)', 'Ozone (O3)', 'Fine particles (PM 2.5)']]
target = air_quality_df['Fine particles (PM 2.5)'] #can replace


scaler = StandardScaler()
feature_columns = ['Nitrogen dioxide (NO2)', 'Ozone (O3)', 'Fine particles (PM 2.5)']
air_quality_df[feature_columns] = scaler.fit_transform(air_quality_df[feature_columns])

#DEBUG PRINT
#print(adjacency_df)

#convert to edge_index list
edge_index = []
for src, row in enumerate(adjacency_df.values):
    for dest, val in enumerate(row):
        if val == 1:
            edge_index.append([src, dest])



# Convert to tensor
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()


#features and nodes count
num_features = 3  # NO2, O3, PM2.5
num_nodes = len(air_quality_df['UHF34 Zone'].unique())

#DEBUG check
# print("Edge Index Tensor:", edge_index_tensor)
# print("Max index in edge_index tensor:", edge_index_tensor.max())
# print("Min index in edge_index tensor:", edge_index_tensor.min())

# MLP Layer
class MLP(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_input_features, 3),
            nn.ReLU(),
            nn.Linear(3, num_output_features)  # Adjust the output size to match GCN input
        )

    def forward(self, x):
        return self.layers(x)

#DEBUG wrapper class
class ChebConvDebug(ChebConv):
    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # DEBUG statements to log the shapes of the input tensors
        #print("ChebConvDebug Forward - Input x shape:", x.shape)
        #print("ChebConvDebug Forward - Input edge_index shape:", edge_index.shape)

        # Call the original ChebConv forward method
        return super().forward(x, edge_index, edge_weight, batch)

# GCN Layer
class GCN(nn.Module):
    def __init__(self, num_input_features, gcn_output_features, transformer_feature_size):
        super(GCN, self).__init__()
        self.conv1 = ChebConvDebug(num_input_features, 3, K=2)
        self.conv2 = ChebConvDebug(3, gcn_output_features, K=2)
        self.linear_transform = nn.Linear(gcn_output_features, transformer_feature_size)

    def forward(self, x, edge_index, batch=None):
        print("GCN Forward - Input x shape:", x.shape)
        print("GCN Forward - Input edge_index shape:", edge_index.shape)

        x = F.relu(self.conv1(x, edge_index))
        print("GCN Forward - Post conv1 x shape:", x.shape)

        x = F.relu(self.conv2(x, edge_index))
        print("GCN Forward - Post conv2 x shape:", x.shape)

        x = self.linear_transform(x)
        print("GCN Forward - Post linear_transform x shape:", x.shape)

        return x

# Transformer Layer 
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

# GCNWeather model
class GCNInformer(nn.Module):
    def __init__(self, num_features, num_nodes, transformer_feature_size):
        super(GCNInformer, self).__init__()
        self.mlp = MLP(num_input_features=num_features, num_output_features=num_features)
        self.gcn = GCN(num_input_features=num_features, gcn_output_features=16, transformer_feature_size=transformer_feature_size)
        self.transformer = TransformerLayer(d_model=transformer_feature_size)
        self.final_layer = nn.Linear(transformer_feature_size, 1) 

    def forward(self, x, edge_index, batch):
        x_mlp = self.mlp(x)
        x_gcn = self.gcn(x_mlp, edge_index)  # The GCN layer does not use the batch vector
        x_transformer = self.transformer(x_gcn, x_gcn)  # Using the same tensor for src and tgt as an example
        x_final = self.final_layer(x_transformer)  # Pass through the final layer
        return x_final

# Define the model, loss function, optimizer
model = GCNInformer(num_features=3, num_nodes=num_nodes, transformer_feature_size=512)
loss_function = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.0001)

def prepare_data(df, edge_index_tensor):
    data_list = []

    # Group data by 'Start_Date' (each date is a separate graph)
    grouped_data = df.groupby('Start_Date')

    for date, group in grouped_data:
        # Node features: ['Nitrogen dioxide (NO2)', 'Ozone (O3)', 'Fine particles (PM 2.5)']
        node_features = group[feature_columns].to_numpy(dtype=float)
        x_tensor = torch.tensor(node_features, dtype=torch.float)

        # Target values: 'Fine particles (PM 2.5)'
        target_values = group['Fine particles (PM 2.5)'].to_numpy(dtype=float)
        y_tensor = torch.tensor(target_values, dtype=torch.float).view(-1, 1)

        # Create a Data object for the graph
        data = Data(x=x_tensor, edge_index=edge_index_tensor.clone(), y=y_tensor)
        data_list.append(data)

    return data_list


# Prepare datasets
train_end = air_quality_df['Start_Date'].min() + pd.DateOffset(months=8)
valid_end = train_end + pd.DateOffset(months=1)

train_data = air_quality_df[air_quality_df['Start_Date'] < train_end]
valid_data = air_quality_df[(air_quality_df['Start_Date'] >= train_end) & (air_quality_df['Start_Date'] < valid_end)]
test_data = air_quality_df[air_quality_df['Start_Date'] >= valid_end]

train_dataset = prepare_data(train_data, edge_index_tensor)
valid_dataset = prepare_data(valid_data, edge_index_tensor)
test_dataset = prepare_data(test_data, edge_index_tensor)

# DEBUG check if train_dataset is correctly initialized
# if train_dataset is None:
#     print("train_dataset is None")
# else:
#     print("Number of graphs in train_dataset:", len(train_dataset))

# # Check the first few elements of train_dataset
# for i, item in enumerate(train_dataset):
#     print(f"Item {i} type:", type(item))
#     if isinstance(item, tuple):
#         print("Tuple contents:", item)
#     elif isinstance(item, Data):
#         print("Nodes (Features):", item.x)
#         print("Edges (Edge Index):", item.edge_index)
#     else:
#         print("Unknown type")
#     if i == 4:
#         break


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


#DEBUG Sanity Check
# for batch in train_loader:
#     #DEBUG
#     # print("Batch edge index max:", batch.edge_index.max().item())
#     # print("Number of nodes in batch:", batch.num_nodes)
#     if batch.edge_index.max().item() >= batch.num_nodes:
#         print("Warning: Edge index references a non-existent node in this batch.")
#     break  # inspect 1 batch
# for i, data in enumerate(train_dataset):
#     print(f"Graph {i}:")
#     print("Nodes (Features):", data.x)
#     print("Edges (Edge Index):", data.edge_index)
#     print("Target:", data.y)
#     if i == 4:  # Limit to printing the first 5 graphs
#         break

# for batch in train_loader:
#     print("Batch Features Shape:", batch.x.shape)
#     print("Batch Edge Index Shape:", batch.edge_index.shape)
#     print("Batch Edge Index Max:", batch.edge_index.max())
#     print("Batch Edge Index Min:", batch.edge_index.min())
#     print("Batch Size:", batch.batch.size(0))
#     break  # Remove this line to inspect all batches




# Training loop with early stopping
best_loss = float('inf')
patience = 5
counter = 0

for epoch in range(30):
    print(f"Starting Epoch {epoch+1}")
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        #DEBUG
        # print("Shape of batch.x:", batch.x.shape)
        # print("Shape of batch.edge_index:", batch.edge_index.shape)
        # print("Shape of batch.batch:", batch.batch.shape)
        # print("Shape of batch.y:", batch.y.shape)

        max_edge_index = torch.max(batch.edge_index).item()
        #DEBUG
        # print("Maximum index in edge_index tensor:", max_edge_index)
        # print("Number of nodes (size of batch.x):", batch.x.size(0))

        # This will help in identifying if there's an edge referencing a non-existent node
        if max_edge_index >= batch.x.size(0):
            print("Error: Edge index references a non-existent node.")

        output = model(batch.x, batch.edge_index, batch.batch) #pass batchvector
        loss = loss_function(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in valid_loader:
            # Pass batch.batch to the model
            val_output = model(batch.x, batch.edge_index, batch.batch)  

            # Reshape output to match the target shape
            val_output = val_output.view_as(batch.y)

            val_loss = loss_function(val_output, batch.y)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(valid_loader)

    print(f'Epoch {epoch+1}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

    # Early stopping criteria
    if val_loss < best_loss:
        best_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter == patience:
            print('Early stopping!')
            break

