import pandas as pd
import networkx as nx
import torch
import json


# Load the CSV data
csv_file_path = '/Users/pana/projects/github/AirQualityModel/AirQualityModel/Data/Ranges_To_Dates_Cleaned_Air_Quality.csv'  # Replace with your file path
csv_data = pd.read_csv(csv_file_path)

# Convert 'Start_Date' to DateTime and sort the data
# Load the CSV data
csv_data = pd.read_csv(csv_file_path)
csv_data['Start_Date'] = pd.to_datetime(csv_data['Start_Date'])
csv_data['Year'] = csv_data['Start_Date'].dt.year
csv_data['Month'] = csv_data['Start_Date'].dt.month
csv_data['Day'] = csv_data['Start_Date'].dt.day

# Load the graph
graph_file_path = '/Users/pana/projects/github/AirQualityModel/AirQualityModel/Data/zone_graph_from_topojson2.gexf'
G = nx.read_gexf(graph_file_path)

# Initialize time series data for each node in the graph
for node in G.nodes():
    G.nodes[node]['time_series_data'] = []

# Merge the aggregated data into the graph
for index, row in csv_data.iterrows():
    uhf_zone = str(row['UHF34 Zone'])
    if uhf_zone in G:
        time_series_entry = {
            'year': row['Year'],
            'month': row['Month'],
            'day': row['Day'],
            'particle_type': row['Name'],
            'data_value': row['Data Value']
        }
        G.nodes[uhf_zone]['time_series_data'].append(time_series_entry)

# Serialize 'time_series_data' for each node
for node in G.nodes():
    if 'time_series_data' in G.nodes[node]:
        time_series_data = G.nodes[node]['time_series_data']
        # Convert the list of dictionaries to a JSON string
        G.nodes[node]['time_series_data'] = json.dumps(time_series_data)


# Save the updated graph to a new file
updated_graph_file_path = '/Users/pana/projects/github/AirQualityModel/AirQualityModel/Data/updated_zone_graph.gexf'
nx.write_gexf(G, updated_graph_file_path)
