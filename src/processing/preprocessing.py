import pandas as pd
import networkx as nx
import json
from pathlib import Path


# Load the CSV data
script_dir = Path(__file__).parent.absolute()
csv_file_path = script_dir/'../../data/processed/ranges_to_dates_cleaned_air_quality.csv'
csv_data = pd.read_csv(csv_file_path)

# Convert 'Start_Date' to DateTime and sort the data
csv_data['Start_Date'] = pd.to_datetime(csv_data['Start_Date'])
csv_data['Year'] = csv_data['Start_Date'].dt.year
csv_data['Month'] = csv_data['Start_Date'].dt.month
csv_data['Day'] = csv_data['Start_Date'].dt.day

# Load the graph
graph_file_path = script_dir/'../../data/processed/zone_graph_from_topojson2.gexf'
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
updated_graph_file_path = script_dir/'../../data/processed/updated_zone_graph.gexf'
nx.write_gexf(G, updated_graph_file_path)
