import json
import networkx as nx
import itertools
import geopandas as gpd
import matplotlib.pyplot as plt


def load_topojson(topojson_path):
    with open(topojson_path, 'r') as file:
        return json.load(file)

def flatten_arcs(arcs):
    flattened_arcs = []
    for item in arcs:
        if isinstance(item, list):  # Nested list
            for subitem in item:
                if isinstance(subitem, list):
                    flattened_arcs.extend(subitem)
                else:
                    flattened_arcs.append(subitem)
        else:  # Single integer
            flattened_arcs.append(item)
    return flattened_arcs

def create_graph_from_topojson(topojson_data):
    geometries = topojson_data['objects']['collection']['geometries']
    G = nx.Graph()

    # Add nodes
    for geometry in geometries:
        uhf_code = geometry['properties']['UHF']
        G.add_node(uhf_code, properties=geometry['properties'])

    # Add edges based on shared arcs
    for geo_i, geo_j in itertools.combinations(geometries, 2):
        arcs_i = flatten_arcs(geo_i['arcs'])
        arcs_j = flatten_arcs(geo_j['arcs'])
        shared_arcs = set([abs(arc) for arc in arcs_i]) & set([abs(arc) for arc in arcs_j])
        if shared_arcs:
            G.add_edge(geo_i['properties']['UHF'], geo_j['properties']['UHF'])

    return G

def save_graph(graph, file_path):
    nx.write_gexf(graph, file_path)

def main():
    topojson_path = '/Users/pana/Downloads/UHF34.topo.json'
    #output_graph_path = '/Users/pana/projects/github/AirQualityModel/AirQualityModel/Data/zone_graph_from_topojson2.gexf'

    topojson_data = load_topojson(topojson_path)
    G = create_graph_from_topojson(topojson_data)
    save_graph(G, output_graph_path)
    
    print(f"Graph saved to {output_graph_path}")

if __name__ == "__main__":
    main()
    G = create_graph_from_topojson(load_topojson('/Users/pana/Downloads/UHF34.topo.json'))
    gdf = gpd.read_file('/Users/pana/Downloads/UHF34.topo.json')
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.plot(ax=ax, color='white', edgecolor='black')

    # Overlaying the graph on the map
    # For simplicity, using centroids of polygons as node positions
    pos = {zone: (centroid.x, centroid.y) for zone, centroid in zip(gdf['GEOCODE'], gdf.centroid)}
    nx.draw(G, pos, ax=ax, node_size=50, node_color='blue', edge_color='red')

    # Show the plot
    plt.show()
