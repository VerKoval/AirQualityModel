from dash import dash, dcc, callback, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.express as px
import json

df = pd.read_csv('Data/Cleaned_Air_Quality.csv')
uhf34 = gpd.read_file('GeoJSON/UHF34.geo.json')
uhf42 = gpd.read_file('GeoJSON/UHF42.geo.json')
cd_geo = gpd.read_file('GeoJSON/CD.geo.json')
# with open('Data/UHF34_zone_names.json', 'r') as file:
#     uhf_zone_names = json.load(file)
    
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Concat the value with its unit
df['Value'] = df['Data Value'].astype(str) + ' ' + df['Measure Info']

# indicator_id_names = {
#     365: 'Fine particles (PM 2.5)',
#     375: 'Nitrogen dioxide (NO2)',
#     386: 'Ozone (O3)'
# }

# Dash layout
app.layout = dbc.Container([
    html.H1("NYC Air Quality Mapping", className="text-center mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Choose a GeoJSON Map:", className="mb-2"),
            dcc.Dropdown(
                id='geojson-dropdown',
                options=[
                    {'label': 'UHF34', 'value': 'UHF34'},
                    {'label': 'UHF42', 'value': 'UHF42'},
                    {'label': 'Community District', 'value': 'CD'}
                ],
                value='uhf34'
            )
        ], width=12),
        dbc.Col([
            html.Label("Choose a Time Period:", className="mb-2"),
            dcc.Dropdown(
                id='time-period-dropdown',
                placeholder="Select a time period"
            )
        ], width=12),
        dbc.Col([
            html.Label("Choose a Parameter:", className="mb-2"),
            dcc.Dropdown(
                id='name-dropdown',
                placeholder="Select a parameter"
            )
        ], width=12),
        dbc.Col(dcc.Graph(id='choropleth-map', style={'height': '80vh'}), width=12)
    ])
], fluid=True)

@app.callback(
    Output('time-period-dropdown', 'options'),
    Input('geojson-dropdown', 'value')
)
def set_time_period_options(selected_geojson):
    # Filter the dataframe based on selected geography type
    filtered_df = df[df['Geo Type Name'] == selected_geojson]
    # Get the unique values in 'Time Period' for the filtered dataframe
    time_periods = filtered_df['Time Period'].unique()
    return [{'label': tp, 'value': tp} for tp in time_periods]

@app.callback(
    Output('name-dropdown', 'options'),
    [Input('geojson-dropdown', 'value'), 
     Input('time-period-dropdown', 'value')]
)
def set_names_options(selected_geojson, selected_time_period):
    # Filter the dataframe based on both selected geography type and time period
    filtered_df = df[(df['Geo Type Name'] == selected_geojson) & (df['Time Period'] == selected_time_period)]
    # Get the unique values in 'Name' for the filtered dataframe
    names = filtered_df['Name'].unique()
    return [{'label': name, 'value': name} for name in names]

@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('geojson-dropdown', 'value'),
     Input('time-period-dropdown', 'value'),
     Input('name-dropdown', 'value')]
)
def update_map(selected_geojson, selected_time_period, selected_name):
    if selected_geojson is not None and selected_time_period is not None and selected_name is not None:
        filtered_df = df[
            (df['Geo Type Name'] == selected_geojson) &
            (df['Time Period'] == selected_time_period) &
            (df['Name'] == selected_name)
        ]

        # Load the correct GeoJSON data
        geojson = {
            'UHF34': uhf34,
            'UHF42': uhf42,
            'CD': cd_geo
        }[selected_geojson]

        # Create the choropleth map
        fig = px.choropleth(
            filtered_df,
            geojson=geojson,
            locations='Geo Join ID',
            color='Data Value',
            featureidkey='properties.GEOCODE',
            hover_data={'Data Value': False, 'Geo Place Name': True, 'Name': True, 'Value': True}
        )
        fig.update_geos(
            center={'lat': 40.6828, 'lon': -73.9060},
            projection_scale=300,
            visible=False
        )
        return fig
    else:
        return {}  # Return an empty figure or a default figure if selections are not complete

if __name__ == '__main__':
    app.run_server(debug=True)