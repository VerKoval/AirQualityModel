from dash import dash, dcc, callback, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.express as px

df = pd.read_csv('Data/Cleaned_Air_Quality.csv')
geojson = gpd.read_file('Data/UHF34.geo.json')
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

# Concat the value with it's unit
df['Data with Unit'] = df['Data Value'].astype(str) + ' ' + df['Measure Info']

# Dash layout
app.layout = dbc.Container([
    html.H1("Air Quality Mapping", className="text-center mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Choose a Time Period:", className="mb-2"),
            dcc.Dropdown(
                id='time-period-dropdown',
                options=[{'label': i, 'value': i} for i in df['Time Period'].unique()],
                value=df['Time Period'].unique()[0],
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
    Output('name-dropdown', 'options'),
    Input('time-period-dropdown', 'value')
)
def set_names_options(selected_time_period):
    # Filter the dataframe based on selected time period
    filtered_df = df[df['Time Period'] == selected_time_period]
    # Get the unique values in 'Name' for the filtered dataframe
    names = filtered_df['Name'].unique()
    # Create options for the 'Name' dropdown
    return [{'label': name, 'value': name} for name in names]

@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('time-period-dropdown', 'value'),
     Input('name-dropdown', 'value')]
)
def update_map(selected_time_period, selected_name):
    if selected_name is not None:
        filtered_df = df[(df['Time Period'] == selected_time_period) & (df['Name'] == selected_name)]
        # Create a new figure with the filtered data
        new_fig = px.choropleth(
            filtered_df,
            geojson=geojson,
            locations='UHF34 Zone',
            color='Data Value',
            featureidkey='properties.UHF',
            hover_data={'Data Value': False, 'Name': True, 'Data with Unit': True}
        )
        new_fig.update_geos(
            center={'lat': 40.6828, 'lon': -73.9060},
            projection_scale=300,
            visible=False
        )
        return new_fig
    else:
        # Return an empty figure if no name is selected
        return {}

if __name__ == '__main__':
    app.run_server(debug=True)