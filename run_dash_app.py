from dash import dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.express as px
from pathlib import Path


# Load files
script_dir = Path(__file__).parent.absolute()
df = pd.read_csv(script_dir/'data/processed/reorganized_air_quality.csv')
uhf34 = gpd.read_file(script_dir/'data/raw/UHF34.geo.json')

# Start Dash module with the VAPOR theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])

#App format
app.layout = dbc.Container([
    #Type of air selector
    html.H1("NYC Air Quality Mapping", className="text-center mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Air Type", className="mb-2"),
            dcc.Dropdown(
                id='name-dropdown',
                options=[
                    {'label': 'NO2', 'value': 'Nitrogen dioxide (NO2)'},
                    {'label': 'Fine Particles', 'value': 'Fine particles (PM 2.5)'},
                    {'label': 'Ozone', 'value': 'Ozone (O3)'}]
            )
        ], width=6),
        #Date Selector
        dbc.Col([
            html.Label("Select Date", className="mb-2"),
            dcc.DatePickerSingle(
                id='given_date',
                date='2008-12-01'  #Default
            )
        ], width=6), 
    ]),
    #"Enter" button to activate map
    dbc.Row([
        dbc.Col(html.Button('Enter', id='submit-button', n_clicks=0), width=12)
    ]),
    #map
    dbc.Row([
        dbc.Col(dcc.Graph(id='choropleth-map', style={'display': 'none'}), width=18)
    ])
], fluid=True, style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center'})

@app.callback(
    [Output('choropleth-map', 'figure'),
     Output('choropleth-map', 'style')],
    [Input('submit-button', 'n_clicks')],
    [Input('given_date', 'date'),
     Input('name-dropdown', 'value')]
)
def update_map(n_clicks, selected_time_period, selected_name):
    #checks if button activated
    if n_clicks is not None and n_clicks > 0 and selected_time_period and selected_name:
        filtered_df = df[df['Start_Date'] == selected_time_period]

        column_to_use = None #the column with selected type of air
        if selected_name == 'Fine particles (PM 2.5)':
            column_to_use = 'Fine particles (PM 2.5)'
        elif selected_name == 'Nitrogen dioxide (NO2)':
            column_to_use = 'Nitrogen dioxide (NO2)'
        elif selected_name == 'Ozone (O3)':
            column_to_use = 'Ozone (O3)'

        # Create the choropleth map
        fig = px.choropleth(
            filtered_df,
            geojson=uhf34,
            locations='UHF34 Zone',
            color=column_to_use,
            featureidkey='properties.GEOCODE',
            hover_data={'UHF34 Zone': True},
            color_continuous_scale='mint'
        )
        # Center and resize the map to be in NYC
        fig.update_geos(
            center={'lat': 40.6828, 'lon': -73.9060},
            projection_scale=375,
            visible=False
        )
        fig.update_layout(
            height=600,  # Use the entire vertical height of the screen
            width = 1200,
            margin=dict(l=0, r=0, t=0, b=0),  # Adjust margins to fill the whole screen
        )

        return fig, {'display': 'block', 'height': '84vh'}
    else:
        return {}, {'display': 'none'}
    
if __name__ == '__main__':
    app.run_server(debug=True)