from dash import dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import geopandas as gpd
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
from pathlib import Path


# Load files
script_dir = Path(__file__).parent.absolute()
df = pd.read_csv(script_dir/'data/final/predicted_aqi.csv')
uhf34 = gpd.read_file(script_dir/'data/raw/UHF34.geo.json')

# Start Dash module with the MINTY theme, add templates for figures
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
load_figure_template(["minty", "minty_dark"])

#App format
app.layout = dbc.Container([
    # Type of air selector
    html.H1("NYC Air Quality Mapping", className="text-center mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Air Type", className="mb-2"),
            dcc.Dropdown(
                id='name-dropdown',
                options=[
                    {'label': 'NO2', 'value': 'Nitrogen dioxide (NO2)'},
                    {'label': 'Fine Particles', 'value': 'Fine particles (PM 2.5)'},
                    {'label': 'Ozone', 'value': 'Ozone (O3)'},
                    {'label': 'AQI', 'value': 'AQI'},
                    {'label': 'Predicted AQI', 'value': 'Predicted AQI'}
                    ]
            )
        ], width=6),
        # Date Selector
        dbc.Col([
            html.Label("Select Date", className="mb-2"),
            dcc.DatePickerSingle(
                id='given_date',
                date='2009-06-08',  #Default
            )
        ], width=6), 
    ]),
    # "Enter" button to activate map
    dbc.Row([
        dbc.Col(html.Button('Enter', id='submit-button', n_clicks=0, className="btn-lg"), width=12)
    ], className="mb-2"),
    # Map
    dbc.Row([
        dbc.Col(dcc.Graph(id='choropleth-map', style={'display': 'none'}), width=12)
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
    # Checks if button activated
    if n_clicks > 0 and selected_time_period and selected_name:
        filtered_df = df[df['Start_Date'] == selected_time_period]

        # Create the choropleth map
        fig = px.choropleth(
            filtered_df,
            geojson=uhf34,
            locations='UHF34 Zone',
            color=selected_name,
            featureidkey='properties.GEOCODE',
            hover_data={'UHF34 Zone': True},
            color_continuous_scale='mint',
            template='minty_dark'
        )
        # Center and resize the map to be in NYC
        fig.update_geos(
            center={'lat': 40.6828, 'lon': -73.9060},
            projection_scale=390,
            visible=False
        )
        fig.update_layout(
            height=725,  # Use the entire vertical height of the screen
            width=1400,
            margin=dict(l=0, r=0, t=20, b=0)  # Adjust margins to fill whole figure
        )
        return fig, {'display': 'block', 'height': '100vh'}
    else:
        return {}, {'display': 'none'}
    
if __name__ == '__main__':
    app.run_server(debug=True)