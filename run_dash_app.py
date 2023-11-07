from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

air_quality_data = pd.read_csv('Data/Cleaned_Air_Quality.csv')

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1("Air Quality Index Dashboard"),

    html.Label("Select an Indicator:"),
    dcc.Dropdown(
        id='indicator-dropdown',
        options=[{'label': i, 'value': i} for i in air_quality_data['Name'].unique()],
        value=air_quality_data['Name'].unique()[0]
    ),
    
    html.Label("Select a Time Period:"),
    dcc.Dropdown(
        id='time-period-dropdown',
    ),
    
    dcc.Graph(id='indicator-graph'),
    
    dash_table.DataTable(
        id='data-table',
        columns=[{"name": i, "id": i} for i in air_quality_data.columns],
        data=air_quality_data.to_dict('records'),
        page_size=10
    )
])

@app.callback(
    Output('time-period-dropdown', 'options'),
    [Input('indicator-dropdown', 'value')]
)
def set_time_period_options(selected_indicator):
    filtered_data = air_quality_data[air_quality_data['Name'] == selected_indicator]
    time_periods = filtered_data['Time Period'].unique()
    return [{'label': i, 'value': i} for i in time_periods]

@app.callback(
    Output('indicator-graph', 'figure'),
    [Input('indicator-dropdown', 'value'),
     Input('time-period-dropdown', 'value')]
)
def update_graph(selected_indicator, selected_time_period):
    filtered_data = air_quality_data[
        (air_quality_data['Name'] == selected_indicator) &
        (air_quality_data['Time Period'] == selected_time_period)
    ]

    fig = px.bar(
        filtered_data,
        x='Data Value',
        y='Geo Place Name',
        title=f'{selected_indicator} Levels by Location',
        orientation='h',
    )
    fig.update_layout(
        xaxis_title=filtered_data['Data Value'].name,
        yaxis_title=filtered_data['Geo Place Name'].name
    )
    fig.update_yaxes(categoryorder='total ascending')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
