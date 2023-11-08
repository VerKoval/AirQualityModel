# NYC Air Quality Data Analysis and Predictions
## Description
This project aims to analyze and predict air quality in New York City to aid residents in gauging the safety of their environment. Leveraging data from NYC Open Data and other external datasets, our goal is to model, track, and visualize the air quality with a focus on the impact of various particles and location-based trends.

## Data Sources
- [**NYC Air Quality Dataset**](https://data.cityofnewyork.us/Environment/Air-Quality/c3uy-2p5r): Contains air quality surveillance results from different neighborhoods in NYC.

## Authors
- Veronica Koval
- Mohammad Kahaf Bhuiyan
- Panagiotis (Peter) Kokolis

# To-Do
## Data Cleaning
- Normalize the locations to be consistent with zip codes only (resolve multiple zip codes for areas)
- Standardize measurement units for particle concentrations (e.g., parts per million)
- Filter out data columns that are not relevant to air quality
### Research
- Identify the particles that have a significant impact on air quality
- Determine acceptable levels for these particle concentrations

## Dash App
- Implement a geojson-based map for spatial data representation
- Improve the aesthetics of the app
- Improve app interactivity
  - Enable users to select specific dates/times for predictions.
  - Introduce filters for different particle types and areas.

## Predictions
- Spatiotemporal prediction model
