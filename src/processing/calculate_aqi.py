import pandas as pd
from pathlib import Path


script_dir = Path(__file__).parent.absolute()
AQdf = pd.read_csv(script_dir/'../../data/processed/reorganized_air_quality.csv', parse_dates=['Start_Date'])

df = AQdf.copy()
# Make sure to replace null values with 0s
df['Fine particles (PM 2.5)'].fillna(0, inplace=True)
df['Nitrogen dioxide (NO2)'].fillna(0, inplace=True)
df['Ozone (O3)'].fillna(0, inplace=True)

pm25_breakpoints = {
    (0.0, 12.0): (0, 50),
    (12.1, 35.4): (51, 100),
    (35.5, 55.4): (101, 150),
    (55.5, 150.4): (151, 200),
    (150.5, 250.4): (201, 300),
    (250.5, 350.4): (301, 400),
    (350.5, 500.4): (401, 500)
}
no2_breakpoints = {
    (0, 53): (0, 50),
    (54, 100): (51, 100),
    (101, 360): (101, 150),
    (361, 649): (151, 200),
    (650, 1249): (201, 300),
    (1250, 1649): (301, 400),
    (1650, 2049): (401, 500)
}
ozone_breakpoints = {
    (0, 54): (0, 50),
    (55, 70): (51, 100),
    (71, 85): (101, 150),
    (86, 105): (151, 200),
    (106, 200): (201, 300)
    # Ozone breakpoints usually go up to 0.2 ppm
}

# AQI calculation function
def calculate_aqi_from_concentration(concentration, breakpoints):
    for (bp_lo, bp_hi), (aqi_lo, aqi_hi) in breakpoints.items():
        if bp_lo <= concentration <= bp_hi:
            return ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + aqi_lo
    # Return maximum AQI value otherwise
    return 500

# Truncate the concentration values according to the rules
df['Fine particles (PM 2.5)'] = df['Fine particles (PM 2.5)'].apply(lambda x: round(x, 1))
df['Nitrogen dioxide (NO2)'] = df['Nitrogen dioxide (NO2)'].apply(lambda x: round(x))
df['Ozone (O3)'] = df['Ozone (O3)'].apply(lambda x: round(x))

# Calculate the AQI for each pollutant
df['PM2.5 AQI'] = df['Fine particles (PM 2.5)'].apply(lambda x: calculate_aqi_from_concentration(x, pm25_breakpoints))
df['NO2 AQI'] = df['Nitrogen dioxide (NO2)'].apply(lambda x: calculate_aqi_from_concentration(x, no2_breakpoints))
df['Ozone AQI'] = df['Ozone (O3)'].apply(lambda x: calculate_aqi_from_concentration(x, ozone_breakpoints))

# The overall AQI for the date would be the maximum AQI value across all pollutants
AQdf['AQI'] = df[['PM2.5 AQI', 'NO2 AQI', 'Ozone AQI']].mean(axis=1).round().astype(int)


# Save the disaggregated dataframe
AQdf.to_csv(script_dir/'../../data/processed/reorganized_air_quality_with_aqi.csv', index=False)