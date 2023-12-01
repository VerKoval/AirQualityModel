import pandas as pd
from pathlib import Path


# Make sure to load the dates as dates instead of strings when reading .csv
script_dir = Path(__file__).parent.absolute()
AQdf = pd.read_csv(script_dir/'Data/Air_Quality.csv', parse_dates=['Start_Date'])

# Check what the most common Geo Type is
print(f"Number of values for column: {AQdf['Geo Type Name'].value_counts()}\n")

# Drop Borough and Citywide
print(f'Dataframe size before: {len(AQdf)}\n')
AQdf = AQdf[AQdf['Geo Type Name'].isin(['CD', 'UHF34', 'UHF42'])]
print(f'Dataframe size after: {len(AQdf)}\n')

# Since measures vary, count the occurence of each measure
print(f"Number of values for column: {AQdf['Measure'].value_counts()}\n")

# Only keeps the data that has the mean for a certain measure
print(f'Dataframe size before: {len(AQdf)}\n')
AQdf = AQdf[AQdf['Measure'] == 'Mean']
print(f'Dataframe size after: {len(AQdf)}\n')

# Drop no longer necessary columns
AQdf.drop(['Message', 'Measure'], inplace=True, axis=1)

# Save this as a dataframe
AQdf.to_csv(script_dir/'Data/TTCleaned_Air_Quality.csv', index=False)