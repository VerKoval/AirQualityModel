import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# Load the CSV data
script_dir = Path(__file__).parent.absolute()
csv_file_path = script_dir/'../../data/processed/ranges_to_dates_cleaned_air_quality.csv'
csv_data = pd.read_csv(csv_file_path)

# Convert 'Start_Date' to a datetime object and filter relevant fields
csv_data['Start_Date'] = pd.to_datetime(csv_data['Start_Date'])
filtered_data = csv_data[['Start_Date', 'Name', 'UHF34 Zone', 'Data Value']]

# Pivot 'Name' to create separate columns for each particle type
pivoted_data = filtered_data.pivot_table(
    index=['Start_Date', 'UHF34 Zone'], 
    columns='Name', 
    values='Data Value'
)

# Fill missing values with the prior day's value
filled_data = pivoted_data.fillna(method='ffill')

# Reset index to make 'Start_Date' and 'UHF34 Zone' columns again
final_data = filled_data.reset_index()

# Save the final reorganized data to a new CSV file
output_csv_path = script_dir/'../../data/processed/reorganized_air_quality.csv'
final_data.to_csv(output_csv_path, index=False)

# Load and preprocess the air quality data
df = pd.read_csv(script_dir/'../../data/processed/reorganized_air_quality.csv')
df['Start_Date'] = pd.to_datetime(df['Start_Date'])

# Normalize data values
scaler = StandardScaler()
df[['Nitrogen dioxide (NO2)', 'Ozone (O3)', 'Fine particles (PM 2.5)']] = scaler.fit_transform(
    df[['Nitrogen dioxide (NO2)', 'Ozone (O3)', 'Fine particles (PM 2.5)']]
)
#Save the final reorganized data to a new CSV file
# TODO: For some reason the normalized data isn't working
output_csv_path = script_dir/'../../data/processed/normalized_air_quality_TEST.csv' 
final_data.to_csv(output_csv_path, index=False)
