import pandas as pd

#Make sure to load the dates as dates instead of strings when reading csv
AQdf = pd.read_csv('Data/Air_Quality.csv', parse_dates=['Start_Date'])

#Check for nulls
print(f'Number of nulls: \n{AQdf.isnull().sum()}\n')
#The only column with nulls is the "Message" column which is unnecessary

#Check what the most common Geo Type is
print(f"Number of values for column: {AQdf['Geo Type Name'].value_counts()}\n")

#UHF is the most common type of row we have for Geo Type, and the easiest to work with, so filter out all other Geo Types
AQdf = AQdf[AQdf['Geo Type Name'].isin(['UHF42', 'UHF34'])]
print(f'Dataframe size after dropping all non-UHF rows: {len(AQdf)}\n')

#We want to aggregate UHF42 to UHF34, which we can do like so:
uhf42_to_uhf34 = {
    305: 305307,
    307: 305307,
    306: 306308,
    308: 306308,
    309: 309310,
    310: 309310,
    404: 404406,
    406: 404406,
    501: 501502,
    502: 501502,
    503: 503504,
    504: 503504,
    105: 105106107,
    106: 105106107,
    108: 105106107
}
def map_uhf42_to_uhf34(row):
    if row['Geo Type Name'] == 'UHF42':
        #If the value is in the dictionary, replace with the UHF34 value. Otherwise, default to the existing value
        return uhf42_to_uhf34.get(row['Geo Join ID'], row['Geo Join ID'])
    #For UHF34 rows, no change is needed
    return row['Geo Join ID']  

# Apply the mapping to AQdf
AQdf['Geo Join ID'] = AQdf.apply(map_uhf42_to_uhf34, axis=1)
AQdf.rename(columns={'Geo Join ID': 'UHF34 Zone'}, inplace=True)

#Since measures vary, count the occurence of each measure
print(f"Number of values for column: {AQdf['Measure'].value_counts()}\n")
#Mean is most common

#Only keeps the data that has the mean for a certain measure
AQdf = AQdf[AQdf['Measure'] == 'Mean']

#Drop no longer necessary columns
AQdf.drop(['Message', 'Measure', 'Geo Place Name', 'Geo Type Name'], inplace=True, axis=1)

#Save this as a dataframe
AQdf.to_csv('Data/Cleaned_Air_Quality.csv', index=False)

#We want data for specific dates, and since the data is mean in ranges
#the following code generates more data for each range, that will copy
#the mean to all the dates in that range

#Time ranges are either seasonal or annual so two functions are to be implemented that
#generate dataframes with a range of dates starting from the data in "start_date" column
def get_date_ranges(start_date, time_period):    
    if 'Annual Average' in time_period or time_period.isdigit():
        # Handle 'Annual Average' and simple year entries
        end_date = start_date + pd.DateOffset(years=1)
    elif 'Winter' in time_period:
        end_date = start_date + pd.DateOffset(months=3)
    elif '2-Year Summer Average' in time_period:
        # Handle the unique case of a two-year summer average
        end_date = start_date + pd.DateOffset(years=2)
    elif 'Summer' in time_period:
        end_date = start_date + pd.DateOffset(months=3)
    else:
        raise ValueError(f"Unhandled time period format: {time_period}")

    return pd.date_range(start=start_date, end=end_date - pd.DateOffset(days=1))

addedData = []
#Produces a list containing the new rows
for _, row in AQdf.iterrows():
    date_ranges = get_date_ranges(row['Start_Date'], row['Time Period'])
    for date in date_ranges:
        new_row = row.copy()
        new_row['Start_Date'] = date
        addedData.append(new_row)

#Convert final list into dataframe
newAQ = pd.DataFrame(addedData, columns=AQdf.columns)

#Drop 'Time Period' as it is no longer necessary
newAQ.drop(['Time Period'], inplace=True, axis=1)

print(f'New dataframe size: {len(newAQ)}')

#Save the disaggregated dataframe
newAQ.to_csv('Data/Ranges_To_Dates_Cleaned_Air_Quality.csv', index=False)
