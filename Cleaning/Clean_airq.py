#Libraries
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

#Download the Air Quality csv as a AQdf
AQdf = pd.read_csv('Data/Air_Quality.csv')

#Checks for nulls
print(AQdf.isnull().sum())
#The only column with nulls is the "Message" column which is unnecessary

#Check for dublicates
print(len(AQdf))
AQdf.drop_duplicates()
print(len(AQdf))
#No duplicates

#Since measures vary, count the occurence of each measure
print(AQdf['Measure'].value_counts())
#Mean is most common

#Only keeps the data that has the mean for a certain measure
AQdf = AQdf[AQdf['Measure'] == 'Mean']

print(AQdf['Start_Date'].dtype)

#We want data for specific dates, and since the data is mean in ranges
#the following code generates more data for each range, that will copy
#the mean to all the dates in that range

#time ranges are either seasonal or annual so two functions are to be implemented
#that generate dataframes with a range of dates starting from the data in 
#"start_date" column
def seasonalGenerator(start_date):
    end_date = start_date + relativedelta(months=3)
    return pd.date_range(start=start_date, end=end_date)

def annualGenerator(start_date):
    end_date = start_date + relativedelta(months=12)
    return pd.date_range(start=start_date, end=end_date)

#Dataframe only of values that fit contain either seasonal or annual range
tempDF = AQdf[AQdf['Time Period'].str.contains(r'\b(?:summer|spring|fall|winter|annual)\b', case=False, na=False)]

addedData = []

#produces a list containing the new rows
for _, row in tempDF.iterrows():
    start_date = pd.Timestamp(row['Start_Date'])
    #determines if annual or seasonal and use respective function
    if 'annual' in row['Measure'].lower():
        date_ranges = annualGenerator(start_date)
    else:
        date_ranges = seasonalGenerator(start_date)
    #copies given row and changes date to the new date value
    for date in date_ranges:
        new_row = row.copy()
        new_row['Start_Date'] = date
        addedData.append(new_row)

#Converts final list into dataframe
added_df = pd.DataFrame(addedData, columns=['Start_Date', 'Time Period'])

#joins final additional data and previous data
AQdf = pd.concat([AQdf, added_df], ignore_index=True)
#AQdf.sort_values(by='Start_Date', inplace=True)
AQdf.drop_duplicates(inplace=True)
AQdf.reset_index(drop=True, inplace=True)

#drop no longer necessary columns
to_be_dropped = ['Indicator ID', 'Message', 'Measure','Time Period']

AQdf.drop(to_be_dropped, inplace = True, axis = 1)

print(len(AQdf))

AQdf.to_csv('Data/Cleaned_Air_Quality.csv', index=False)
