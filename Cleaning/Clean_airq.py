#Libraries
import pandas as pd
import numpy as np

#Download the Air Quality csv as a AQdf
AQdf = pd.read_csv('/Users/parisababaali/Documents/AirQualityModel/Data/Air_Quality.csv')

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

#preliminary drop
to_be_dropped = ['Indicator ID', 'Message']

AQdf.drop(to_be_dropped, inplace = True, axis = 1)