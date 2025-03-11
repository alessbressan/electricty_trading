import openmeteo_requests
import requests_cache
from retry_requests import retry

import requests
import zipfile
import io
import pandas as pd
import numpy as np

class DataScrapping:
    def __init__(self, start_date:int, n_years:int):
        self.url = 'http://mis.nyiso.com/public/'
        self.start_year = start_date
        self.n_years = n_years
        self.features = pd.DataFrame()
    
    def get_features(self):
        self.update_features(self.rt_prices(), name= 'RT Prices')
        # self.update_features(self.da_prices(), name= 'DA Prices')
        self.update_features(self.weather_forecast_api(), name= 'Weather')
        # self.update_features(self.load_forecast(), name= 'Load Forecast')
        # self.update_features(self.capacity(), name= 'Capacity')
    
    def update_features(self, new_feature:pd.Series, name:str):
        new_feature = new_feature.rename(name)
        
        # apparently pd.concat adds duplicate indices 
        # self.features = self.features.loc[~self.features.index.duplicated(keep='first')]

        if self.features.empty:
            self.features = new_feature.to_frame()  # Convert Series to DataFrame
            self.features.index = pd.to_datetime(self.features.index)
        else:
            if name in self.features.columns:
                raise ValueError(f"Column '{name}' already exists in features. Choose a different name.")
            
            self.features.index = pd.to_datetime(self.features.index)
            new_feature.index = pd.to_datetime(new_feature.index)
            
            self.features = pd.merge(self.features, new_feature.to_frame(), how='left', left_index= True, right_index= True)
            # self.features = pd.concat([self.features, new_feature.to_frame()], axis=1)
    
    def compute_HDD(max_temp, min_temp, base_temp= 65):
        """
        Parameters:
            max_temp (float): The maximum temperature for the day in Fahrenheit.
            min_temp (float): The minimum temperature for the day in Fahrenheit.
            base_temp (float): The base temperature in Fahrenheit (default is 65°C).
            
        Returns:
            float: The Heating Degree Days for the day.
        """
        avg_temp = (max_temp + min_temp) / 2
        hdd = max(0, base_temp - avg_temp)
        return hdd

    def compute_CDD(max_temp, min_temp, base_temp= 65):
        """
        Parameters:
            max_temp (float): The maximum temperature for the day in Fahrenheit.
            min_temp (float): The minimum temperature for the day in Fahrenheit.
            base_temp (float): The base temperature in Fahrenheit (default is 65°C).
            
        Returns:
            float: The Heating Degree Days for the day.
        """
        avg_temp = (max_temp + min_temp) / 2
        hdd = max(0, avg_temp - base_temp)
        return hdd
    
    def increment_date_by_month(self, date_str, increment):
        # Step 1: Convert the string to a datetime object
        date = pd.to_datetime(date_str, format='%Y%m%d')
        
        # Step 2: Add one month
        next_month_date = date + pd.DateOffset(months=increment)
        
        # Step 3: Convert the datetime object back to a string in the desired format
        return next_month_date.strftime('%Y%m%d')
    
    def rt_prices(self):
        df_year = []

        for i in range(12*self.n_years):
            url_date = self.increment_date_by_month(self.start_year, i)
            zip_url = f'csv/realtime/{url_date}realtime_zone_csv.zip'

            #GET request
            response = requests.get(self.url+zip_url)

            #check if it went through
            if response.status_code == 200:
                #extract zip file
                with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    #initialize an empty list

                    dfs =[]

                    #loop through CSV
                    for file_name in file_list:
                        if file_name.endswith('.csv'):
                            with zip_ref.open(file_name) as csv_file:
                                df = pd.read_csv(csv_file)
                            
                            dfs.append(df)
                    
                    #merge dataframes
                    combined_df = pd.concat(dfs, ignore_index=True)
                    df_year.append(combined_df)
            else:
                print('Failed to download ZIP files')

        real_time = pd.concat(df_year, ignore_index=True)
        real_time = real_time[real_time['Name'] == 'LONGIL']
        real_time.drop(columns=['Name', 'PTID'], inplace= True)
        real_time.drop_duplicates('Time Stamp', inplace= True)
        real_time['Time Stamp'] = pd.to_datetime(real_time['Time Stamp'])
        real_time.set_index('Time Stamp', inplace= True)
        real_time = real_time.resample('60min').mean()

        return real_time['LBMP ($/MWHr)']
    
    def da_prices(self):
        df_year = []

        for i in range(12*self.n_years):
            url_date = self.increment_date_by_month(self.start_year, i)
            zip_url = f'csv/damlbmp/{url_date}damlbmp_zone_csv.zip'

            #GET request
            response = requests.get(self.url+zip_url)

            #check if it went through
            if response.status_code == 200:
                #extract zip file
                with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    #initialize an empty list

                    dfs =[]

                    #loop through CSV
                    for file_name in file_list:
                        if file_name.endswith('.csv'):
                            with zip_ref.open(file_name) as csv_file:
                                df = pd.read_csv(csv_file)
                            
                            dfs.append(df)
                    
                    #merge dataframes
                    combined_df = pd.concat(dfs, ignore_index=True)
                    df_year.append(combined_df)
            else:
                print('Failed to download ZIP files')

        day_ahead = pd.concat(df_year, ignore_index=True)
        day_ahead = day_ahead[day_ahead['Name'] == 'LONGIL']
        day_ahead.drop(columns=['Name', 'PTID'], inplace= True)
        day_ahead.drop_duplicates('Time Stamp', inplace= True)
        day_ahead['Time Stamp'] = pd.to_datetime(day_ahead['Time Stamp'])
        day_ahead.set_index('Time Stamp', inplace= True)

        return day_ahead['LBMP ($/MWHr)']
    
    def weather_forecast_api(self):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        start_date = pd.to_datetime(self.start_year, format="%Y%m%d")
        end_date = start_date + pd.DateOffset(years= self.n_years)
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 40.8168,
            "longitude": -73.0662,
            "start_date": f"{start_date.strftime("%Y-%m-%d")}",
            "end_date": f"{end_date.strftime("%Y-%m-%d")}",
            "hourly": ["temperature_2m", "weather_code", "relative_humidity_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"]
        }
        responses = openmeteo.weather_api(url, params=params)

        # Process first location. Add a for-loop for multiple locations or weather models
        response = responses[0]
        # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        # print(f"Elevation {response.Elevation()} m asl")
        # print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")
        # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(3).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
        hourly_wind_direction_10m = hourly.Variables(5).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
        hourly_data["precipitation"] = hourly_precipitation
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
        hourly_data['date'] = hourly_data['date'].to_series().dt.strftime("%Y-%m-%d %H:%M:%S")

        hourly_weather = pd.DataFrame(data = hourly_data)
        hourly_weather.set_index('date', inplace= True)

        # hourly_weather['HDD'] = hourly_weather.apply(lambda x: self.compute_HDD(max_temp= x['temperature_2m'], min_temp= x['temperature_2m']), axis= 1)
        # hourly_weather['CDD'] = hourly_weather.apply(lambda x: self.compute_CDD(max_temp= x['temperature_2m'], min_temp= x['temperature_2m']), axis= 1)
        
        return hourly_weather["temperature_2m"]

    def weather_forecast_nyiso(self):
        df_year = []

        for i in range(12*self.n_years):
            url_date = self.increment_date_by_month(self.start_year, i)
            zip_url = f'csv/lfweather/{url_date}lfweather_csv.zip'

            #GET request
            response = requests.get(self.url+zip_url)

            #check if it went through
            if response.status_code == 200:
                #extract zip file
                with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    #initialize an empty list

                    dfs =[]

                    #loop through CSV
                    for file_name in file_list:
                        if file_name.endswith('.csv'):
                            with zip_ref.open(file_name) as csv_file:
                                df = pd.read_csv(csv_file)
                                df = df[(df['Station ID'].isin(['ISP', 'LGA', 'JFK'])) & (df['Vintage'] == 'Forecast')]                    
                            dfs.append(df)
                    
                    #merge dataframes
                    combined_df = pd.concat(dfs, ignore_index=True)
                    df_year.append(combined_df)
            else:
                print('Failed to download ZIP files')

        weather = pd.concat(df_year, ignore_index=True)
        weather.dropna(axis= 1, inplace= True)
        weather['Forecast Date'] = pd.to_datetime(weather['Forecast Date'])
        weather['Vintage Date'] = pd.to_datetime(weather['Vintage Date'])

        weather['Forecast Date'] + pd.DateOffset(days=1)

        weather['HDD'] = weather.apply(lambda x: self.compute_HDD(max_temp= x['Max Temp'], min_temp= x['Min Temp']), axis= 1)
        weather['CDD'] = weather.apply(lambda x: self.compute_CDD(max_temp= x['Max Temp'], min_temp= x['Min Temp']), axis= 1)

        #############################################################################
        ##### NEED TO GET HOURLY FORECASTS OF WEATHER INSTEAD OF DAILY FORECAST ##### 
        #############################################################################

        return weather
    
    def load_forecast(self):
        df_year = []

        for i in range(12*self.n_years):
            url_date = self.increment_date_by_month(self.start_year, i)
            zip_url = f'csv/isolf/{url_date}isolf_csv.zip'

            #GET request
            response = requests.get(self.url+zip_url)

            #check if it went through
            if response.status_code == 200:
                #extract zip file
                with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    #initialize an empty list

                    dfs =[]

                    #loop through CSV
                    for file_name in file_list:
                        if file_name.endswith('.csv'):
                            with zip_ref.open(file_name) as csv_file:
                                df = pd.read_csv(csv_file)
                            
                            dfs.append(df)
                    
                    #merge dataframes
                    combined_df = pd.concat(dfs, ignore_index=True)
                    df_year.append(combined_df)
            else:
                print('Failed to download ZIP files')

        load = pd.concat(df_year, ignore_index=True)
        load.dropna(axis=1, inplace= True)
        load.drop_duplicates('Time Stamp', inplace= True)
        load['Time Stamp'] = pd.to_datetime(load['Time Stamp'], format="%m/%d/%Y %H:%M")

        # format of timestamp does not include seconds must add
        load['Time Stamp'] = load['Time Stamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
        load.set_index('Time Stamp', inplace= True)

        return load['Longil']
    
    def capacity(self):
        df_year = []

        for i in range(12*self.n_years):
            url_date = self.increment_date_by_month(self.start_year, i)
            zip_url = f'csv/ExternalLimitsFlows/{url_date}ExternalLimitsFlows_csv.zip'

            #GET request
            response = requests.get(self.url+zip_url)

            #check if it went through
            if response.status_code == 200:
                #extract zip file
                with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    #initialize an empty list

                    dfs =[]

                    #loop through CSV
                    for file_name in file_list:
                        if file_name.endswith('.csv'):
                            with zip_ref.open(file_name) as csv_file:
                                df = pd.read_csv(csv_file)
                            
                            dfs.append(df)
                    
                    #merge dataframes
                    combined_df = pd.concat(dfs, ignore_index=True)
                    df_year.append(combined_df)
            else:
                print('Failed to download ZIP files')        

        flows = pd.concat(df_year, ignore_index=True)
        flows.drop(columns= ['Interface Name', 'Point ID'], inplace= True)
        flows.drop_duplicates('Timestamp', inplace= True)
        
        flows['Timestamp'] = pd.to_datetime(flows['Timestamp'])
        flows.set_index('Timestamp', inplace= True)

        flows_5m = flows.resample('5min').sum()
        flows_60m = flows_5m.resample('60min').mean()
        
        return flows_60m['Positive Limit (MWH)']

if __name__ == '__main__':
    data = DataScrapping(start_date= 20200101, n_years= 2)
    data.get_features()
    print(data.features)