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
    
    def update_features(self, new_feature:pd.DataFrame, name:str):
        new_feature.columns = [name]
        
        if self.features.empty:
            self.features = new_feature
        else:
            
            self.features.merge(new_feature, left_index= True, right_index= True)
    
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
        real_time['Time Stamp'] = pd.to_datetime(real_time['Time Stamp'])
        real_time.set_index('Time Stamp', inplace= True)
        real_time = real_time.resample('60min').mean()

        self.update_features(real_time['LBMP ($/MWHr)'], name= 'RT_prices' )
        return real_time
    
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
        day_ahead['Time Stamp'] = pd.to_datetime(day_ahead['Time Stamp'])
        day_ahead.set_index('Time Stamp', inplace= True)

        self.update_features(day_ahead['LBMP ($/MWHr)'], name= 'DA_prices' )
        return day_ahead

    def weather_forecast(self):
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
        load.set_index('Time Stamp', inplace= True)

        self.update_features(load['Longil'], name= 'Load_Forecast')

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
        flows['Timestamp'] = pd.to_datetime(flows['Timestamp'])

        flows.set_index('Timestamp', inplace= True)
        flows.drop(columns= ['Interface Name', 'Point ID'], inplace= True)
        flows_5m = flows.resample('5min').sum()
        flows_60m = flows_5m.resample('60min').mean()
        
        self.update_features(flows_60m['Positive Limit (MWH)'], name= 'Capacity')
        return flows_60m['Positive Limit (MWH)']

if __name__ == '__main__':
    data = DataScrapping(start_date= 20200101, n_years= 1)
    print(data.features)