import requests
import zipfile
import io
import pandas as pd

def increment_date_by_month(date_str, increment):
    # Step 1: Convert the string to a datetime object
    date = pd.to_datetime(date_str, format='%Y%m%d')
    
    # Step 2: Add one month
    next_month_date = date + pd.DateOffset(months=increment)
    
    # Step 3: Convert the datetime object back to a string in the desired format
    return next_month_date.strftime('%Y%m%d')

def increment_date_by_year(date_str, increment):
    # Step 1: Convert the string to a datetime object
    date = pd.to_datetime(date_str, format='%Y%m%d')
    
    # Step 2: Add the specified number of years
    next_year_date = date + pd.DateOffset(years=increment)
    
    # Step 3: Convert the datetime object back to a string in the desired format
    return next_year_date.strftime('%Y%m%d')

def NYISODataRT(year):
    url = 'http://mis.nyiso.com/public/'
    df_year = []
    for i in range(12):
        url_date = increment_date_by_month(year, i)
        zip_url = f'csv/realtime/{url_date}realtime_zone_csv.zip'

        #GET request
        response = requests.get(url+zip_url)

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
    
    return pd.concat(df_year, ignore_index=True)

def NYISODataDA(year):
    url = 'http://mis.nyiso.com/public/'
    df_year = []
    for i in range(12):
        url_date = increment_date_by_month(year, i)
        zip_url = f'csv/damlbmp/{url_date}damlbmp_zone_csv.zip'

        #GET request
        response = requests.get(url+zip_url)

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
    
    return pd.concat(df_year, ignore_index=True)



if __name__ == "__main__":
   start_date = 20200101
   df_data = []
   
   for i in range(9):
      df = NYISODataRT(increment_date_by_year(start_date, i))
      df_data.append(df)
    
   data = pd.concat(df_data, ignore_index=True)

   data.to_csv('data/Real Time Locational Based Marginal Pricing.csv')
   
