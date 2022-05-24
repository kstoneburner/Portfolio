from pathlib import Path
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime
from IPython.display import clear_output
import time
import pickle
from io import StringIO
import datetime as DT

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors

class process_covid():
    
    def process_patient_impact(self,**kwargs):
        
        data_folder_name = "raw_data"
        
        #//*** Identify columns to process. Sum() the Yes Values of those COlumns
        base_cols = ["case_month","res_state","state_fips_code","res_county","county_fips_code"]
        process_cols = ['hosp_yn','icu_yn','death_yn','underlying_conditions_yn'] 

        #//*** Convert case_month to datetime format
        convert_date = True
        reset_index = True

        rename_cols = []

        for key,value in kwargs.items():

            if key == 'data_folder_name':
                data_folder_name = value
                
            if key == 'base_cols':
                base_cols = value

            if key == 'process_cols':
                process_cols = value
        
            if key == 'convert_date':
                convert_date = value

            if key == 'reset_index':
                reset_index = value

            if key == 'rename_cols':
                rename_cols = value

        current_dir = Path(os.getcwd()).absolute()
        data_dir = current_dir.joinpath(data_folder_name)
        base_filename = "covid_people.pkl.zip"
        out_df = pd.DataFrame()

        start_time = time.time()

        #//**** Load each file
        for filename in os.listdir(data_dir):
            if base_filename in filename:
                print("Processing:",filename)
                loop_df = pd.read_pickle(data_dir.joinpath(filename))
                print("Loaded: ", len(loop_df), " records", "Current Record Size:", len(out_df))
                
                #//*** Drop Rows with NaN county FIPS Code
                loop_df = loop_df.dropna(subset="county_fips_code")
                
                loop_df = self.sum_cols(loop_df, base_cols=base_cols, process_cols=process_cols)
                out_df = pd.concat([out_df,loop_df])
                
                
        columns = list(out_df.columns)
        
        #//*** Convert column names if specified
        print(f"Elapsed Time: {int(time.time() - start_time)}s")

        for rc in rename_cols:

            find = rc[0]
            replace = rc[1]

            columns = [replace if item == find else item for item in columns]

        out_df.columns = columns

        if convert_date:
            out_df['case_month'] = pd.to_datetime(out_df['case_month'])

        if reset_index:
            out_df = out_df.reset_index(drop=True)

        return out_df
    
    def sum_cols(self,df,**kwargs):
        
        process_cols = None
        
        base_cols = None

        for key,value in kwargs.items():
            if key == 'process_cols':
                process_cols = value
        
            if key == 'base_cols':
                base_cols = value
                
        if process_cols == None:
            print("Needs to declare Process Cols. These are the columns to process and sum")
            return df
        
        if base_cols == None:
            print("Needs to declare Base Cols. These are the utility description columns to keep but not sum.")
            return df
        
        #//*** Declare output dataframe. Returned Results go here
        out_df = pd.DataFrame()
        
        
        #//*** COmbine the Base and Process columns
        use_cols = base_cols + process_cols

        #//*** Sum the cases by month, hate to lose the demographic data
        #//*** GRoup By Month
        for date_group in df[use_cols].groupby('case_month'):

            for col in date_group[1].columns:

                #//*** Convert Yes == 1, everything else to 0
                if col in process_cols:

                    #//*** Identify the non-Yes values
                    unique_replace = date_group[1][ date_group[1][col] != "Yes" ][col].unique()

                    #//*** Replace Non-Yes values to 0
                    for replace in unique_replace:
                        date_group[1][col] = date_group[1][col].replace(replace,0)

                    #//*** Change the Yes Values to 1
                    date_group[1][col] = date_group[1][col].replace("Yes",1)

            #//*** For Each Month Group by COunty
            for fips_group in date_group[1].groupby('county_fips_code'):

                tds =  fips_group[1].iloc[0].copy()


                for col in process_cols:

                    tds[col] = fips_group[1][col].sum()
                
                tds['case_count'] = len(fips_group[1])
                out_df = pd.concat([out_df,tds],axis=1)
            
        
        return out_df.transpose()



def download_data(**kwargs):

    data_folder_name = "raw_data"

    current_dir = Path(os.getcwd()).absolute()
    data_dir = current_dir.joinpath(data_folder_name)

    confirmed_data_filename = data_dir.joinpath("z_us_confirmed.csv")
    death_data_filename = data_dir.joinpath("z_us_death_cases.csv")
    vaccine_data_filename = data_dir.joinpath("z_us_vaccination.csv")
    county_vaccine_data_filename = data_dir.joinpath("z_us_county_vaccination.csv.zip")
    county_hospital_filename = data_dir.joinpath("z_county_hospital.csv.zip")






    #//***********************************************************************************************
    #//*** California COVID Data website:
    #//**************************************
    #//*** https://data.chhs.ca.gov/dataset/covid-19-time-series-metrics-by-county-and-state
    #//***********************************************************************************************

    
    try:
        response = requests.get("https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv")
        if response.ok:
            print("US Confirmed Data Downloaded")
            f = open(confirmed_data_filename, "w")
            f.write(response.text)
            f.close()
            print("US Confirmed Data Written to file.")
    except:
        print("US Confirmed Data: Trouble Downloading From Johns Hopkins Github")

    
    try:
        response = requests.get("https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv")
        if response.ok:
            print("US Deaths Data Downloaded")
            f = open(death_data_filename, "w")
            f.write(response.text)
            f.close()
            print("US Death Data Written to file.")
    except:
        print("US Death Data: Trouble Downloading From Johns Hopkins Github")
        
    try:
        #response = requests.get("https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD")
        response = requests.get("https://data.cdc.gov/api/views/unsk-b7fc/rows.csv?accessType=DOWNLOAD")
        if response.ok:
            print("Vaccination Data Downloading")
            f = open(vaccine_data_filename, "w")
            f.write(response.text)
            f.close()
            print("US Vaccination Data Written to file.")
    except:
        print("US Vaccine Data: Trouble Downloading From CDC")

    
    #//*** CDC Vaccination County Data
    #//*** Source: https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh
    print("County Vaccination Data Downloading")
    response = requests.get("https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD")
    if response.ok:

        county_vaccine_data_filename = data_dir.joinpath("z_us_county_vaccination.csv")
        county_vaccine_data_filename_1 = data_dir.joinpath("z_1_us_county_vaccination.csv.zip")
        county_vaccine_data_filename_2 = data_dir.joinpath("z_2_us_county_vaccination.csv.zip")
        
        print("County Vaccination Data Downloaded")
        #//*** Write CSV File
        f = open(str(county_vaccine_data_filename).replace(".zip",""), "w")

        f.write(response.text)
        f.close()

        print("Loading Raw Vaccine File: ")
        tdf = pd.read_csv(county_vaccine_data_filename,low_memory=False)
        print("Saving First Half of DataFrame: ")
        tdf.iloc[:int(len(tdf)/2)].to_pickle(county_vaccine_data_filename_1)
        print("Saving Second Half of DataFrame: ")
        tdf.iloc[int(len(tdf)/2):].to_pickle(county_vaccine_data_filename_2)

        #df = pd.concat([pd.read_pickle(county_vaccine_data_filename_1),pd.read_pickle(county_vaccine_data_filename_2)])
        print("Delete Raw Vaccine File:",county_vaccine_data_filename)
        os.remove(county_vaccine_data_filename)


        print("US County Vaccination Data Written to file.")

    #Hospitalizations - State
    #https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD

    try:
        #response = requests.get("https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD")
        #response = requests.get("https://healthdata.gov/api/views/g62h-syeh/rows.csv?accessType=DOWNLOAD")
        response = requests.get("https://healthdata.gov/api/views/anag-cw7u/rows.csv?accessType=DOWNLOAD")
        if response.ok:
            print("Hospitalization Data Downloading")
            #f = open(county_hospital_filename, "w")
            #f.write(response.text)
            #f.close()
            
            #//*** Convert text to FileIO Object, to load into CSV
            #//*** Pickle to save the DataFrame with Compression
            pd.read_csv(StringIO(response.text)).to_pickle(county_hospital_filename)
            print("county Hospitalization Data Written to file.")
    except:
        print("US Hospitalization: Trouble Downloading From Healthdata.gov")


def build_county_case_death(**kwargs):
    data_folder_name = "raw_data"
    confirmed_data_filename = "z_us_confirmed.csv"
    death_data_filename =  "z_us_death_cases.csv"
    county_daily_df_filename = "z_county_daily_df.csv.zip"

    for key,value in kwargs.items():
        if key == 'folder':
            data_folder_name = value

        if key == 'confirm':
            confirmed_data_filename = value

        if key == 'death':
            death_data_filename = value

        if key == 'export':
            county_daily_df_filename = value
        

    current_dir = Path(os.getcwd()).absolute()
    data_dir = current_dir.joinpath(data_folder_name)

    confirmed_data_filename = data_dir.joinpath(confirmed_data_filename)
    death_data_filename = data_dir.joinpath(death_data_filename)
    county_daily_df_filename = data_dir.joinpath("z_county_daily_df.csv.zip")

    start_time = time.time()
    #//****************************************************
    #//*** Prepare Confirmed Cases and Deaths For Merge
    #//****************************************************

    print("Loading Raw Confirm Cases Data....")
    confirm_df = pd.read_csv(confirmed_data_filename)

    confirm_df = confirm_df[confirm_df['Admin2'] != 'Unassigned']

    #//*** Convert Confirmed Date Columns to Date Objects
    cols = []
    confirm_date_cols = []
    for col in confirm_df.columns:
        if "/" not in col:
            cols.append(col)
        else:
            cols.append(datetime.strptime(col, "%m/%d/%y").date())
            confirm_date_cols.append(datetime.strptime(col, "%m/%d/%y").date())

    confirm_df.columns = cols

    print("Loading Raw Deaths Data....")

    death_df = pd.read_csv(death_data_filename)

    death_df

    death_df['Province_State'].unique()
    death_df = death_df[death_df['iso2'] =='US']
    death_df = death_df[death_df['Province_State'] != "Diamond Princess"]
    death_df = death_df[death_df['Province_State'] != "Grand Princess"]
    death_df = death_df[death_df['Admin2'] != 'Unassigned']
    death_df.dropna(inplace=True)
    death_df['FIPS'] = death_df['FIPS'].astype(int)


    #//*** Convert Confirmed Date Columns to Date Objects
    cols = []
    death_date_cols = []

    for col in death_df.columns:
        if "/" not in col:
            cols.append(col)
        else:
            cols.append(datetime.strptime(col, "%m/%d/%y").date())
            death_date_cols.append(datetime.strptime(col, "%m/%d/%y").date())

    death_df.columns = cols


    ##///**** REBUILD COUNTY_DAILY_DF - This takes a while 15ish Minutes


    #//*** Integrate Confirmed and Deaths with Vaccine Data. Build derived Values
    i = 0

    print("Begin Merge Confirm and Deaths Columns with Vaccination Rows....")
    county_daily_df = pd.DataFrame()

    all_dfs = []

    for FIPS in death_df.sort_values(['FIPS'])['FIPS'].unique():

        
        i += 1

        attrib = death_df[death_df['FIPS'] == FIPS]



        #loop_df = pd.concat([loop_df] * (len(death_df[death_df['FIPS']==GEOID])),ignore_index=True)

        #//*** Merge Combined Key and Population. Grab a subset of FIPS from death_df
        #loop_df = loop_df.merge(death_df[death_df['FIPS']==GEOID][['FIPS','Combined_Key','Population']],left_on='GEOID',right_on='FIPS')

        #//*** Get Confirmed Values for FIPS County
        loop_df = confirm_df[confirm_df['FIPS']==FIPS][confirm_date_cols].transpose()

        loop_df = loop_df.reset_index()

        loop_df.columns = ['Date','tot_confirm']



        #//*** Build Total Deaths for FIPS County
        ds = death_df[death_df['FIPS']==FIPS][death_date_cols].transpose()

        ds = ds.reset_index()
        ds.columns = ['Date','tot_deaths']
        del ds['Date']

        #//*** Keep Relevant Columns

        for col in ['FIPS','Admin2','Province_State','Combined_Key','Population']:
            loop_df[col]=attrib[col].iloc[0]

        loop_df = loop_df[['Date','FIPS','Admin2','Province_State','Combined_Key','Population','tot_confirm']]

        #//*** Generate new rows based on length of death series
        #loop_df = pd.concat([loop_df] * len(ds),ignore_index=True)




        #//*** Join Confirmed Values
        #loop_df = loop_df.join(cs)

        #loop_df = cs

        #//*** Merge Death Values
        loop_df = loop_df.join(ds)

        #//*** Build New Confirmed Cases
        loop_df['New_Confirm']  = loop_df['tot_confirm'].diff()
        #//*** Reset Negative Cases to 0
        loop_df.loc[loop_df['New_Confirm'] < 0,f'New_Confirm']=0


        #//*** Build New Death Cases
        loop_df['New_Deaths']  = loop_df['tot_deaths'].diff()

        #//*** Reset Negative Deaths to 0
        loop_df.loc[loop_df['New_Deaths'] < 0,f'New_Deaths']=0
        #print(cs)
        #print(ds)

        #//*** Build New Confirmed 7 Day Average
        loop_df['case_7_day_avg']  = loop_df['New_Confirm'].rolling(7).mean()

        #//*** Build New Deaths 7 Day Average
        loop_df['death_7_day_avg']  = loop_df['New_Deaths'].rolling(7).mean()

        #//*** Build New Confirmed 100k 7 day  Average
        loop_df['case_100k_avg']  = loop_df['case_7_day_avg'] / (loop_df['Population'] / 100000 )

        #//*** Build New Confirmed 100k 7 day  Average
        loop_df['death_100k_avg']  = loop_df['death_7_day_avg'] / (loop_df['Population'] / 100000 )

        #//*** Set scaled Values to a max of 100 for heatmap purposes
        loop_df['case_scaled_100k'] = loop_df['case_100k_avg']
        loop_df['death_scaled_100k'] = loop_df['death_100k_avg']

        loop_df.loc[loop_df[f"case_scaled_100k"] > 100,f"case_scaled_100k"]=100
        loop_df.loc[loop_df[f"death_scaled_100k"] > 5,f"death_scaled_100k"]=5


        #loop_df['Date'] = loop_df['Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y").date())
        #print(vax_df[vax_df['FIPS']==GEOID])

        #loop_df = loop_df[ loop_df['Date'] >= vax_df['Date'].min() ]
        #del loop_df['FIPS']
        #loop_df = loop_df.merge(vax_df[vax_df['FIPS'] == GEOID],left_on='Date',right_on='Date',how='left')


        #//*** All Data merged and Calculated. Merge with temporary Dataframe()
        #county_daily_df = pd.concat([county_daily_df,loop_df])

        #//*** Add Loop_df to List, Concat Later
        all_dfs.append(loop_df)

        if i % 100 == 0:
            print(f"Working: {i} of {len(death_df['FIPS'].unique())}")

    county_daily_df = pd.concat(all_dfs)

    county_daily_df = county_daily_df.dropna()

    print(f"Writing county daily to File: {county_daily_df_filename}")
    county_daily_df.to_pickle(county_daily_df_filename)
    print(f"Elapsed Time: {int(time.time() - start_time)}s")


def load_data(**kwargs):

    #//*** Load Dataframe from file

    data_folder_name = "raw_data"

    #//*** Action is Either None or 'county_vaccine'
    #//*** county_vaccine is specifically load the the county vaccination data.
    #//*** This dataset is specially split to meet filesize github requiments
    action = None

    filename = None

    min_date = None

    date_col = "Date"

    trim_first_date = False
    trim_last_date = False

    clean_col_names = None
    rename_cols = None
    remove_cols = None
    auto_convert_to_float = False
    for key,value in kwargs.items():

        if key == "action":
            action = value

        if key == 'folder':
            data_folder_name = value

        if key == 'file':
            filename = value

        if key == 'min_date':
            min_date = value

        if key == 'date_col':
            date_col = value

        if key == 'filename':
            filename = value

        if key == 'trim_first_date':
            trim_first_date = value

        if key == 'trim_last_date':
            trim_last_date = value

        if key == 'clean_col_names':
            clean_col_names = value

        if key == 'rename_cols':
            rename_cols = value

        if key == 'remove_cols':
            remove_cols = value

        if key == 'auto_convert_to_float':
            auto_convert_to_float = value

    current_dir = Path(os.getcwd()).absolute()
    data_dir = current_dir.joinpath(data_folder_name)



    if action == "county_vaccine":
        county_vaccine_data_filename_1 = data_dir.joinpath("z_1_us_county_vaccination.csv.zip")
        county_vaccine_data_filename_2 = data_dir.joinpath("z_2_us_county_vaccination.csv.zip")


        #print("Loading Raw Vaccine Data")
        #//*** read Raw Vaccine csv
        county_vax_df = pd.concat([pd.read_pickle(county_vaccine_data_filename_1),pd.read_pickle(county_vaccine_data_filename_2)])


        #//*** Filter Columns to get just the Completed Values
        cols = ['Date','FIPS','Recip_County','Recip_State','Series_Complete_Pop_Pct','Series_Complete_Yes','Administered_Dose1_Pop_Pct','Administered_Dose1_Recip']


        #//*** remove States not in continental US
        county_vax_df = county_vax_df[county_vax_df["Recip_State"] != "AK" ]
        county_vax_df = county_vax_df[county_vax_df["Recip_State"] != "HI" ]
        county_vax_df = county_vax_df[county_vax_df["Recip_State"] != "AS" ]
        county_vax_df = county_vax_df[county_vax_df["Recip_State"] != "GU" ]
        county_vax_df = county_vax_df[county_vax_df["Recip_State"] != "MP" ]
        county_vax_df = county_vax_df[county_vax_df["Recip_State"] != "PR" ]
        county_vax_df = county_vax_df[county_vax_df["Recip_State"] != "VI" ]
        county_vax_df = county_vax_df[county_vax_df["FIPS"] != "UNK" ]
        county_vax_df['FIPS'] = county_vax_df['FIPS'].astype(int)
        county_vax_df['Date'] = county_vax_df['Date'].apply(lambda x: datetime.strptime(x, "%m/%d/%Y").date())

        county_vax_df = county_vax_df[cols]

        #//*** Cleanup Column Names
        ren_cols = {
        'Administered_Dose1_Recip' : 'first_dose_count',
        'Administered_Dose1_Pop_Pct' : 'first_dose_pct',
        'Series_Complete_Yes' : 'total_vaccinated_count',
        'Series_Complete_Pop_Pct' : 'total_vaccinated_percent',
        }

        cols = list(county_vax_df.columns)

        for find,replace in ren_cols.items():
            cols = [replace if i==find else i for i in cols]

        county_vax_df.columns=cols
            
        return county_vax_df



    if filename != None and action == None:


        if ".zip" in filename:

            df = pd.read_pickle(data_dir.joinpath(filename))

        else:

            df = pd.read_csv(data_dir.joinpath(filename))

        if min_date != None:

            df = df [ df[date_col] >= min_date]
        
        #//*** Remove All Dates that include the last date
        if trim_first_date:
            df = df[df[date_col] != df[date_col].unique()[0]]

        #//*** Remove All Dates that include the last date
        if trim_last_date:
            df = df[df[date_col] != df[date_col].unique()[-1]]

        #//*** Tidy up column names by removing fragments in clean_col_names
        if clean_col_names != None:

            cols = []
            for orig_col in df.columns:

                for find in clean_col_names:
                   orig_col = orig_col.replace(find,"")
                cols.append(orig_col)

            df.columns = cols

        if rename_cols != None:

            cols = []

            for col in df.columns:
                skip = False
                #//*** frt = find replace tuple. First key is find, Second is replace
                for frt in rename_cols:
                    if col == frt[0]:
                        cols.append(frt[1])
                        skip = True
                if skip:
                    continue

                cols.append(col)

            df.columns = cols

        #//*** Delete Columns if Specified
        if remove_cols != None:
            for col in remove_cols:
                #print(col,out_df.columns,col in out_df.columns)
                if col in df.columns:
                    del df[col]


        if auto_convert_to_float:

            #//*** Convert all Numeric Columns to Float if possible
            for col in df.columns:
                try:
                    df[col] = df[col].astype(float)
                except:
                    pass

        return df

def merge_df(df1,df2,**kwargs):

    by = 'Date'
    left_col = None
    right_col = None
    date_col = "Date"
    export_filename = None
    data_dir = "raw_data"
    remove_cols = None
    start_time = time.time()

    for key,value in kwargs.items():

        if key == 'by':
            by = value

        if key == 'left_col':
            left_col = value

        if key == 'right_col':
            right_col = value

        if key == 'date_col':
            date_col = value
        
        if key == 'export':
            export_filename = value

        if key == 'folder':
            data_dir = value

        if key == 'remove_cols':
            remove_cols = value


    out_df = pd.DataFrame()
    
    all_dfs = []

    for group in df1.groupby(by):
        
        clear_output(wait=True)
        print("Processing: ", group[0])

        loop_df1 = group[1].copy()

        loop_df2 = df2[df2[by] == group[0]]

        del loop_df2[date_col]


        #out_df = pd.concat([out_df,loop_df1.merge(loop_df2,left_on=left_col,right_on=right_col)])
        #//*** Merge DataFrames on by Value (Date)
        #//*** Append to all_dfs for later concatenation
        #//*** This method is quite a timesaver! as Concat is inefficient compared to list append with a single concat
        all_dfs.append(loop_df1.merge(loop_df2,left_on=left_col,right_on=right_col))
    
    out_df = pd.concat(all_dfs)
    clear_output(wait=True)
    #print("Remove Cols:",remove_cols)
    #//*** Delete Columns if Specified
    if remove_cols != None:
        for col in remove_cols:
            #print(col,out_df.columns,col in out_df.columns)
            if col in out_df.columns:
                del out_df[col]
    print(f"Elapsed Time: {int(time.time() - start_time)}s")

    #//*** Export File if value is defined
    if export_filename != None:
        current_dir = Path(os.getcwd()).absolute()
        data_dir = current_dir.joinpath(data_dir)
        export_filename = data_dir.joinpath(export_filename)

        print("Exporting DataFrame to Disk.")
        if ".zip" in str(export_filename):
            out_df.to_pickle(export_filename)
            return
        else:
            out_df.to_csv(export_filename)
            return

    #//*** Otherwise return out_df
    else:
        return out_df

def create_monthly_data(df, **kwargs):

    
    data_folder_name = "raw_data"
    export_filename = "z_county_monthly_df.csv.zip"
    export_filename = None
    date_col = "Date"
    FIPS_col = "FIPS"
    sum_cols = ['New_Confirm','New_Deaths']
    remove_cols = ['case_7_day_avg','death_7_day_avg','case_100k_avg','death_100k_avg','case_scaled_100k','death_scaled_100k']


    for key,value in kwargs.items():
        if key == 'folder':
            data_folder_name = value

        if key == 'date_col':
            date_col = value

        if key == 'FIPS_col':
            FIPS_col = value

        if key == 'sum_cols':
            sum_cols = value

        if key == 'remove_cols':
            remove_cols = value

        if key == 'export':
            export_filename = value
        

    current_dir = Path(os.getcwd()).absolute()
    data_dir = current_dir.joinpath(data_folder_name)

    if export_filename != None:
        export_filename = data_dir.joinpath(export_filename)

    start_time = time.time()

    out_df = pd.DataFrame()
    date_col = "Date"
    sum_cols = ['New_Confirm','New_Deaths']
    remove_cols = ['case_7_day_avg','death_7_day_avg','case_100k_avg','death_100k_avg','case_scaled_100k','death_scaled_100k']
    each_col = []
    start_time = time.time()
    counter = 0
    tot_count = len(df[FIPS_col].unique())
    for FIPS_group in df.groupby('FIPS'):
        FIPS_group[1].index = pd.to_datetime(FIPS_group[1]['Date'])
        counter += 1
        #//*** Group FIPS by month & Year (ie each individual month)
        for date_group in FIPS_group[1].groupby(by=[FIPS_group[1].index.month,FIPS_group[1].index.year] ):
            clear_output(wait=True)
            print(counter,"/",tot_count,"Processing FIPS:",FIPS_group[0],FIPS_group[1].iloc[0]['Combined_Key'])
            tds = date_group[1].iloc[-1].copy()
            
            for col in df.columns:
                if col == date_col:
                    tds[date_col] = date_group[1][date_col].iloc[0]
                    continue
                if col in sum_cols:
                    tds[col] = date_group[1][col].sum()
                    tds[f"{col}_100k"] = tds[col] / (tds['Population'] / 100000)
                    tds[f"{col}_avg_daily_100k"] = date_group[1][col].mean() / (tds['Population'] / 100000)
                    continue
                if col in remove_cols:
                    del tds[col]
                    continue
            #//*** TDS is a column, Store to a list for later concatenating 
            each_col.append(tds)
            
            
            
    clear_output(wait=True)

    #//*** Build a dataframe using each_col as a column. Transpose, sort by Date then reset the Index.
    out_df = pd.concat(each_col,axis=1).transpose().sort_values("Date").reset_index(drop=True)
    out_df[date_col] = pd.to_datetime(out_df[date_col])
    print(f"Elapsed Time: {int(time.time() - start_time)}s")
    #//*** If export_filename is defined, save the df to disk
    if export_filename != None:

        print("Saving Combined Monthly DataFrame to Disk.")
        if ".zip" in str(export_filename):
            out_df.to_pickle(export_filename)
            return
        else:
            out_df.to_csv(export_filename)
            return

    #//*** Return the DF
    return df
    

           
def aggregate_columns(df, **kwargs):

    #//*** Process Columns by date and another field (typically FIPS)
    #//*** Sum Columns adds the total values for all entries and returns a single entry

    data_folder_name = "raw_data"
    export_filename = None
    date_col = None
    FIPS_col = None
    process_cols = []
    by = None
    base_cols = []
    disp_cols = None
    method = "sum"

    for key,value in kwargs.items():
        if key == 'folder':
            data_folder_name = value

        if key == 'export':
            export_filename = value

        if key == 'date_col':
            date_col = value

        if key == 'FIPS_col':
            FIPS_col = value

        if key == 'process_cols':
            process_cols = value

        if key == 'by':
            by = value

        if key == 'base_cols':
            base_cols = value

        if key == 'disp_cols':
            disp_cols = value
 
        if key == 'method':
            method = value    

    current_dir = Path(os.getcwd()).absolute()
    data_dir = current_dir.joinpath(data_folder_name)

    if export_filename != None:
        export_filename = data_dir.joinpath(export_filename)

    start_time = time.time()

    df = df.replace(-999999.0,1)

    #//*** Hold all rows as a list and append once when done
    all_rows = [] 
    
    max_date = df[date_col].max()
    #//*** for each Date in df
    for date_group in df.groupby(date_col):
        clear_output(wait=True)
        try:
            print("Processing:",date_group[0], "/", max_date )
        except:
            pass
        
        for by_group in date_group[1].groupby(by):
            #display(by_group[1][disp_cols])

            #display(by_group[1])

            #//*** Create a Temporary Series
            tds = by_group[1][disp_cols].iloc[-1].copy()

            #//*** PRocess each col in process_cols
            for col in process_cols:

                #//*** Perform calc based on method
                if method == 'sum':

                    tds[col] = by_group[1][col].sum()
                    
                else:
                    print(f"Unknown Method: {method}")
                    print("quitting!")
                    return

            
            #print(tds)
            all_rows.append(tds)
        
        
    out_df = pd.concat(all_rows,axis=1).transpose()


    print(f"Elapsed Time: {int(time.time() - start_time)}s")
    #//*** If export_filename is defined, save the df to disk
    if export_filename != None:

        print("Saving Combined Monthly DataFrame to Disk.")
        if ".zip" in str(export_filename):
            out_df.to_pickle(export_filename)
            return
        else:
            out_df.to_csv(export_filename)
            return

    
def create_weekly_data(df, **kwargs):

    
    data_folder_name = "raw_data"
    export_filename = "z_county_monthly_df.csv.zip"
    export_filename = None
    date_col = "Date"
    dates = None
    FIPS_col = "FIPS"
    sum_cols = ['New_Confirm','New_Deaths']
    remove_cols = ['case_7_day_avg','death_7_day_avg','case_100k_avg','death_100k_avg','case_scaled_100k','death_scaled_100k']


    for key,value in kwargs.items():
        if key == 'folder':
            data_folder_name = value

        if key == 'date_col':
            date_col = value

        if key == 'dates':
            dates = value

        if key == 'FIPS_col':
            FIPS_col = value

        if key == 'sum_cols':
            sum_cols = value

        if key == 'remove_cols':
            remove_cols = value

        if key == 'export':
            export_filename = value
        

    current_dir = Path(os.getcwd()).absolute()
    data_dir = current_dir.joinpath(data_folder_name)

    if export_filename != None:
        export_filename = data_dir.joinpath(export_filename)

    start_time = time.time()

    out_df = pd.DataFrame()
    date_col = "Date"
    sum_cols = ['New_Confirm','New_Deaths']
    remove_cols = ['case_7_day_avg','death_7_day_avg','case_100k_avg','death_100k_avg','case_scaled_100k','death_scaled_100k']
    each_col = []
    start_time = time.time()
    counter = 0
    tot_count = len(dates)

    #//*** Convert Dates to TimeStamp
    dates = pd.Series(dates).apply(lambda x: pd.Timestamp(x))
    df[date_col] = df[date_col].apply(lambda x: pd.Timestamp(x))
    #dates = pd.to_datetime(pd.Series(dates))
    display(df.head())

    for date in dates:
        counter += 1
        #date_end = datetime.strptime(date,"%Y/%m/%d").date()
        date_start = date - DT.timedelta(days=6)

        date_df = df[ (df[date_col] >= date_start) & (df[date_col] <= date) ]

        #tdf =  date_df.groupby([date_col,FIPS_col]).agg({'New_Confirm' : ['sum'],'New_Deaths' : ['sum']})

        clear_output(wait=True)

        print(counter,"/",tot_count,"Processing Dates:",date.date(),"/",dates.iloc[-1].date(),date_df[date_col].min(),date_df[date_col].max())
        for FIPS_group in date_df.groupby(FIPS_col):

            tds = FIPS_group[1].iloc[-1].copy()
            #print(len(FIPS_group[1]))

            
            for col in df.columns:
                #if col == date_col:
                #    tds[date_col] = FIPS_group[1][date_col].iloc[0]
                #    continue
                if col in sum_cols:
                    tds[col] = FIPS_group[1][col].sum()
                    try:
                        tds[f"{col}_100k"] = tds[col] / (tds['Population'] / 100000)
                    except:
                        tds[f"{col}_100k"] = 0

                    try:
                        tds[f"{col}_avg_daily_100k"] = FIPS_group[1][col].mean() / (tds['Population'] / 100000)
                    except:
                        tds[f"{col}_avg_daily_100k"] = 0
                    continue
                if col in remove_cols:
                    del tds[col]
                    continue
            #print(tds)       
            #//*** TDS is a column, Store to a list for later concatenating 
            each_col.append(tds)
        
    clear_output(wait=True)

    #//*** Build a dataframe using each_col as a column. Transpose, sort by Date then reset the Index.
    out_df = pd.concat(each_col,axis=1).transpose().sort_values("Date").reset_index(drop=True)
    out_df[date_col] = pd.to_datetime(out_df[date_col])
    print(f"Elapsed Time: {int(time.time() - start_time)}s")
    display(out_df)
    
    #//*** If export_filename is defined, save the df to disk
    if export_filename != None:

        print("Saving Weekly Aggregated DataFrame to Disk.")
        if ".zip" in str(export_filename):
            out_df.to_pickle(export_filename)
            return
        else:
            out_df.to_csv(export_filename)
            return

    #//*** Return the DF
    return df


def create_weekly_data_v2(df, **kwargs):

    
    data_folder_name = "raw_data"
    export_filename = "z_county_monthly_df.csv.zip"
    export_filename = None
    date_col = "Date"
    dates = None
    FIPS_col = "FIPS"
    sum_cols = ['New_Confirm','New_Deaths']
    remove_cols = ['case_7_day_avg','death_7_day_avg','case_100k_avg','death_100k_avg','case_scaled_100k','death_scaled_100k']


    for key,value in kwargs.items():
        if key == 'folder':
            data_folder_name = value

        if key == 'date_col':
            date_col = value

        if key == 'dates':
            dates = value

        if key == 'FIPS_col':
            FIPS_col = value

        if key == 'sum_cols':
            sum_cols = value

        if key == 'remove_cols':
            remove_cols = value

        if key == 'export':
            export_filename = value
        

    current_dir = Path(os.getcwd()).absolute()
    data_dir = current_dir.joinpath(data_folder_name)

    if export_filename != None:
        export_filename = data_dir.joinpath(export_filename)

    start_time = time.time()

    out_df = pd.DataFrame()
    date_col = "Date"
    sum_cols = ['New_Confirm','New_Deaths']
    remove_cols = ['case_7_day_avg','death_7_day_avg','case_100k_avg','death_100k_avg','case_scaled_100k','death_scaled_100k']
    each_col = []
    start_time = time.time()
    counter = 0
    tot_count = len(dates)

    #//*** Convert Dates to TimeStamp
    dates = pd.Series(dates).apply(lambda x: pd.Timestamp(x))
    df[date_col] = df[date_col].apply(lambda x: pd.Timestamp(x))
    #dates = pd.to_datetime(pd.Series(dates))
    display(len(df))
    
    disp_cols=['Date', 'FIPS', 'Recip_County', 'Recip_State','total_vaccinated_percent', 'total_vaccinated_count', 'first_dose_pct','first_dose_count', 'Admin2', 'Province_State', 'Combined_Key','Population', 'tot_confirm', 'tot_deaths', 'New_Confirm', 'New_Deaths']
    all_df = []
    for group in df[df[date_col].isin(dates)].groupby("FIPS"):
        counter +=1

        group[1]['New_Confirm'] = group[1]['tot_confirm'].diff().fillna(0)
        group[1]['New_Deaths'] = group[1]['tot_deaths'].diff().fillna(0)
        group[1]['pv_tot_confirm'] = group[1]['New_Confirm'].cumsum()
        if 'pv_tot_confirm' not in disp_cols:
            disp_cols.append('pv_tot_confirm')
        group[1]['pv_tot_death'] = group[1]['New_Deaths'].cumsum()
        if 'pv_tot_death' not in disp_cols:
            disp_cols.append('pv_tot_death')


    
        all_df.append(group[1])



    clear_output(wait=True)

    #//*** Build a dataframe using each_col as a column. Transpose, sort by Date then reset the Index.
    out_df = pd.concat(all_df).sort_values("Date").reset_index(drop=True)


    out_df[date_col] = pd.to_datetime(out_df[date_col])
    print(f"Elapsed Time: {int(time.time() - start_time)}s")
    display(out_df)
    
    #//*** If export_filename is defined, save the df to disk
    if export_filename != None:

        print("Saving Weekly Aggregated DataFrame to Disk.")
        if ".zip" in str(export_filename):
            out_df.to_pickle(export_filename)
            return
        else:
            out_df.to_csv(export_filename)
            return

    #//*** Return the DF
    return df


def eda_models(model,x,y, **kwargs):
    model_action = None
    labels = None

    disp_col = None
    test_size = .2 #//*** Test Split Size
    title = ""
    for key,value in kwargs.items():
        if key == 'disp_col':
            disp_col = value

        if key == 'test_size':
            test_size = value

        if key == 'title':
            title = value

        if key == 'labels':
            labels = value

    if model == 'linear' or model == 'lr':
        model_action = 'linear_regression'

    if model == 'linear_regression_analysis' or model == 'lra':
        model_action = 'linear_regression_analysis'

    if model_action  == 'linear_regression':

        x = np.array(x)

        #//*** Reshape x if a single dimension
        if x.ndim == 1:
            x = x.reshape(-1,1)

        y = np.array(y)

        #//*** Perform Test Train Split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle=False)

        regr = LinearRegression()
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_test)
        xx_pred = regr.predict(x_train)
        print("R2: ", r2_score(y_test, y_pred))
        
        #//*** If Not defined, use a range value for x display.
        try:
            if disp_col == None:
                disp_col = range(len(y_pred))
        except:
            disp_col = np.array(disp_col)[len(y_pred) * -1 : ]


        plt.figure(figsize=(12, 8))
        plt.style.use('fivethirtyeight')
        plt.plot(disp_col,y_test, label='actual')
        plt.plot(disp_col,y_pred, label='predict')
        plt.title(title) 
        plt.legend()

        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(y_train, label='actual')

        plt.plot(xx_pred, label='predict')

        plt.legend()

        plt.show()

    elif model_action == 'linear_regression_analysis':
        x = np.array(x)

        #//*** Reshape x if a single dimension
        if x.ndim == 1:
            x = x.reshape(-1,1)

        y = np.array(y)

        regr = LinearRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)
        
        print("R2: ", r2_score(y, y_pred))

        print("Coef:", regr.coef_, "Intercept: ",regr.intercept_)
        if labels == None:
            print("Coef:", regr.coef_, "Intercept: ",int(regr.intercept_))

        else:
            msg = "Coef:\n"
            for x in range(len(labels)):
                msg += f"\t{labels[x]}: {round(regr.coef_[x],4)}\n"

            print(msg, "Intercept: ",regr.intercept_)
        
        #//*** If Not defined, use a range value for x display.
        try:
            if disp_col == None:
                disp_col = range(len(y_pred))
        except:
            disp_col = np.array(disp_col)[len(y_pred) * -1 : ]


        plt.figure(figsize=(12, 8))
        plt.style.use('fivethirtyeight')
        plt.plot(disp_col,y, label='actual')
        plt.plot(disp_col,y_pred, label='predict')
        plt.title(title) 
        plt.legend()

        plt.show()


def qplot(x,y=None):


        try:
            x = x.values
        except:
            pass

        if isinstance(y,type(None)):
            plt.figure(figsize=(12, 8))
            plt.style.use('fivethirtyeight')
            plt.plot(range(len(x)))
            plt.show()
        else:
            try:
                y = y.values
            except:
                pass



            plt.figure(figsize=(12, 8))
            plt.style.use('fivethirtyeight')
            plt.plot(x,y)
            plt.show()

def build_stats_for_analysis(df,**kwargs):

    
    #//*** Default values
    kw = {
        'action' : None,
        'vax_col' : 'vax_ct',
        'pop_col' : 'pop',
        'by_col' : 'Date',
        'agg' : 'sum',
        'cols' : None,
        'reset_index' : True,
        'label' : None,
        'cumsum' : None,
        'build_100k' : False,
        'outcome_ratios': None,
        'verbose' : True,
    }
    
    #//*******************************************************************************************
    #//*** Assign every key,value pair in kwargs.
    #//*** It's a little insecure since there is no validation and bad things can be passed in
    #//*** It's not accepting random API info from the wild. It's fine
    #//*******************************************************************************************
    for key,value in kwargs.items():
        kw[key] = value
        
    
    verbose = kw['verbose']    

    
    if isinstance(kw['cols'],type(None)):
        print("Need to define cols= as a list of columns to aggregate")
        return None
    
    out_df = df.groupby(kw['by_col'], dropna=False)[kw['cols']].agg([kw['agg']])
    if verbose:
        print("Aggregating Columns by", kw['by_col'], "Action:",kw['agg'], " Columns:",kw['cols'])
    
    #//*** Rename the columns to the input columns
    out_df.columns = kw['cols']
    
    #//*** Add Label Column if it is Not None
    if isinstance(kw['label'],type(None)) == False:
        #//*** Add Label Column
        label_column = kw['label'][0]
        label_value = kw['label'][1]
        if verbose:
            print("Adding Label Column: ", label_column,"-",label_value)

        out_df[ label_column ] = label_value

        #//*** Reorder Columns so label comes first
        out_df = out_df[[label_column] + list(out_df.columns)[:-1]]

    
    if kw['reset_index']:
        if verbose:
            print("Resetting Index")
        out_df.reset_index(inplace=True)
    
    #//*** Build Unvaccinated Value
    if verbose:
        print("Building Unvaccinated")

    try:
        out_df['uvx'] = out_df[kw['pop_col']] - out_df[kw['vax_col']]
    except:
        if verbose:
            print("Skipping Unvaccinated")
        pass

    if isinstance(kw['cumsum'],list):
        if verbose:
            print("Building Cumulative Sums for Columns: ", kw['cumsum'])
        #//*** Add Cumulatively Sum Columns if defined - Defaults to None
        for col in kw['cumsum']:
            out_df[f"{col}_tot"] = out_df[col].cumsum()

    #//*** Build percapita 100k for every column after 'pop'
    if kw['build_100k']:
        #//*** Get a list of every column after pop_col
        cols_100k = list(out_df.columns[list(out_df.columns).index(kw['pop_col'])+1:])
        
        if verbose:
            print("Building 100k Values:",cols_100k)
        for col in cols_100k:
            #//*** Verify it's a float column    
            if out_df[col].dtype == 'float64':
                out_df[f"{col}_100k"] = (out_df[col] / (out_df[kw['pop_col']] / 100000)).astype(int)


    #//*** Build Vax and UnVax Percentages
    if verbose:
        print("Build Vax and UnVax Percentage Columns")

    #//*** Build if Vax Col Exists
    if kw['vax_col'] in out_df.columns:
        out_df['vax_pct'] = out_df[ kw['vax_col'] ] / out_df[ kw['pop_col'] ]

    if 'uvx' in out_df.columns:
        out_df['uvx_pct'] = out_df['uvx'] / out_df[ kw['pop_col'] ]
    
    #//*** Build Outcome Ratios if defined
    if isinstance(kw['outcome_ratios'],type(None))==False:
        if verbose:
            print("Building Vaccine Outcome Ratios")
        
        for cn in kw['outcome_ratios']:
            
            col1 = cn[0]
            col2 = cn[1]
            name = cn[2]
            
            #//*** Check if there are rows where vaccine count is 0
            if len(out_df[ out_df[ kw['vax_col'] ] == 0  ]) > 0:
                #//*** If yes, get a list of rows with with vax = 0. Set the initial value to the last instance
                initial_value = out_df[ out_df[ kw['vax_col'] ] == 0  ].iloc[-1][col2]
            else:
                #//*** Just use the first row value
                initial_value = out_df.iloc[0][col2]

            #//*** Reads as 1 / vax_ct or unvax_ct / (outcomes totals from vaccine start)
            out_df[name] = 1 / ( out_df[ col1 ] / ( out_df[col2] - initial_value ) ).fillna(0)
    
    if "Date" in out_df.columns:
        out_df["Date"] = pd.to_datetime(out_df["Date"])
    return out_df

def rename_state_abbreviations(input_series):

    rename_dict = {
        'AL' : 'Alabama',
        'AR' : 'Arkansas',
        'AZ' : 'Arizona',
        'CA' : 'California',
        'CO' : 'Colorado',
        'CT' : 'Connecticut',
        'DC' : 'District of Columbia',
        'DE' : 'Delaware',
        'FL' : 'Florida',
        'GA' : 'Georgia',
        'IA' : 'Iowa',
        'ID' : 'Idaho',
        'IL' : 'Illinois',
        'IN' : 'Indiana',
        'KS' : 'Kansas', 
        'KY' : 'Kentucky',
        'LA' : 'Louisiana',
        'MA' : 'Massachusetts',
        'MD' : 'Maryland',
        'ME' : 'Maine',
        'MI' : 'Michigan',
        'MN' : 'Minnesota',
        'MO' : 'Missouri',
        'MS' : 'Mississippi',
        'MT' : 'Montana',
        'NC' : 'North Carolina',
        'ND' : 'North Dakota',
        'NE' : 'Nebraska',
        'NH' : 'New Hampshire',
        'NJ' : 'New Jersey',
        'NM' : 'New Mexico',
        'NV' : 'Nevada',
        'NY' : 'New York',
        'OH' : 'Ohio',
        'OK' : 'Oklahoma',
        'OR' : 'Oregon',
        'PA' : 'Pennsylvania',
        'RI' : 'Rhode Island',
        'SC' : 'South Carolina',
        'SD' : 'South Dakota',
        'TN' : 'Tennessee',
        'TX' : 'Texas',
        'UT' : 'Utah',
        'VA' : 'Virginia',
        'VT' : 'Vermont',
        'WA' : 'Washington',
        'WI' : 'Wisconsin',
        'WV' : 'West Virginia',
        'WY' : 'Wyoming',         
    }

    for key,value in rename_dict.items():
        input_series = input_series.replace(key,value)

    return input_series

class collect_dataframes():
    def __init__(self):
        #//*** Initialize Collection of DataFrames as Dictionary
        self.dfs = {

        }

    def add(self,df,key,description='None'):
        self.dfs[key] = {
            'df' : df,
            'desc' : description,
            'key' : key
        }

    def get(self,key):
        if key in self.dfs.keys():
            return self.dfs[key]['df']
        else:
            print("DataFrame Key:[",key,"] Not Found")
            print("Current DFs")
            for key, df_dict in self.dfs.items():
                print("[",key,"] Desc:", self.dfs[key]['desc'])
            return None

    def l(self,mode="desc" ):
        self.list(mode)

    def list(self,mode="desc"):

        if mode == 'list':
            return self.dfs.keys()

        if mode == 'desc':
            print("===================================================")
            print("Collected Dataframe Count: ", len(self.dfs.keys()))
            print("===================================================")
            for key, df_dict in self.dfs.items():
                print(f"[{key}] Desc:", self.dfs[key]['desc'])

            return ""
        print("Unknown list Mode: ",mode)
        print("Valid Options: list,desc")
        return ""

def cumsum_cols(df,**kwargs):

    cols = []
    by = "fips_code"
    date_col = "Date"
    suffix = ""
    prefix = ""
    zero = False

    for key,value in kwargs.items():

        if key == "cols":
            cols = value

        if key == "by":
            by = value

        if key == "date_col":
            date_col = value

        if key == "suffix":
            suffix = value

        if key == "prefix":
            prefix = value

        if key == "zero":
            zero = value

    counter = 0
    all_df = []

    #//*** Group by col Usually FIPS
    for group in df.groupby(by):

        #//*** Sort Col (FIPS) by Datae
        loop_df = group[1].sort_values(date_col)

        #//*** Cumsum Columns
        for col in cols:

            #//*** If Zero, Subtract the First Value from the list to get a running cumsum
            if zero:
                loop_df[f'{prefix}{col}{suffix}'] = loop_df[col].cumsum() - loop_df[col].iloc[0]
            else:
                loop_df[f'{prefix}{col}{suffix}'] = loop_df[col].cumsum()

        #//*** Add Partial Data Frame to List
        all_df.append(loop_df)

    #//*** ConCat list of DFs to a single DF
    out_df = pd.concat(all_df).sort_values("Date").reset_index(drop=True)


    out_df[date_col] = pd.to_datetime(out_df[date_col])

    return out_df

def sum_cols(df,**kwargs):

    cols = []
    by = "fips_code"
    date_col = "Date"
    suffix = ""
    prefix = ""

    for key,value in kwargs.items():

        if key == "cols":
            cols = value

        if key == "by":
            by = value

        if key == "date_col":
            date_col = value

        if key == "suffix":
            suffix = value


    counter = 0
    all_df = []

    #//*** Group by col Usually FIPS
    for group in df.groupby(by):

        #//*** Sort Col (FIPS) by Datae
        loop_df = group[1].sort_values(date_col)

        #//*** Cumsum Columns
        for col in cols:
            loop_df[f'{col}{suffix}'] = loop_df[col].sum()

        #//*** Add Partial Data Frame to List
        all_df.append(loop_df)

    #//*** ConCat list of DFs to a single DF
    out_df = pd.concat(all_df).sort_values("Date").reset_index(drop=True)

    
    out_df[date_col] = pd.to_datetime(out_df[date_col])

    return out_df

def build_100k(df,**kwargs):

    cols = []
   
    date_col = "Date"
    pop_col = "pop"
    suffix = "_100k"
    prefix = ""

    for key,value in kwargs.items():

        if key == "cols":
            cols = value

        if key == "pop_col":
            pop_col = value

        if key == "date_col":
            date_col = value

        if key == "suffix":
            suffix = value


    counter = 0
    all_df = []



    #//*** 100k Columns
    for col in cols:
        df[f'{col}{suffix}'] = (df[col].fillna(0) / (df[pop_col] / 100000)).astype(int)

    return df

def diff_cols(df,**kwargs):

    cols = []
    by = "fips_code"
    date_col = "Date"
    suffix = ""



    for key,value in kwargs.items():

        if key == "cols":
            cols = value

        if key == "by":
            by = value

        if key == "date_col":
            date_col = value

        if key == "suffix":
            suffix = value


    counter = 0
    all_df = []

    #//*** Group by col Usually FIPS
    for group in df.groupby(by):

        #//*** Sort Col (FIPS) by Datae
        loop_df = group[1].sort_values(date_col)

        #//*** Cumsum Columns
        for col in cols:

            loop_df[col[1]] = loop_df[col[0]].diff().fillna(0)

        #//*** Add Partial Data Frame to List
        all_df.append(loop_df)

    #//*** ConCat list of DFs to a single DF
    out_df = pd.concat(all_df).sort_values("Date").reset_index(drop=True)

    
    out_df[date_col] = pd.to_datetime(out_df[date_col])

    return out_df

def plot(input_df,**kwargs):

    kw = {
        'col' : None,
        'action' : None,
        'show' : False,
        'vlines' : [],
    }
    for key,value in kwargs.items():

        kw[key] = value

    if kw['col'] == None:
        print("Need col=Column to plot")
        return
    if kw['col'] not in input_df.columns:
        print("Col must be a valid column.",kw[col],"not in",list(input_df.columns))
    
    year_color = {
        2020 : 'steelblue',
        2021 : 'red',
        2022 : 'gold',
    }
    if kw['action'] == "plot_by_year":

        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12, 8))

        
        #//*** Plot each Yeah
        for year in input_df['Date'].dt.year.unique():
            tdf = input_df[input_df['Date'].dt.year >= year]

            

            plt.plot(tdf['Date'],tdf[kw['col']],color=year_color[year],label=year)


        font = {

            'size' : 15
        }

        for vline in kw['vlines']:
            v_date = pd.to_datetime(vline['date'])
            v_text = vline['text']
            v_height = vline['height']
            v_offset = vline['offset']
            plt.vlines(x = input_df[input_df['Date'] == v_date]['Date'], ymin=0, ymax=v_height, color='purple', linestyle="dotted")
            plt.text(input_df[input_df['Date'] ==  v_date - pd.Timedelta(v_offset, unit='W')]['Date'],v_height*1.02,v_text, fontdict=font)

        plt.legend()
        if kw['show']:
            plt.show()
            return
        else:
            return plt

    print("Need a Valid Action= value")
    return

def qgeo(df,**kwargs):
    
    df = df.copy()
    
    column=None
    vcenter=None
    suptitle=None
    title=None
    reverse = False
    cmap='coolwarm'
    vmin=None
    vmax=None
    std=2
    state_geo = None
    figsize=(5,4)
    input_ax=None
    fontsize=30
    return_ax = None
    timelapse = False
    for key,value in kwargs.items():
        if key == 'column':
            column = value
        
        if key == 'vcenter':
            vcenter = value
        
        if key == 'suptitle':
            suptitle = value

        if key == 'title':
            title = value
        
        if key == 'cmap':
            cmap = cmap

        if key == 'reverse':
            reverse = value
            
        if key == 'vmin':
            vmin = value

        if key == 'vmax':
            vmax = value
        
        if key == 'figsize':
            figsize = value
        
        if key == 'state_geo':
            state_geo = value
        
        if key == 'std':
            std = value

        if key == 'ax':
            input_ax = value

        if key == 'return_ax':
            return_ax = value

        if key == 'timelapse':
            timelapse = value
    
    if column not in df.columns:
        print(f"Column: {column} not found in ")
        print(list(df.columns))
        return

    #//*** If No Center Default to Mean
    if vcenter == None:
        vcenter = df[column].mean()
            
    #//*** If vmin and vmax Not Specified. Calculate vmin and vmax based on a multiple
    #//*** of standard deviation. Default is 2
    if vmin == None:
        vmin = df[column].mean() - (std * df[column].std() )
        
        if vmin < df[column].min():
            vmin = df[column].min()

    if vmax == None:
        vmax = df[column].mean() + (std * df[column].std() )
            
        if vmax > df[column].max():
            vmax = df[column].max()
    
        
    #//*** Temporarily replace values larger than vmax with vmax
    for val in df[df[column] <= vmin][column].values:
        df[column] = df[column].replace(val,vmin)

    vmin = df[column].min()

    #//*** Temporarily replace values larger than vmax with vmax
    for val in df[df[column] >= vmax][column].values:
        df[column] = df[column].replace(val,vmax)

    vmax = df[column].max()
    
    cmap=cm.get_cmap(cmap)
    
    if reverse:
        cmap=cmap.reversed()
        
    if (vcenter <= vmin) or (vcenter >= vmax):
        vcenter = (vmin+vmax) /2
    
    # Dynamic ColorBar, set vmin & vmax, centered around the US mean
    # dynamic range:
    try:
        divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        ax= df.plot(column=column,cmap=cmap,norm=divnorm,ax=input_ax )
    except:
        print("Trouble with normalization values!", vmin, vcenter, vmax)
        print(df[column].describe(),vmin, vcenter, vmax)
        #if timelapse == False:
        #    print("Trouble with normalization values!", vmin, vcenter, vmax)
        #    print(df[column].describe(),vmin, vcenter, vmax)
        #else:
        #    divnorm=None
        #    ax= df.plot(column=column,cmap=cmap,ax=input_ax )

    



     

    
    if isinstance(state_geo,type(None)):
        #//*** Draw State Shapes over top, Set Color to transparant with Black edgecolor
        ax = df.plot(categorical=True,legend=True, linewidth=1,edgecolor=(0,0,0,.5),color=(1,1,1,0),ax=ax)
    else:
        ax = df.plot(categorical=True,legend=True, linewidth=.5,edgecolor=(0,0,0,.15),color=(1,1,1,0),ax=ax)
        
        keep_state = df['STATEFP'].unique()

        all_state = state_geo['STATEFP'].unique()

        for state in all_state:
            if state not in keep_state:
                state_geo = state_geo[state_geo['STATEFP'] != state]
        
        ax = state_geo.plot(categorical=True,legend=True, linewidth=1.5,edgecolor=(0,0,0,.5),color=(1,1,1,0),ax=ax)
        
    
    ax.grid(False)
    ax.axis('off')

    plt.suptitle(suptitle,fontsize=fontsize)
    plt.title(title)
    
    if timelapse:
        if pd.to_datetime(df['Date'].max().date()) < pd.to_datetime('2021-07-02'):
            plt.annotate(f"[ {df['Date'].max().date()} ]",(0,0),xycoords='axes fraction',fontsize=fontsize*.75)

        elif (pd.to_datetime(df['Date'].max().date()) >= pd.to_datetime('2021-07-02')) and (pd.to_datetime(df['Date'].max().date()) < pd.to_datetime('2021-12-24')):
            plt.annotate(f"[ {df['Date'].max().date()} ] Delta",(0,0),xycoords='axes fraction',fontsize=fontsize*.75)

        elif pd.to_datetime(df['Date'].max().date()) >= pd.to_datetime('2021-12-24'):
            plt.annotate(f"[ {df['Date'].max().date()} ] Omicron BA.1",(0,0),xycoords='axes fraction',fontsize=fontsize*.75)

    else:
        plt.annotate(f"[ {df['Date'].min().date()} - {df['Date'].max().date()} ]",(0,0),xycoords='axes fraction',fontsize=fontsize*.75)

    
    fig = ax.get_figure()
    fig = plt.gcf()
    fig.set_size_inches(figsize)
    cax = fig.add_axes([.9, 0.1, 0.03, 0.8])
    cax.grid(False)
    if isinstance(divnorm,type(None)):
        sm = plt.cm.ScalarMappable(cmap=cmap)    
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=divnorm)
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cb = fig.colorbar(sm, cax=cax)
    #cb.set_ticks([cb_dict[vmin],cb_dict[vcenter],cb_dict[vmax]])
    #cb.set_ticks(tick_list)

    #tick_list=[vmin,vcenter,vmax]
    #cb.set_ticks(tick_list)
    if return_ax == None:
        plt.show()
    else:
        return fig

    
def part_geoplot(df,**kwargs):

    all_geo_df = None
    state_geo_df = None
    color = "blue"
    column = None
    figsize = (5,8)
    title = None
    suptitle = None
    for key,value in kwargs.items():

        if key == 'all_geo_df':
            all_geo_df = value

        if key == 'state_geo_df':
            state_geo_df = value

        if key == 'color':
            color = value

        if key == 'column':
            column = value

        if key == 'figsize':
            figsize = value

        if key == 'suptitle':
            suptitle = value

        if key == 'title':
            title = value

    ax = None
    if isinstance(all_geo_df,type(None))==False:
        ax = all_geo_df.plot(categorical=True, linewidth=1,edgecolor=(0,0,0,.15),color=(1,1,1,0), ax=ax)

        if isinstance(state_geo_df,type(None))==False:
            state_geo_df = state_geo_df[state_geo_df['STATEFP'].isin(list(all_geo_df['STATEFP'].unique()))]
            ax = state_geo_df.plot(categorical=True,legend=True, linewidth=3,edgecolor=(0,0,0,.5),color=(1,1,1,0),ax=ax)

    df.plot(column=column,color=color,ax=ax)

    ax.grid(False)
    ax.axis('off')

    fig = ax.get_figure()
    fig = plt.gcf()
    fig.set_size_inches(figsize)

    plt.suptitle(suptitle)
    plt.title(title)

    plt.show()

def standard_recalc_cols(df,**kwargs):

    df=df.copy()

    to_build_100k = True
    pop_col = 'pop'
    tot_confirm_col = 'tot_confirm'
    tot_death_col = 'tot_deaths'

    for key,value in kwargs.items():

        if key == 'to_build_100k':
            to_build_100k = value

        if key == 'pop_col':
            pop_col = value

        if key == 'tot_confirm_col':
            tot_confirm_col = value

        if key == 'tot_death_col':
            tot_death_col = value

    #//*** Build Daily Values
    df['New_Confirm'] = df[tot_confirm_col].diff().fillna(0)
    df['New_Deaths'] = df[tot_death_col].diff().fillna(0)

    #//*** Build Post Vax Totals
    df['pv_New_Confirm_tot'] = df['New_Confirm'].cumsum()
    df['pv_New_Deaths_tot'] = df['New_Deaths'].cumsum()

    #//*** Build Total Hospital Beds
    df['beds_covid_tot'] = df['beds_covid'].cumsum()
    df['icu_covid_tot'] = df['icu_covid'].cumsum()

    df['beds_tot']      = df['beds'].cumsum()
    df['beds_used_tot'] = df['beds_used'].cumsum()
    df['all_icu_tot']   = df['all_icu'].cumsum()
    df['icu_used_tot']  = df['icu_used'].cumsum()


    #//*** Build Post Vax Hospital Beds

    df['pv_beds_covid_tot'] = df['beds_covid_tot'] - df['beds_covid_tot'].iloc[0]
    df['pv_icu_covid_tot'] = df['icu_covid_tot'] - df['icu_covid_tot'].iloc[0]

    df['pv_beds_tot']      = df['beds_tot']      - df['beds_tot'].iloc[0]
    df['pv_beds_used_tot'] = df['beds_used_tot'] - df['beds_used_tot'].iloc[0]
    df['pv_all_icu_tot']   = df['all_icu_tot']   - df['all_icu_tot'].iloc[0]
    df['pv_icu_used_tot']  = df['icu_used_tot']  - df['icu_used_tot'].iloc[0]

    if to_build_100k:
        for col in [tot_confirm_col,tot_death_col,'New_Confirm','New_Deaths','pv_New_Confirm_tot',
                    'pv_New_Deaths_tot','beds_covid','icu_covid','beds_covid_tot','icu_covid_tot','pv_beds_covid_tot','pv_icu_covid_tot','vax_ct',
                    'pv_beds_tot','pv_beds_used_tot','pv_all_icu_tot','pv_icu_used_tot']:

            #//*** Some Vaccines Report as NaN, fill as
            df[f"{col}_100k"] = (df[col].fillna(0) / (df[pop_col] / 100000)).astype(int)
        
    #//*** Build Bed Percentages
    df['all_bed_util'] = df['pv_beds_used_tot'] / df['pv_beds_tot'] 
    df['covid_bed_util'] = df['pv_beds_covid_tot'] / df['pv_beds_used_tot'] 
    df['icu_util'] = df['pv_icu_used_tot'] / df['pv_all_icu_tot']
    df['icu_covid_util'] = df['pv_icu_covid_tot'] / df['pv_icu_used_tot']

    return df