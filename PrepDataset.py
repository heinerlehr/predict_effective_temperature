# General imports
import sys
import math
import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from pathlib import Path
from dotenv import dotenv_values
from argparse import ArgumentParser, Namespace

# Basic variables and functions for data preparation

# Key and labels of the different features
farm_features = {'ET': 'ET inside','T':'T inside', 'RH': 'RH inside','AH': 'AH inside', 'AS': 'air speed', 'CO2':'CO2 inside', 'weight':'weight'}
weather_features = {'et': 'ET outside','t': 'T outside', 'rh': 'RH outside', 'ah':'AH outside','ws': 'wind speed', 'wd': 'wind direction', 'pc': 'precipitation', 'p': 'air pressure'}
targets_features = {'t_ET': 'ET target','t_T':'T target', 't_RH': 'RH target', 't_AS': 'AS target'}
# Maximum day of production (for filtering). Minimum is assumed to be 0
MAX_DAY_OF_PRODUCTION = 50
# Cutoff for Effective Temperature (for filtering)
ET_MIN = 10
ET_MAX = 50

def create_effective_temperature(T:pd.DataFrame, RH:pd.DataFrame, AS:pd.DataFrame, CO2:pd.DataFrame)->pd.DataFrame:
    """
    Creates an effective temperature DataFrame by merging temperature (T), relative humidity (RH),
    airspeed (AS), and carbon dioxide (CO2) DataFrames. Performs data transformations and cleaning
    before returning the resulting DataFrame.
    
    Args:
        T (pd.DataFrame): DataFrame containing temperature data.
        RH (pd.DataFrame): DataFrame containing relative humidity data.
        AS (pd.DataFrame): DataFrame containing airspeed data.
        CO2 (pd.DataFrame): DataFrame containing carbon dioxide data.
        
    Returns:
        pd.DataFrame: DataFrame containing the merged and transformed data.
    """
    
    T   = T.rename(columns={'value': 'T'}).drop(columns=['observable_name','round_id','round_number'])
    RH  = RH.rename(columns={'value': 'RH'}).drop(columns=['observable_name','round_id','round_number'])
    AS  = AS.rename(columns={'value': 'AS'}).drop(columns=['observable_name','round_id','round_number'])
    CO2 = CO2.rename(columns={'value': 'CO2'}).drop(columns=['observable_name','round_id','round_number'])
    df = pd.merge(pd.merge(pd.merge(T, RH, on=['time', 'day_of_production', 'x', 'y', 'z']), 
                        AS , on=['time', 'day_of_production', 'x', 'y', 'z']), 
                        CO2, on=['time', 'day_of_production', 'x', 'y', 'z'])
    df = transform(df)
    df = clean(df)
    df = df.drop(columns=['x', 'y', 'z'])
    df = reduce_to_hourly(df)
    return df

def post_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform post-processing on the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be processed.

    Returns:
        pd.DataFrame: The processed DataFrame.

    """
    df = df[['time','date','hour','day_of_production']+list(farm_features.keys())+list(weather_features.keys())+list(targets_features.keys())]
    df = df.sort_values(by=['day_of_production', 'hour'])
    df = df.set_index('time')
    df = df.asfreq('h')
    
    # Add missing values
    if df.isna().sum().sum() > 0:
        df = df.interpolate()
    return df

def prep_weather_data(weather:pd.DataFrame, start:datetime, end:datetime)->pd.DataFrame:
    """
    Preprocesses weather data. Create a continuous hourly date range from start to end
    and maps measured weather variables onto by that range, interpolating missing values.
    
    Args:
        weather (pd.DataFrame): The weather data to be preprocessed.
        start (datetime): The start date of the date range.
        end (datetime): The end date of the date range.
        
    Returns:
        pd.DataFrame: The preprocessed weather data with interpolated values and merged with the date range.
    """
    
    weather['date'] = weather['time'].dt.date
    weather['hour'] = weather['time'].dt.hour
    # Weather data for some hours is missing, we interpolate the values
    date_range = pd.date_range(start=start, end=end+timedelta(hours=0.5), freq='H')
    df = pd.DataFrame(index=date_range, columns=['date','hour'])
    df['date'] = df.index.date
    df['hour'] = df.index.hour
    df['day_of_production'] = [(row['date']-start.date()).days for _,row in df.iterrows()]
    df = df.merge(weather, left_on=['date','hour'], right_on=['date','hour'],how='left').drop(columns=['day'])
    if df.isna().sum().sum() > 0:
        df = df.interpolate()
    df['et'] = effective_temperature_for_weight_gain(df['t'], df['rh'], 0)
    return df

def merge_weather_data(df:pd.DataFrame, weather:pd.DataFrame, start:datetime)->pd.DataFrame:
    """
    Merge weather data with the observational data. Merging is done on the date and hour columns.
    The resulting DataFrame is sorted by date and hour and the columns are reordered.

    Args:
        df (pd.DataFrame): The DataFrame to merge the weather data with.
        weather (pd.DataFrame): The weather data DataFrame to merge.
        start (datetime): The start datetime for merging.

    Returns:
        pd.DataFrame: The merged DataFrame with weather data.

    """
    df = weather.merge(df, left_on=['date','hour'], right_on=['date','hour'],how='left').drop(columns=['time_y','day_of_production_y']).rename(columns={'day_of_production_x':'day_of_production','time_x':'time'})
    return df

def prep_targets(targets:pd.DataFrame)->pd.DataFrame:
    """
    Preprocesses the target data by calculating the effective temperature (ET) for weight gain.
    
    Args:
        targets (pd.DataFrame): The target data containing columns 'T' (temperature), 'RH' (relative humidity),
                                and 'AS' (air speed).
    
    Returns:
        pd.DataFrame: The preprocessed target data with additional column 'ET' (effective temperature) and
                      renamed columns 'day_of_production', 't_T', 't_RH', 't_AS', and 't_ET'.
    """
    targets['ET'] = effective_temperature_for_weight_gain(targets['T'], targets['RH'], targets['AS'])
    targets.rename(columns={'Day':'day_of_production','T':'t_T','RH':'t_RH','AS':'t_AS','ET':'t_ET'}, inplace=True)
    return targets

def merge_targets(df:pd.DataFrame, targets:pd.DataFrame)->pd.DataFrame:
    """
    Merge target values into the dataset based on the day of production. The target values are interpolated if missing.

    Parameters:
    df (pd.DataFrame): The dataset to merge the target values into.
    targets (pd.DataFrame): The target values to merge into the dataset.

    Returns:
    pd.DataFrame: The dataset with the target values merged.

    """
    # Add target values to the dataset
    days = df['day_of_production'].unique()

    for feature in targets_features.keys():
        df[feature] = -1

    for day in days:
        for feature in targets_features.keys():
            df.loc[df['day_of_production']==day, feature] = np.interp(day, targets['day_of_production'], targets[feature])
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the given DataFrame by removing rows with missing values,
    filtering out rows with negative values in 'x', 'y', and 'z' columns,
    filtering out rows with 'day_of_production' values outside the range [0, MAX_DAY_OF_PRODUCTION),
    and optionally filtering out rows with 'ET' values outside the range (ET_MIN, ET_MAX).

    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned.

    Returns:
    pd.DataFrame: The cleaned DataFrame.
    """
    df = df.dropna()
    df = df[(df.x > 0) & (df.y > 0) & (df.z > 0)]
    df = df[(df['day_of_production'] >= 0) & (df['day_of_production'] < MAX_DAY_OF_PRODUCTION)]
    if 'ET' in df.columns:
        df = df[(df['ET'] < ET_MAX) & (df['ET'] > ET_MIN)]
    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the input DataFrame by calculating the effective temperature (ET) for weight gain.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing columns 'T' (temperature), 'RH' (relative humidity),
                       and 'AS' (air speed).

    Returns:
    pd.DataFrame: The transformed DataFrame with an additional column 'ET' (effective temperature).

    """
    df['ET'] = effective_temperature_for_weight_gain(df['T'], df['RH'], df['AS'])
    return df
def transform(df:pd.DataFrame)->pd.DataFrame:
    
    df['ET'] = effective_temperature_for_weight_gain(df['T'], df['RH'], df['AS'])
    return df

def reduce_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the given DataFrame to hourly data by aggregating the values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing time series data.

    Returns:
    pd.DataFrame: The reduced DataFrame with hourly data.

    """
    df['hour'] = df['time'].dt.hour
    aggmeth = {'time': 'mean', 'ET': 'median', 'T': 'median', 'RH': 'median', 'AS': 'median', 'CO2': 'median'}
    df = df.groupby(['day_of_production', 'hour']).agg(aggmeth).reset_index()
    df['date'] = df['time'].dt.date
    return df

def add_weight_target(df:pd.DataFrame, weight:pd.DataFrame)->pd.DataFrame:
    """
    Adds the weight target to the dataset. The weight target is interpolated if missing.

    Parameters:
    df (pd.DataFrame): The dataset to add the weight target to.
    weight (pd.DataFrame): The weight target to add to the dataset.

    Returns:
    pd.DataFrame: The dataset with the weight target added.

    """
    days = df['day_of_production'].unique()
    for day in days:
        df.loc[df['day_of_production']==day, 'weight'] = weight[weight['Day']==day]['Weight'].values[0]
    return df

def add_absolute_humidity(df:pd.DataFrame)->pd.DataFrame:
    df['AH'] = [absolute_humidity(row['RH'], row['T']) for _,row in df.iterrows()]
    df['ah'] = [absolute_humidity(row['rh'], row['t']) for _,row in df.iterrows()]
    return df


def effective_temperature_for_weight_gain(t:float, rh:float, v:float)->float:
        """
        Calculates the effective temperature for weight gain based on the given parameters.

        Parameters:
        t (float): Dry bulb temperature in degrees Celsius.
        rh (float): Relative humidity as a percentage.
        v (float): Air speed in meters per second.

        Returns:
        float: The effective temperature for weight gain.

        References:
        - Effective Temperature for Poultry and Pigs in Hot Climate
            https://www.intechopen.com/books/animal-husbandry-and-nutrition/effective-temperature-for-poultry-and-pigs-in-hot-climate
            DOI: 10.5772/intechopen.72821
        - Tao and Xin work: https://pdfs.semanticscholar.org/ec37/8c494f7e80d4715dfcd1408a47ae7588cfa6.pdf
        """
        
        # Constants for weight gain parameters
        c = 0.15
        d = 41
        e = 1

        # Calculate the effective temperature for weight gain
        return 0.794 * t + 0.25 * wetbulb_temperature(t, rh) + 0.7 - c * (d - t) * (v**e-0.2**e)

def wetbulb_temperature(t:float, rh:float)->float:
    """
    Calculates the wet-bulb temperature using the Stull formula.
    
    Parameters:
    t (float): The dry-bulb temperature in degrees Celsius.
    rh (float): The relative humidity as a percentage.
    
    Returns:
    float: The wet-bulb temperature in degrees Celsius.
    
    References:
    - Stull formula; see https://journals.ametsoc.org/doi/10.1175/JAMC-D-11-0143.1
    """
    # Stull formula	Tw = T * arctan[0.151977 * (rh% + 8.313659)^(1/2)] + arctan(T + rh%) - arctan(rh% - 1.676331) + 0.00391838 *(rh%)^(3/2) * arctan(0.023101 * rh%) - 4.686035
    return t*np.arctan(0.151977*(rh + 8.313659)**0.5)+np.arctan(t+rh)-np.arctan(rh-1.676331)+0.00391838*(rh)**1.5*np.arctan(0.023101*rh)-4.686035


def absolute_humidity(RH:float, T:float)->float:
    """
    Calculate the absolute humidity in kg/m^3 given the relative humidity (RH) and temperature (T).
    
    Parameters:
    RH (float): Relative humidity in percentage.
    T (float): Temperature in degrees Celsius.
    
    References:
    https://pubs.aip.org/aip/jpr/article-abstract/31/2/387/241937/The-IAPWS-Formulation-1995-for-the-Thermodynamic?redirectedFrom=fulltext
    
    Returns:
    float: Absolute humidity in g/m^3.
    """
    
    T0 = 273.15 #K
    T = T + T0
    TC = 647.096 #K Critical temperature for water
    t = 1-T/TC
    PC = 22.064e6 # hPa Critical pressure for water
    a1 =-7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502
    PS = PC*math.exp(TC/T*(a1*t + a2*t**1.5 + a3*t**3 + a4*t**3.5 + a5*t**4 + a6*t**7.5))
    
    PA = PS*RH/100
    
    RW = 461.5
    AH = PA/(RW*T)*1000
    
    return AH #g/m^3

def get_rt_args()->Namespace:
    """Reads command line arguments and returns a Namespace object with them

    Returns:
        Namespace: Namespace object with the command line arguments
    """    
    parser = ArgumentParser()
    parser.add_argument("-c", "--client",
                        action="store", dest="client",
                        help="The client")
    parser.add_argument("-pc", "--cycle",
                        action="store", dest="cycle",
                        help="The production cycle")
    return parser.parse_args()


# Running the data set preparation or if the prepared data set exists, loading it

# Should define the path to the data directory
env = dotenv_values('.env')
REPORTBASEDIR = env['REPORTBASEDIR']

args = get_rt_args()

basepath = Path('.')/"data"
datapath = Path(REPORTBASEDIR)/args.client/f'Cycle {args.cycle}'

if not basepath.exists():
    print("The data directory does not exist")
    sys.exit(1)
    
FLOCK = args.cycle
full_dataset = basepath/(FLOCK + '.parquet')
if not full_dataset.exists():
    T =  pd.read_csv(datapath/"raw"/"temperature.csv", parse_dates=['time'])
    T = clean(T)
    FLOCK = T.loc[0, 'round_id'] 
    start = T['time'].min()
    end = T['time'].max()
    RH = pd.read_csv(datapath/"raw"/"humidity.csv", parse_dates=['time'])
    AS = pd.read_csv(datapath/"raw"/"airspeed.csv", parse_dates=['time'])
    CO2 = pd.read_csv(datapath/"raw"/"co2.csv", parse_dates=['time'])
    weight = pd.read_csv(basepath/"ROSS308_weight_target.csv")
    weather = pd.read_csv(datapath/"raw"/"weather_station.csv", parse_dates=['time'])

    df = create_effective_temperature(T, RH, AS, CO2)
    df = add_weight_target(df, weight)
    weather = prep_weather_data(weather, start=start, end=end)
    df = merge_weather_data(df, weather, start=start)
    targets = pd.read_csv(datapath/"Targets.csv")
    targets = prep_targets(targets)
    df = merge_targets(df, targets)
    df = add_absolute_humidity(df)
    
    df = post_process(df)
    
    full_dataset = basepath/(FLOCK + '.parquet')
    df.to_parquet(full_dataset)
else:
    print(f"The data set for client {args.client} and flock {args.cycle} already exists")
