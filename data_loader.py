import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

# Create a cache directory for FastF1
cache_dir = 'f1_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
fastf1.Cache.enable_cache(cache_dir)

def fetch_race_data(years):
    all_results = []
    
    for year in years:
        print(f"Fetching data for {year}...")
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule['EventFormat'] != 'testing']
        
        for _, race in tqdm(races.iterrows(), total=len(races)):
            try:
                session = fastf1.get_session(year, race['RoundNumber'], 'R')
                # Load laps to get tire info
                session.load(laps=True, telemetry=False, weather=False)
                results = session.results
                laps = session.laps
                
                # Extract majority tire info for each driver
                tire_data = []
                for driver in results['Abbreviation']:
                    driver_laps = laps[laps['Driver'] == driver]
                    if not driver_laps.empty:
                        # Get most used compound
                        compound = driver_laps['Compound'].value_counts().idxmax()
                        # Get max tire life
                        max_life = driver_laps['TyreLife'].max()
                        tire_data.append({'Abbreviation': driver, 'Compound': compound, 'TyreLife': max_life})
                    else:
                        tire_data.append({'Abbreviation': driver, 'Compound': 'UNKNOWN', 'TyreLife': 0})
                
                tire_df = pd.DataFrame(tire_data)
                df = results[['DriverNumber', 'Abbreviation', 'TeamName', 'GridPosition', 'ClassifiedPosition', 'Status', 'Points']].copy()
                df = df.merge(tire_df, on='Abbreviation')
                
                df['Year'] = year
                df['Round'] = race['RoundNumber']
                df['EventName'] = race['EventName']
                
                all_results.append(df)
            except Exception as e:
                print(f"Error fetching {year} Round {race['RoundNumber']}: {e}")
                
    return pd.concat(all_results, ignore_index=True)

def preprocess_data(df):
    # Drop rows where critical data is missing
    df = df.dropna(subset=['GridPosition', 'ClassifiedPosition'])
    
    # Convert ClassifiedPosition to numeric
    df['Position'] = pd.to_numeric(df['ClassifiedPosition'], errors='coerce')
    df['Position'] = df['Position'].fillna(21)
    
    # Simple encoding
    df['TeamID'] = df['TeamName'].astype('category').cat.codes
    df['DriverID'] = df['Abbreviation'].astype('category').cat.codes
    df['EventID'] = df['EventName'].astype('category').cat.codes
    df['CompoundID'] = df['Compound'].astype('category').cat.codes
    
    # Fill remaining NaNs
    df['GridPosition'] = pd.to_numeric(df['GridPosition'], errors='coerce').fillna(20)
    df['TyreLife'] = pd.to_numeric(df['TyreLife'], errors='coerce').fillna(0)
    
    return df

def create_sequences(df, seq_length=5):
    # Sequence of past race results for each driver
    drivers = df['Abbreviation'].unique()
    X, y = [], []
    
    # Sort by year and round to ensure chronological order
    df = df.sort_values(['Year', 'Round'])
    
    for driver in drivers:
        driver_data = df[df['Abbreviation'] == driver]
        
        if len(driver_data) <= seq_length:
            continue
            
        # Features to use: GridPosition, TeamID, EventID, LastPosition, CompoundID, TyreLife
        features = driver_data[['GridPosition', 'TeamID', 'EventID', 'Position', 'CompoundID', 'TyreLife']].values
        targets = driver_data['Position'].values
        
        for i in range(len(features) - seq_length):
            X.append(features[i : i + seq_length])
            y.append(targets[i + seq_length])
            
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Fetch 2021-2023 for training, 2024 for testing
    data = fetch_race_data([2021, 2022, 2023])
    processed_data = preprocess_data(data)
    X, y = create_sequences(processed_data)
    
    # Normalize features to [0, 1] range
    X = X.astype(float)
    X[:, :, 0] = X[:, :, 0] / 20.0
    X[:, :, 1] = X[:, :, 1] / processed_data['TeamID'].max()
    X[:, :, 2] = X[:, :, 2] / processed_data['EventID'].max()
    X[:, :, 3] = X[:, :, 3] / 21.0
    X[:, :, 4] = X[:, :, 4] / processed_data['CompoundID'].max()
    X[:, :, 5] = X[:, :, 5] / 100.0 # TyreLife usually < 100
    
    # Target normalization (1-21 -> 0-1)
    y = y / 21.0
    
    np.save('X.npy', X)
    np.save('y.npy', y)
    processed_data.to_csv('processed_f1_data.csv', index=False)
    print(f"Saved {len(X)} sequences.")
