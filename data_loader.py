import fastf1
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json

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
    # 'R' (retired), 'D' (disqualified) etc become NaN, fill with 21
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
    
    # Arrays for new model inputs
    history_X = []       # LSTM input
    current_driver_ids = []
    current_team_ids = []
    current_grid_pos = [] # Dense input
    targets = []
    
    # Sort by year and round to ensure chronological order
    df = df.sort_values(['Year', 'Round'])
    
    for driver in drivers:
        driver_data = df[df['Abbreviation'] == driver]
        
        if len(driver_data) <= seq_length:
            continue
            
        # Features to use for HISTORY: 
        # GridPosition, TeamID, EventID, LastPosition, CompoundID, TyreLife
        features = driver_data[['GridPosition', 'TeamID', 'EventID', 'Position', 'CompoundID', 'TyreLife']].values
        
        # Target data
        target_pos = driver_data['Position'].values
        driver_ids = driver_data['DriverID'].values
        team_ids = driver_data['TeamID'].values
        grid_positions = driver_data['GridPosition'].values
        
        for i in range(len(features) - seq_length):
            # Input Sequence (Last 5 races)
            history_X.append(features[i : i + seq_length])
            
            # Target Race Info (The race we are predicting)
            target_idx = i + seq_length
            
            current_driver_ids.append(driver_ids[target_idx])
            current_team_ids.append(team_ids[target_idx])
            current_grid_pos.append(grid_positions[target_idx])
            
            # Label
            targets.append(target_pos[target_idx])
            
    return (
        np.array(history_X),
        np.array(current_driver_ids),
        np.array(current_team_ids),
        np.array(current_grid_pos),
        np.array(targets)
    )

if __name__ == "__main__":
    # Fetch 2021-2023 for training, 2024 for testing
    data = fetch_race_data([2021, 2022, 2023, 2024])
    processed_data = preprocess_data(data)
    
    h_X, d_ids, t_ids, g_pos, y = create_sequences(processed_data)
    
    # Normalize Continuous Features
    # History Features: [Grid, Team, Event, Pos, Comp, Life]
    h_X = h_X.astype(float)
    h_X[:, :, 0] /= 20.0
    h_X[:, :, 1] /= processed_data['TeamID'].max()
    h_X[:, :, 2] /= processed_data['EventID'].max()
    h_X[:, :, 3] /= 21.0
    h_X[:, :, 4] /= processed_data['CompoundID'].max()
    h_X[:, :, 5] /= 100.0
    
    # Current Grid Pos
    g_pos = g_pos.astype(float) / 20.0
    
    # Target normalization
    y = y / 21.0
    
    # Save Data
    np.save('history_X.npy', h_X)
    np.save('driver_ids.npy', d_ids)
    np.save('team_ids.npy', t_ids)
    np.save('grid_pos.npy', g_pos)
    np.save('y.npy', y)
    processed_data.to_csv('processed_f1_data.csv', index=False)
    
    # Save Metadata
    metadata = {
        'num_drivers': int(processed_data['DriverID'].max() + 1),
        'num_teams': int(processed_data['TeamID'].max() + 1),
        'max_event_id': int(processed_data['EventID'].max()),
        'max_compound_id': int(processed_data['CompoundID'].max())
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f)
        
    print(f"Saved {len(y)} sequences.")
    print(f"Metadata: {metadata}")
