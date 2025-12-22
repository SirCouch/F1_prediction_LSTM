import torch
import pandas as pd
import numpy as np
import fastf1
from model import F1LSTM
import os

# Enable cache
fastf1.Cache.enable_cache('f1_cache')

def predict_race(year, round_num):
    # Load model
    input_size = 6 # Grid, Team, Event, Pos, Compound, TyreLife
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = F1LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    
    try:
        model.load_state_dict(torch.load('f1_lstm_model.pth'))
        model.eval()
    except FileNotFoundError:
        print("Model file not found. Run 'python f1_cli.py update' first.")
        return

    # Load historical data
    df = pd.read_csv('processed_f1_data.csv')
    
    # Get race info
    session = fastf1.get_session(year, round_num, 'R')
    session.load(laps=True, telemetry=False, weather=False)
    results = session.results
    laps = session.laps
    
    predictions = []
    
    max_team_id = df['TeamID'].max()
    max_event_id = df['EventID'].max()
    max_comp_id = df['CompoundID'].max()

    for _, driver in results.iterrows():
        abbr = driver['Abbreviation']
        grid = driver['GridPosition']
        team = driver['TeamName']
        event = session.event['EventName']
        
        try:
            team_id = df[df['TeamName'] == team]['TeamID'].iloc[0]
            event_id = df[df['EventName'] == event]['EventID'].iloc[0] 
        except IndexError:
            team_id = 0
            event_id = 0
            
        # Get last 5 race results for this driver
        driver_history = df[df['Abbreviation'] == abbr].tail(5)
        
        if len(driver_history) < 5:
            # Fill with current grid if not enough history
            history_features = np.zeros((5, 6))
            history_features[:, 0] = grid / 20.0
            history_features[:, 1] = team_id / max_team_id
            history_features[:, 2] = event_id / max_event_id
            history_features[:, 3] = grid / 21.0
            history_features[:, 4] = 0 # Default compound
            history_features[:, 5] = 0 # Default life
        else:
            history_features = driver_history[['GridPosition', 'TeamID', 'EventID', 'Position', 'CompoundID', 'TyreLife']].values.copy().astype(float)
            history_features[:, 0] /= 20.0
            history_features[:, 1] /= max_team_id
            history_features[:, 2] /= max_event_id
            history_features[:, 3] /= 21.0
            history_features[:, 4] /= max_comp_id
            history_features[:, 5] /= 100.0
        
        input_tensor = torch.from_numpy(history_features).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_norm = model(input_tensor).item()
            pred_pos = pred_norm * 21.0
            predictions.append({
                'Driver': abbr,
                'Grid': grid,
                'PredictedPosition': pred_pos
            })
            
    # Sort by predicted position
    pred_df = pd.DataFrame(predictions).sort_values('PredictedPosition')
    pred_df['PredictedRank'] = range(1, len(pred_df) + 1)
    
    print(f"\nPredictions for {year} {session.event['EventName']}:")
    print(pred_df[['PredictedRank', 'Driver', 'Grid', 'PredictedPosition']].to_string(index=False))
    
    return pred_df

if __name__ == "__main__":
    # Test on a 2024 race (e.g., Round 1)
    predict_race(2024, 1)
