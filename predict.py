import torch
import pandas as pd
import numpy as np
import fastf1
from model import F1Predictor
import os
import json

# Enable cache
fastf1.Cache.enable_cache('f1_cache')

def predict_race(year, round_num):
    # Load metadata
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print("Metadata not found. Run 'python f1_cli.py update' first.")
        return

    # Load model
    hidden_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = F1Predictor(
        num_drivers=metadata['num_drivers'],
        num_teams=metadata['num_teams'],
        hidden_size=hidden_size
    ).to(device)
    
    try:
        model.load_state_dict(torch.load('f1_lstm_model.pth'))
        model.eval()
    except FileNotFoundError:
        print("Model file not found. Run 'python f1_cli.py update' first.")
        return

    # Load historical data for lookups and history sequences
    df = pd.read_csv('processed_f1_data.csv')
    
    # Get race info
    session = fastf1.get_session(year, round_num, 'R')
    session.load(laps=True, telemetry=False, weather=False)
    results = session.results
    
    predictions = []
    
    # Normalization constants (from training)
    # History Features: [Grid, Team, Event, Pos, Comp, Life]
    max_team_id = df['TeamID'].max()
    max_event_id = df['EventID'].max()
    max_comp_id = df['CompoundID'].max()
    
    for _, driver in results.iterrows():
        abbr = driver['Abbreviation']
        grid = driver['GridPosition']
        team = driver['TeamName']
        
        # 1. Lookup IDs
        try:
            # Get the ID used in training for this driver/team
            # We take the last known ID for this abbreviation
            d_id_lookup = df[df['Abbreviation'] == abbr]['DriverID']
            if not d_id_lookup.empty:
                driver_id = d_id_lookup.iloc[0]
            else:
                # New driver? Use 0 or handle gracefully. 
                # For now, let's assume unknown = 0 if not found, but ideally we'd have an UNK token
                driver_id = 0 
                
            t_id_lookup = df[df['TeamName'] == team]['TeamID']
            if not t_id_lookup.empty:
                team_id = t_id_lookup.iloc[0]
            else:
                team_id = 0
                
        except IndexError:
            driver_id = 0
            team_id = 0
            
        # 2. Construct History Sequence (Last 5 races)
        driver_history = df[df['Abbreviation'] == abbr].tail(5)
        
        if len(driver_history) < 5:
            # Fill with zeros or basic info if not enough history
            # (5, 6) shape
            history_features = np.zeros((5, 6))
            # Fallback: fill with current grid/team info vaguely 
            # to avoid pure zero vectors if possible, or just leave as zeros.
            # Leaving as zeros is safer for "no history".
        else:
            # [GridPosition, TeamID, EventID, Position, CompoundID, TyreLife]
            history_features = driver_history[['GridPosition', 'TeamID', 'EventID', 'Position', 'CompoundID', 'TyreLife']].values.copy()
            
            # Normalize
            history_features = history_features.astype(float)
            history_features[:, 0] /= 20.0
            history_features[:, 1] /= max_team_id
            history_features[:, 2] /= max_event_id
            history_features[:, 3] /= 21.0
            history_features[:, 4] /= max_comp_id
            history_features[:, 5] /= 100.0
            
        # 3. Prepare Tensors
        # Driver/Team IDs: (1)
        # History: (1, 5, 6)
        # Grid: (1, 1) normalized
        
        t_d_id = torch.tensor([driver_id]).long().to(device)
        t_t_id = torch.tensor([team_id]).long().to(device)
        t_hist = torch.from_numpy(history_features).float().unsqueeze(0).to(device)
        
        # Normalize current grid pos
        norm_grid = float(grid) / 20.0
        t_grid = torch.tensor([[norm_grid]]).float().to(device)
        
        # 4. Predict
        with torch.no_grad():
            pred_norm = model(t_d_id, t_t_id, t_hist, t_grid).item()
            pred_pos = pred_norm * 21.0
            
            predictions.append({
                'Driver': abbr,
                'Grid': grid,
                'PredictedPosition': pred_pos
            })
            
    # Sort and Display
    pred_df = pd.DataFrame(predictions).sort_values('PredictedPosition')
    pred_df['PredictedRank'] = range(1, len(pred_df) + 1)
    
    print(f"\nPredictions for {year} {session.event['EventName']}:")
    print(pred_df[['PredictedRank', 'Driver', 'Grid', 'PredictedPosition']].to_string(index=False))
    
    return pred_df

if __name__ == "__main__":
    # Test
    predict_race(2024, 1)
