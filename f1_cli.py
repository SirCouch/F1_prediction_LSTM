import argparse
import os
import subprocess
import pandas as pd
import torch
from data_loader import fetch_race_data, preprocess_data, create_sequences
from train import train_model
from predict import predict_race
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser(description="F1 Race Outcome Predictor CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Update command
    update_parser = subparsers.add_parser("update", help="Update dataset and retrain model")
    update_parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024], help="Years to fetch data for")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict outcome for a specific race")
    predict_parser.add_argument("--year", type=int, required=True, help="Year of the race")
    predict_parser.add_argument("--round", type=int, required=True, help="Round number of the race")

    # Status command
    subparsers.add_parser("status", help="Check model and dataset status")

    args = parser.parse_args()

    if args.command == "update":
        print(f"Updating data for years: {args.years}")
        data = fetch_race_data(args.years)
        processed_data = preprocess_data(data)
        
        # New split return values
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

        print(f"Saved {len(y)} sequences. Starting training...")
        train_model()

    elif args.command == "predict":
        print(f"Predicting results for {args.year} Round {args.round}...")
        predict_race(args.year, args.round)

    elif args.command == "status":
        if os.path.exists('f1_lstm_model.pth'):
            print("Model: f1_lstm_model.pth found.")
        else:
            print("Model: Not found. Run 'update' first.")
        
        if os.path.exists('processed_f1_data.csv'):
            df = pd.read_csv('processed_f1_data.csv')
            print(f"Dataset: {len(df)} records across {df['Year'].nunique()} seasons.")
        else:
            print("Dataset: Not found.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
