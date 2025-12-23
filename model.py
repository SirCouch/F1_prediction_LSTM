import torch
import torch.nn as nn
import torch.nn.functional as F

class F1Predictor(nn.Module):
    def __init__(self, num_drivers, num_teams, hidden_size=64):
        super(F1Predictor, self).__init__()
        
        # 1. Embeddings for static IDs
        # Embedding dim 10 is arbitrary but sufficient for ~20-100 categories
        self.driver_emb = nn.Embedding(num_drivers, 10)
        self.team_emb = nn.Embedding(num_teams, 10)
        
        # 2. LSTM for "Driver Form" (Sequence of last 5 races)
        # Input features for history: 
        # [GridPos, TeamID(mapped?), EventID(mapped?), Position, CompoundID, TyreLife] -> 6 features
        # Actually, let's simplify the history input to the core numeric ones or keep it as is.
        # The previous model had 6 input features. Let's keep using that for the history sequence
        # to capture as much context as possible.
        self.lstm = nn.LSTM(input_size=6, hidden_size=hidden_size, batch_first=True, dropout=0.2, num_layers=2)
        
        # 3. Dense layer for Current Race info
        # Inputs: DriverEmb(10) + TeamEmb(10) + GridPos(1) = 21 features
        self.fc_current = nn.Linear(10 + 10 + 1, 32)
        
        # 4. Final Prediction Head
        # Concatenates LSTM output (hidden_size) + Dense output (32)
        self.fc_final = nn.Sequential(
            nn.Linear(hidden_size + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, driver_id, team_id, past_races_seq, current_grid_pos):
        # Embeddings
        # driver_id: (Batch)
        d_emb = self.driver_emb(driver_id) # (Batch, 10)
        t_emb = self.team_emb(team_id)     # (Batch, 10)
        
        # Process History with LSTM
        # past_races_seq: (Batch, Seq_Len, Features)
        # We only care about the final hidden state representation of the history
        out, _ = self.lstm(past_races_seq)
        lstm_out = out[:, -1, :] # Take the output of the last time step: (Batch, hidden_size)
        
        # Process Current Race Context
        # current_grid_pos: (Batch, 1)
        current_features = torch.cat((d_emb, t_emb, current_grid_pos), dim=1)
        dense_out = F.relu(self.fc_current(current_features))
        
        # Combine
        combined = torch.cat((lstm_out, dense_out), dim=1)
        
        # Predict
        return self.fc_final(combined)
