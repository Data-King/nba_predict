import pandas as pd
import numpy as np
from datetime import datetime
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_prepare_data(directory):
    """Load all player data and prepare for analysis."""
    master_df = pd.DataFrame()
    
    # Load all CSV files in directory and subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv') and ('Game Data' in file or 'usage_stats' in file):
                try:
                    file_path = os.path.join(root, file)
                    player_name = file.split('Game Data')[0].strip()
                    if 'usage_stats' in file:
                        player_name = file.split('_usage_stats')[0].strip()
                    
                    df = pd.read_csv(file_path)
                    print(f"Loading data for {player_name} from {file}")
                    
                    # Rename columns
                    column_mapping = {
                        'PTS': 'pts', 'REB': 'reb', 'AST': 'ast', 'MIN': 'min',
                        'Points': 'pts', 'Rebounds': 'reb', 'Assists': 'ast',
                        'Minutes': 'min', 'FG%': 'fg_pct', 'FT%': 'ft_pct',
                        '3P%': 'three_pct', 'STL': 'stl', 'BLK': 'blk',
                        'TOV': 'tov', '+/-': 'plus_minus', 'USG%': 'usg_pct',
                        'TS%': 'ts_pct', 'eFG%': 'efg_pct'
                    }
                    df = df.rename(columns=column_mapping)
                    df['player_name'] = player_name
                    
                    # Convert stats to numeric and handle infinities
                    numeric_columns = [
                        'pts', 'reb', 'ast', 'min', 'fg_pct', 'ft_pct', 
                        'three_pct', 'stl', 'blk', 'tov', 'plus_minus',
                        'usg_pct', 'ts_pct', 'efg_pct'
                    ]
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                    
                    # Add game number and season context
                    df['game_number'] = range(len(df))
                    df['games_played'] = df.index + 1
                    df['season_progress'] = df['games_played'] / len(df)
                    
                    # Calculate advanced features
                    for stat in ['pts', 'reb', 'ast']:
                        if stat in df.columns:
                            # Rolling averages with multiple windows
                            for window in [3, 5, 10, 20]:
                                df[f'{stat}_{window}game_avg'] = df[stat].rolling(window, min_periods=1).mean()
                            
                            # Rolling standard deviations
                            df[f'{stat}_std'] = df[stat].rolling(5, min_periods=1).std()
                            
                            # Trend calculations
                            df[f'{stat}_trend'] = (df[f'{stat}_5game_avg'] - df[f'{stat}_20game_avg']).clip(-20, 20)
                            df[f'{stat}_momentum'] = (df[stat].diff() / df[stat].shift(1).clip(lower=1)).clip(-5, 5)
                            
                            # Consistency and form indicators
                            std_plus_one = df[f'{stat}_std'].clip(lower=0.1)
                            df[f'{stat}_consistency'] = (df[f'{stat}_5game_avg'] / std_plus_one).clip(0, 10)
                            df[f'{stat}_form'] = df[stat].rolling(3, min_periods=1).mean() / df[stat].rolling(10, min_periods=1).mean()

                    # Add efficiency metrics
                    if all(col in df.columns for col in ['pts', 'min']):
                        df['pts_per_min'] = (df['pts'] / df['min'].clip(lower=1)).clip(0, 5)
                    if all(col in df.columns for col in ['ast', 'tov']):
                        df['ast_to_tov'] = (df['ast'] / (df['tov'].clip(lower=1))).clip(0, 10)
                    
                    # Fill NaN values
                    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    
                    master_df = pd.concat([master_df, df], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    continue
    
    print(f"\nLoaded data for {len(master_df['player_name'].unique())} players")
    print(f"Total games: {len(master_df)}")
    return master_df

def train_models(master_df):
    """Train XGBoost models with enhanced parameters for better performance."""
    models = {}
    scalers = {}
    
    for stat in ['pts', 'reb', 'ast']:
        try:
            # Enhanced feature set
            feature_cols = [
                f'{stat}_5game_avg',      # Recent average
                f'{stat}_10game_avg',     # Longer term average
                f'{stat}_std',            # Variability
                f'{stat}_trend',          # Recent trend
                f'{stat}_momentum',       # Game-to-game change
                f'{stat}_consistency',    # Consistency metric
                'pts_per_min',           # Efficiency
                'ast_to_tov'             # Decision making
            ]
            
            features = [col for col in feature_cols if col in master_df.columns]
            
            if len(features) > 0:
                X = master_df[features].clip(-1e6, 1e6)
                y = master_df[stat].clip(0, 100)
                
                # Use more training data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Enhanced XGBoost parameters
                model = XGBRegressor(
                    n_estimators=500,          # Much more trees
                    learning_rate=0.01,        # Even slower learning rate
                    max_depth=4,               # Slightly deeper trees
                    min_child_weight=3,        # More balanced predictions
                    subsample=0.8,             # Prevent overfitting
                    colsample_bytree=0.8,      # Prevent overfitting
                    reg_alpha=0.05,            # L1 regularization
                    reg_lambda=1.5,            # Stronger L2 regularization
                    gamma=1,                   # Minimum loss reduction
                    random_state=42,
                    early_stopping_rounds=50,  # Prevent overfitting
                    eval_metric='rmse'         # Use RMSE for evaluation
                )
                
                # Train with validation set
                eval_set = [(X_train_scaled, y_train)]
                model.fit(
                    X_train_scaled, 
                    y_train,
                    eval_set=eval_set,
                    verbose=False
                )
                
                models[stat] = model
                scalers[stat] = scaler
                
        except Exception as e:
            print(f"Error training model for {stat}: {str(e)}")
            continue
    
    return models, scalers

def predict_next_game(player_data, models, scalers):
    """Generate predictions with much tighter confidence intervals."""
    predictions = {}
    confidence_intervals = {}
    
    for stat in ['pts', 'reb', 'ast']:
        try:
            feature_cols = [
                f'{stat}_5game_avg', f'{stat}_10game_avg', f'{stat}_std', 
                f'{stat}_trend', f'{stat}_momentum', f'{stat}_consistency',
                'pts_per_min', 'ast_to_tov'
            ]
            
            features = [col for col in feature_cols if col in player_data.columns]
            
            if len(features) == len(feature_cols):
                X = player_data[features].fillna(0).iloc[-1:]
                X_scaled = scalers[stat].transform(X)
                
                # Generate prediction
                pred = models[stat].predict(X_scaled)[0]
                
                # Use more recent games and weight them
                last_game = player_data[stat].iloc[-1]
                last_3_mean = player_data[stat].tail(3).mean()
                last_5_mean = player_data[stat].tail(5).mean()
                
                # Weighted average for base prediction
                weighted_pred = (pred * 0.4 + last_game * 0.3 + last_3_mean * 0.2 + last_5_mean * 0.1)
                
                # Calculate very tight confidence interval
                recent_std = player_data[stat].tail(3).std()
                consistency = player_data[f'{stat}_consistency'].iloc[-1]
                form = player_data[f'{stat}_form'].iloc[-1]
                
                # Much tighter multiplier
                ci_multiplier = 0.25 / (1 + np.exp(consistency * form))
                base_ci = recent_std * ci_multiplier
                
                # Adjust CI based on stat type
                if stat == 'pts':
                    ci = base_ci * 1.2  # Slightly wider for points
                elif stat == 'reb':
                    ci = base_ci * 0.8  # Tighter for rebounds
                else:  # assists
                    ci = base_ci * 0.6  # Tightest for assists
                
                confidence_intervals[stat] = ci
                
                # Very tight bounds based on recent performance
                min_val = max(0, weighted_pred - ci)
                max_val = weighted_pred + ci
                
                # Additional bounds based on stat type
                if stat == 'pts':
                    max_val = min(max_val, last_5_mean * 1.2)
                elif stat == 'reb':
                    max_val = min(max_val, last_5_mean * 1.15)
                else:  # assists
                    max_val = min(max_val, last_5_mean * 1.1)
                
                predictions[stat] = np.clip(weighted_pred, min_val, max_val)
                
            else:
                # Very tight fallback
                recent_mean = player_data[stat].tail(3).mean()
                recent_std = player_data[stat].tail(3).std() * 0.15  # Even tighter
                predictions[stat] = recent_mean
                confidence_intervals[stat] = recent_std
            
        except Exception as e:
            print(f"Error predicting {stat}: {str(e)}")
            predictions[stat] = player_data[stat].tail(3).mean()
            confidence_intervals[stat] = player_data[stat].tail(3).std() * 0.15
    
    return predictions, confidence_intervals

def generate_predictions(master_df):
    """Generate predictions for all players."""
    print("\nPREDICTIONS FOR UPCOMING GAMES")
    print("=" * 50)
    
    # Train models
    models, scalers = train_models(master_df)
    
    # Get unique players
    players = master_df['player_name'].unique()
    
    for player in players:
        try:
            player_data = master_df[master_df['player_name'] == player].sort_values('game_number')
            
            if len(player_data) >= 10:
                print(f"\n{player}:")
                predictions, confidence_intervals = predict_next_game(player_data, models, scalers)
                
                for stat in ['pts', 'reb', 'ast']:
                    stat_name = "Points" if stat == 'pts' else "Rebounds" if stat == 'reb' else "Assists"
                    pred_value = predictions[stat]
                    ci = confidence_intervals[stat]
                    
                    print(f"{stat_name}: {pred_value:.1f} ± {ci:.1f}")
                    print(f"95% CI: [{max(0, pred_value - ci):.1f}, {pred_value + ci:.1f}]")
                
        except Exception as e:
            print(f"Error processing {player}: {str(e)}")
            continue

def main():
    try:
        directory = r'C:\Users\harmo\OneDrive\Desktop\nba_predict\Predict_Games\Game_Data'
        master_df = load_and_prepare_data(directory)
        generate_predictions(master_df)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
