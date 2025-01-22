import pandas as pd
import numpy as np
from datetime import datetime
import os
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

def load_and_prepare_data(directory):
    """Load all player data with enhanced analytics."""
    master_df = pd.DataFrame()
    
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
                    
                    # Enhanced column mapping with advanced stats
                    column_mapping = {
                        'PTS': 'pts', 'REB': 'reb', 'AST': 'ast', 'MIN': 'min',
                        'FGA': 'fga', 'FGM': 'fgm', '3PA': 'three_pa', '3PM': 'three_pm',
                        'FTA': 'fta', 'FTM': 'ftm', 'OREB': 'oreb', 'DREB': 'dreb',
                        'STL': 'stl', 'BLK': 'blk', 'TOV': 'tov', '+/-': 'plus_minus',
                        'USG%': 'usg_pct', 'TS%': 'ts_pct', 'eFG%': 'efg_pct',
                        'AST%': 'ast_pct', 'REB%': 'reb_pct', 'BLK%': 'blk_pct',
                        'STL%': 'stl_pct', 'TOV%': 'tov_pct', 'PACE': 'pace'
                    }
                    df = df.rename(columns=column_mapping)
                    df['player_name'] = player_name
                    
                    # Better handling of numeric conversions and infinities
                    for col in df.columns:
                        if col not in ['player_name', 'date', 'match_up']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            # Replace infinities and clip extreme values
                            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                            if col in ['pts', 'reb', 'ast']:
                                df[col] = df[col].clip(0, 100)
                            else:
                                df[col] = df[col].clip(-1e3, 1e3)
                    
                    # Safer calculation of advanced metrics
                    if all(col in df.columns for col in ['fga', 'fta', 'pts']):
                        denominator = 2 * (df['fga'] + 0.44 * df['fta']).clip(lower=0.1)
                        df['true_shooting'] = (df['pts'] / denominator).clip(0, 1)
                    
                    # Safer per minute calculations
                    per_minute_stats = ['pts', 'reb', 'ast', 'stl', 'blk', 'tov']
                    for stat in per_minute_stats:
                        if stat in df.columns and 'min' in df.columns:
                            df[f'{stat}_per_min'] = (df[stat] / df['min'].clip(lower=1)).clip(0, 10)
                    
                    # Safer rolling calculations
                    windows = [3, 5, 7, 10, 15, 20, 30, 40, 50]
                    stats_to_analyze = [
                        'pts', 'reb', 'ast', 'fg_pct', 'three_pct', 'true_shooting',
                        'usg_pct', 'plus_minus', 'min'
                    ]
                    
                    for stat in stats_to_analyze:
                        if stat in df.columns:
                            for window in windows:
                                df[f'{stat}_{window}g_avg'] = df[stat].rolling(window, min_periods=1).mean()
                                df[f'{stat}_{window}g_std'] = df[stat].rolling(window, min_periods=1).std()
                                # Clip the averages and stds
                                df[f'{stat}_{window}g_avg'] = df[f'{stat}_{window}g_avg'].clip(-1e3, 1e3)
                                df[f'{stat}_{window}g_std'] = df[f'{stat}_{window}g_std'].clip(0, 100)
                            
                            # Exponential moving averages
                            df[f'{stat}_ema_fast'] = df[stat].ewm(span=5).mean()
                            df[f'{stat}_ema_slow'] = df[stat].ewm(span=20).mean()
                            
                            # Momentum indicators
                            df[f'{stat}_momentum'] = df[f'{stat}_ema_fast'] - df[f'{stat}_ema_slow']
                            df[f'{stat}_roc'] = df[stat].pct_change(3)  # Rate of change
                            
                            # Consistency metrics
                            df[f'{stat}_consistency'] = 1 - (df[f'{stat}_5g_std'] / df[f'{stat}_5g_avg'].clip(lower=0.1))
                            
                            # Performance vs average
                            df[f'{stat}_vs_avg'] = df[stat] / df[f'{stat}_20g_avg'].clip(lower=0.1)
                        
                    # Game context features
                    df['game_number'] = range(len(df))
                    df['rest_days'] = df['date'].diff().dt.days if 'date' in df.columns else 1
                    
                    # Add seasonal trend analysis
                    df['season_progress'] = df.index / len(df)
                    df['form_trend'] = df[stat].rolling(10).mean() / df[stat].rolling(30).mean()
                    
                    # Add peak performance calculations
                    for stat in stats_to_analyze:
                        if stat in df.columns:
                            # Calculate peak performance over different windows
                            for window in [5, 10, 15]:
                                df[f'{stat}_{window}g_max'] = df[stat].rolling(window, min_periods=1).max()
                                df[f'{stat}_{window}g_min'] = df[stat].rolling(window, min_periods=1).min()
                                df[f'{stat}_{window}g_range'] = df[f'{stat}_{window}g_max'] - df[f'{stat}_{window}g_min']
                            
                            # Hot/Cold streaks
                            df[f'{stat}_hot_streak'] = (df[stat] > df[f'{stat}_15g_avg'] * 1.2).rolling(3).sum()
                            df[f'{stat}_cold_streak'] = (df[stat] < df[f'{stat}_15g_avg'] * 0.8).rolling(3).sum()
                            
                            # Trend strength
                            df[f'{stat}_trend_strength'] = abs(df[f'{stat}_momentum']) / df[f'{stat}_15g_std'].clip(lower=0.1)
                    
                    master_df = pd.concat([master_df, df], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    continue
    
    # Fill remaining NaN values
    master_df = master_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"\nLoaded data for {len(master_df['player_name'].unique())} players")
    print(f"Total games: {len(master_df)}")
    return master_df

def train_models(master_df):
    """Train maximum aggressive models."""
    models = {}
    scalers = {}
    
    for stat in ['pts', 'reb', 'ast']:
        try:
            # Enhanced feature set
            base_features = [
                # Recent performance
                f'{stat}_3g_avg', f'{stat}_5g_avg', f'{stat}_7g_avg',
                f'{stat}_10g_avg', f'{stat}_15g_avg',
                
                # Peak performance indicators
                f'{stat}_5g_max', f'{stat}_10g_max', f'{stat}_15g_max',
                
                # Advanced metrics
                f'{stat}_momentum', f'{stat}_consistency', f'{stat}_per_min',
                f'{stat}_ema_fast', f'{stat}_ema_slow', f'{stat}_vs_avg',
                f'{stat}_roc', 'offensive_rating', 'usg_pct',
                
                # Form and context
                'form_trend', 'season_progress', 'game_number', 
                'rest_days', 'plus_minus', 'pace'
            ]
            
            # Ultra aggressive parameters
            models[stat] = {
                'xgb': XGBRegressor(
                    n_estimators=25000,        # Maximum trees
                    learning_rate=0.3,         # Extremely high learning rate
                    max_depth=45,              # Maximum depth
                    subsample=1.0,
                    colsample_bytree=1.0,
                    min_child_weight=1,
                    gamma=0.000000001,         # Ultra aggressive splitting
                    reg_alpha=0.0000000001,    # Minimal regularization
                    reg_lambda=0.0000000001,
                    tree_method='hist',
                    grow_policy='lossguide',
                    random_state=42
                ),
                'lgb': LGBMRegressor(
                    n_estimators=25000,
                    learning_rate=0.3,
                    num_leaves=131071,         # Absolute maximum
                    subsample=1.0,
                    colsample_bytree=1.0,
                    min_child_samples=1,
                    reg_alpha=0.0000000001,
                    reg_lambda=0.0000000001,
                    boosting_type='goss',
                    random_state=42
                )
            }
            
            # Maximum aggressive stat-specific tuning
            if stat == 'pts':
                models[stat]['xgb'].set_params(
                    learning_rate=0.35,        # Maximum aggression for points
                    max_depth=50
                )
                models[stat]['lgb'].set_params(
                    learning_rate=0.35,
                    num_leaves=262143          # Absolute maximum
                )
            elif stat == 'reb':
                models[stat]['xgb'].set_params(
                    learning_rate=0.32,
                    max_depth=48
                )
            elif stat == 'ast':
                models[stat]['xgb'].set_params(
                    learning_rate=0.3,
                    max_depth=45
                )

            features = [col for col in base_features if col in master_df.columns]
            
            if len(features) > 0:
                X = master_df[features].clip(-1e6, 1e6)
                y = master_df[stat].clip(0, 100)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # Single, consistent model definition
                models[stat] = {
                    'xgb': XGBRegressor(
                        n_estimators=25000,
                        learning_rate=0.3,
                        max_depth=45,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        min_child_weight=1,
                        gamma=0.000000001,
                        reg_alpha=0.0000000001,
                        reg_lambda=0.0000000001,
                        tree_method='hist',
                        grow_policy='lossguide',
                        random_state=42
                    ),
                    'lgb': LGBMRegressor(
                        n_estimators=25000,
                        learning_rate=0.3,
                        num_leaves=131071,
                        subsample=1.0,
                        colsample_bytree=1.0,
                        min_child_samples=1,
                        reg_alpha=0.0000000001,
                        reg_lambda=0.0000000001,
                        boosting_type='goss',
                        random_state=42
                    )
                }
                
                # Train each model
                for name, model in models[stat].items():
                    model.fit(X_train_scaled, y_train)
                
                scalers[stat] = scaler
                
        except Exception as e:
            print(f"Error training {stat} models: {str(e)}")
            
    return models, scalers

def prepare_features(player_data, stat):
    """Prepare features for prediction with safe value handling."""
    # Base features matching train_models exactly
    base_features = [
        # Very recent performance
        f'{stat}_3g_avg', f'{stat}_5g_avg', f'{stat}_7g_avg',
        f'{stat}_10g_avg', f'{stat}_15g_avg',
        
        # Peak performance indicators
        f'{stat}_5g_max', f'{stat}_10g_max', f'{stat}_15g_max',
        
        # Advanced metrics
        f'{stat}_momentum', f'{stat}_consistency', f'{stat}_per_min',
        f'{stat}_ema_fast', f'{stat}_ema_slow', f'{stat}_vs_avg',
        f'{stat}_roc', 'offensive_rating', 'usg_pct',
        
        # Form and context
        'form_trend', 'season_progress', 'game_number', 
        'rest_days', 'plus_minus', 'pace'
    ]
    
    # Get features and handle missing/infinite values
    features = [col for col in base_features if col in player_data.columns]
    X = player_data[features].fillna(0)
    
    # Clip values to prevent infinities
    for col in X.columns:
        if 'pct' in col or 'ratio' in col:
            X[col] = X[col].clip(-1, 1)
        elif 'avg' in col or 'mean' in col:
            X[col] = X[col].clip(-1e3, 1e3)
        elif 'std' in col:
            X[col] = X[col].clip(0, 100)
        else:
            X[col] = X[col].clip(-1e3, 1e3)
    
    return X.iloc[-1:]

def predict_next_game(player_data, models, scalers):
    """Generate ultra-aggressive predictions."""
    predictions = {}
    confidence_intervals = {}
    
    # Higher minimum thresholds
    min_thresholds = {
        'pts': 25.0,  # Much higher minimum points
        'reb': 10.0,  # Much higher minimum rebounds
        'ast': 6.0    # Much higher minimum assists
    }
    
    for stat in ['pts', 'reb', 'ast']:
        try:
            X = prepare_features(player_data, stat)
            X_scaled = scalers[stat].transform(X)
            
            # Get predictions and ensure they're above zero
            model_predictions = {
                name: max(model.predict(X_scaled)[0], min_thresholds[stat])
                for name, model in models[stat].items()
            }
            
            # Calculate recent performance metrics
            last_game = max(player_data[stat].iloc[-1], min_thresholds[stat])
            last_3_mean = max(player_data[stat].tail(3).mean(), min_thresholds[stat])
            last_5_mean = max(player_data[stat].tail(5).mean(), min_thresholds[stat])
            last_10_mean = max(player_data[stat].tail(10).mean(), min_thresholds[stat])
            recent_max = max(player_data[stat].tail(5).max(), min_thresholds[stat])
            recent_avg = max(player_data[stat].tail(5).mean(), min_thresholds[stat])
            recent_std = player_data[stat].tail(5).std()
            
            # Dynamic weights based on recent performance stability
            recent_stability = 1 / (1 + player_data[f'{stat}_5g_std'].iloc[-1])
            long_term_stability = 1 / (1 + player_data[f'{stat}_30g_std'].iloc[-1])
            
            # Aggressive weights favoring higher predictions
            if recent_stability > long_term_stability:
                weights = {
                    'xgb': 0.35, 'lgb': 0.35,
                    'rf': 0.15, 'et': 0.15
                }
            else:
                weights = {
                    'xgb': 0.30, 'lgb': 0.30,
                    'rf': 0.20, 'et': 0.20
                }
            
            # Ensure weighted_pred is above minimum
            weighted_pred = max(
                sum(pred * weights[name] for name, pred in model_predictions.items()),
                min_thresholds[stat]
            )
            
            # Maximum aggressive weighting
            final_pred = (
                weighted_pred * 0.02 +          # Minimal model weight
                last_game * 0.55 +             # Maximum recent game weight
                last_3_mean * 0.25 +           # Recent form
                recent_max * 0.18              # Peak potential
            )
            
            # Ultra aggressive boosting
            if final_pred < recent_avg:
                boost_factor = min(4.0, recent_avg / final_pred)  # Up to 300% boost
                final_pred = final_pred * boost_factor
            
            # Maximum role-based adjustments
            if stat == 'pts':
                if recent_max > 55:  # Super ultra elite scorer
                    final_pred *= 1.8
                elif recent_max > 45:  # Ultra elite scorer
                    final_pred *= 1.7
                elif recent_max > 35:  # Super elite scorer
                    final_pred *= 1.6
                elif recent_max > 25:  # Elite scorer
                    final_pred *= 1.5
            elif stat == 'reb':
                if recent_max > 22:  # Super ultra elite rebounder
                    final_pred *= 1.7
                elif recent_max > 18:  # Ultra elite rebounder
                    final_pred *= 1.6
                elif recent_max > 14:  # Elite rebounder
                    final_pred *= 1.5
            elif stat == 'ast':
                if recent_max > 18:  # Super ultra elite playmaker
                    final_pred *= 1.7
                elif recent_max > 14:  # Ultra elite playmaker
                    final_pred *= 1.6
                elif recent_max > 10:  # Elite playmaker
                    final_pred *= 1.5
            
            # Enhanced hot streak bonus
            if last_3_mean > last_10_mean * 1.1:  # If really hot
                final_pred *= 1.3  # 30% boost for hot streak
            elif last_3_mean > last_10_mean:
                final_pred *= 1.2  # 20% boost for mild hot streak
            
            # Ensure prediction is above minimum threshold
            final_pred = max(final_pred, min_thresholds[stat])
            
            predictions[stat] = final_pred
            confidence_intervals[stat] = recent_std * 0.4  # Even tighter confidence interval
            
        except Exception as e:
            print(f"Error predicting {stat}: {str(e)}")
            continue
            
    return predictions, confidence_intervals

def generate_predictions(master_df):
    """Generate predictions for all players with individual model breakdowns."""
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
                    print(f"\n{stat_name}:")
                    
                    # Get individual model predictions
                    X = prepare_features(player_data, stat)
                    X_scaled = scalers[stat].transform(X)
                    
                    print("Model Predictions:")
                    for name, model in models[stat].items():
                        pred = model.predict(X_scaled)[0]
                        print(f"  {name.upper()}: {pred:.1f}")
                    
                    # Ensemble prediction
                    pred_value = predictions[stat]
                    ci = confidence_intervals[stat]
                    print(f"ENSEMBLE: {pred_value:.1f} ± {ci:.1f}")
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
