import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

def load_and_prepare_data(directory):
    """Load and prepare data from all CSV files in the directory."""
    all_dfs = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            player_name = filename.split('Game Data')[0].strip()
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Clean column names
            df.columns = df.columns.str.strip().str.lower()
            df.columns = df.columns.str.replace(' ', '_')
            df.columns = df.columns.str.replace('3:00_pm', '3pm')
            
            # Add player name
            df['player_name'] = player_name
            
            # Add season column (example logic, adjust as needed)
            # Assuming the filename or a column contains season info
            # Example: filename = "2022_Game Data.csv"
            try:
                season_year = int(filename.split('_')[0])
                df['season'] = season_year
            except ValueError:
                print(f"Could not extract season from filename: {filename}")
                df['season'] = np.nan  # or some default value
            
            # Convert all columns except player_name
            for col in df.columns:
                if col != 'player_name':
                    try:
                        # Handle percentage columns
                        if '%' in str(df[col].iloc[0]):
                            df[col] = df[col].str.rstrip('%').astype(float) / 100
                        # Handle minutes column specifically
                        elif col == 'min':
                            df[col] = df[col].apply(lambda x: 
                                0 if pd.isna(x) or x == 'MIN' 
                                else int(str(x).split(':')[0]) if ':' in str(x)
                                else float(x) if str(x).replace('.','',1).isdigit()
                                else 0
                            )
                        # Handle time format (convert to minutes)
                        elif ':' in str(df[col].iloc[0]):
                            df[col] = df[col].apply(lambda x: 
                                int(str(x).split(':')[0]) if pd.notnull(x) and ':' in str(x)
                                else 0
                            )
                        # Handle other numeric columns
                        else:
                            df[col] = pd.to_numeric(
                                df[col].astype(str).str.replace(',', ''), 
                                errors='coerce'
                            ).fillna(0)
                    except Exception as e:
                        print(f"Error processing column {col}: {str(e)}")
                        continue
            
            all_dfs.append(df)
            print(f"Processed {filename}: {df.shape}")
    
    master_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nFinal master dataframe shape: {master_df.shape}")
    print("Columns:", master_df.columns.tolist())
    
    return master_df

def prepare_features(df, target_columns):
    """Prepare features for machine learning."""
    # Create label encoders for categorical columns
    label_encoders = {}
    
    # Encode player names
    le_player = LabelEncoder()
    df['player_name_encoded'] = le_player.fit_transform(df['player_name'])
    label_encoders['player_name'] = le_player
    
    # Find the matchup column (could be 'matchup', 'match_up', etc.)
    matchup_col = next((col for col in df.columns if 'match' in col.lower()), None)
    if matchup_col:
        le_matchup = LabelEncoder()
        df['matchup_encoded'] = le_matchup.fit_transform(df[matchup_col])
        label_encoders[matchup_col] = le_matchup
    
    # Get ALL numeric columns for features, including target columns
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print("\nFeature columns:", feature_columns)
    print("Target columns:", target_columns)
    
    return df, label_encoders, feature_columns

def train_models(df, target_columns, feature_columns, matchup_col):
    """Train models for each target variable using all historical data."""
    models = {}
    scalers = {}
    metrics = {}
    
    # Add career statistics for each target column
    for target in target_columns:
        # Career averages and statistics
        df[f'{target}_avg'] = df.groupby('player_name')[target].transform('mean')
        df[f'{target}_std'] = df.groupby('player_name')[target].transform('std').fillna(0)
        df[f'{target}_max'] = df.groupby('player_name')[target].transform('max')
        df[f'{target}_min'] = df.groupby('player_name')[target].transform('min')
        df[f'{target}_median'] = df.groupby('player_name')[target].transform('median')
    
    # Add total games played for each player
    df['total_games'] = df.groupby('player_name').cumcount() + 1
    
    # Add home/away feature
    if matchup_col:
        try:
            if not pd.api.types.is_numeric_dtype(df[matchup_col]):
                df['is_home'] = pd.Categorical(df[matchup_col]).codes % 2
            else:
                df['is_home'] = df[matchup_col] % 2
        except:
            print("Could not process matchup column for home/away feature")
            df['is_home'] = 0
    
    # Update feature columns with new features
    feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for target in target_columns:
        print(f"\nTraining model for {target}")
        
        X = df[feature_columns]
        y = df[target]
        
        # Split data with stratification on player_name_encoded
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=df['player_name_encoded']
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Optimize model parameters
        model = RandomForestRegressor(
            n_estimators=500,  # More trees
            max_depth=20,      # Deeper trees
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',  # Use sqrt of features for each split
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            warm_start=True    # Enable warm start for iterative fitting
        )
        
        # Iterative fitting to prevent overfitting
        for i in range(3):  # Fit multiple times with warm start
            model.fit(X_train_scaled, y_train)
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            print(f"Iteration {i+1} - Train R2: {train_score:.3f}, Test R2: {test_score:.3f}")
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        models[target] = model
        scalers[target] = scaler
        metrics[target] = {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape
        }
        
        # Feature importance with more detail
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nMetrics for {target}:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), feature_importance['importance'][:10])
        plt.xticks(range(10), feature_importance['feature'][:10], rotation=45, ha='right')
        plt.title(f'Top 10 Important Features for {target}')
        plt.tight_layout()
        plt.show()
    
    return models, scalers, metrics

def generate_predictions_for_upcoming_games(master_df, models, scalers, feature_columns):
    """Generate predictions for points, rebounds, and assists for upcoming games."""
    
    # Only predict these stats
    target_columns = ['pts', 'reb', 'ast']
    
    # Define upcoming games
    games = [
        {"player": "Jakob P", "game": "TOR vs DET", "home": 1},
        {"player": "Cade C", "game": "TOR vs DET", "home": 0},
        {"player": "Ja M", "game": "MEM vs MIN", "home": 1},
        {"player": "Jaren Jackson Jr", "game": "MEM vs MIN", "home": 1},
        {"player": "Jaden M", "game": "MEM vs MIN", "home": 0},
        {"player": "Rudy G", "game": "MEM vs MIN", "home": 0},
        {"player": "Bam A", "game": "MIA vs POR", "home": 1},
        {"player": "Tyler H", "game": "MIA vs POR", "home": 1},
        {"player": "Anfernee S", "game": "MIA vs POR", "home": 0},
        {"player": "Shaedon S", "game": "MIA vs POR", "home": 0}
    ]
    
    predictions = []
    
    for game in games:
        player = game['player']
        print(f"\nPredicting for {player}...")
        
        # Get player's historical data
        player_data = master_df[master_df['player_name'] == player].copy()
        
        if len(player_data) == 0:
            print(f"No data for {player}")
            continue
        
        try:
            # Calculate necessary features
            features = pd.DataFrame()
            for target in target_columns:
                features[f'{target}_avg'] = [player_data[target].mean()]
                features[f'{target}_std'] = [player_data[target].std()]
                features[f'{target}_max'] = [player_data[target].max()]
                features[f'{target}_min'] = [player_data[target].min()]
                features[f'{target}_median'] = [player_data[target].median()]
            
            # Add game context
            features['is_home'] = game['home']
            features['total_games'] = len(player_data)
            
            # Ensure all feature columns exist
            for col in feature_columns:
                if col not in features.columns:
                    features[col] = 0
            
            # Initialize prediction row
            pred_row = {'Player': player, 'Game': game['game']}
            
            # Generate predictions for each stat
            for target in target_columns:
                if target in models:
                    # Scale features
                    X = features[feature_columns]
                    X_scaled = scalers[target].transform(X)
                    
                    # Get prediction
                    prediction = models[target].predict(X_scaled)[0]
                    pred_row[target] = max(0, round(prediction, 1))
                    
                    print(f"{target}: {pred_row[target]}")
            
            predictions.append(pred_row)
            
        except Exception as e:
            print(f"Error for {player}: {str(e)}")
            continue
    
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        return predictions_df[['Player', 'Game', 'pts', 'reb', 'ast']]
    
    return pd.DataFrame()

def backtest_with_feature_importance(data, model, predictors, target_col='pts', start=1, step=1):
    """ Backtest function to get feature importance """
    all_predictions = []
    feature_importances = pd.DataFrame(index=predictors)

    # Ensure we have the required columns
    if 'season' not in data.columns:
        print("No season column found. Creating dummy seasons based on game order...")
        # Create more granular seasons (every 5 games is a new "season")
        data['season'] = data.index // 5  # Reduced from 10 to 5 for more seasons

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")

    seasons = sorted(data['season'].unique())
    if len(seasons) < 2:  # Need at least 2 seasons for train/test split
        print("Not enough distinct seasons, creating more granular seasons...")
        data['season'] = data.index // 3  # Even more granular seasons
        seasons = sorted(data['season'].unique())

    print(f"Processing {len(seasons)} seasons...")
    
    # Use a minimum amount of data for initial training
    min_train_size = len(data) // 4  # Use 25% of data as minimum training size
    
    for i in range(start, len(seasons)):
        season = seasons[i]
        print(f"Processing season {season}...")

        train = data[data['season'] < season]
        test = data[data['season'] == season]

        if len(train) < min_train_size:
            print(f"Skipping season {season} - insufficient training data")
            continue

        if len(test) == 0:
            print(f"Skipping season {season} - no test data")
            continue

        # Ensure all predictors exist in the data
        missing_predictors = [p for p in predictors if p not in train.columns]
        if missing_predictors:
            print(f"Warning: Missing predictors: {missing_predictors}")
            # Only use available predictors
            predictors = [p for p in predictors if p in train.columns]

        if not predictors:
            raise ValueError("No valid predictors available")

        try:
            # Convert to regression problem if using XGBClassifier
            if isinstance(model, XGBClassifier):
                print("Converting XGBClassifier to XGBRegressor for regression task...")
                from xgboost import XGBRegressor
                model = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )

            model.fit(train[predictors], train[target_col])
            preds = model.predict(test[predictors])
            preds = pd.Series(preds, index=test.index)
            
            combined = pd.DataFrame({
                'actual': test[target_col],
                'prediction': preds
            })
            
            all_predictions.append(combined)

            # Extract feature importance
            if hasattr(model, 'feature_importances_'):
                season_importance = pd.Series(model.feature_importances_, index=predictors)
            else:
                season_importance = pd.Series([0]*len(predictors), index=predictors)
            
            feature_importances[f'season_{season}'] = season_importance
            
        except Exception as e:
            print(f"Error processing season {season}: {str(e)}")
            continue

    if not all_predictions:
        raise ValueError("No predictions were generated during backtesting")

    # Calculate the mean importance across all seasons
    feature_importances['mean_importance'] = feature_importances.mean(axis=1)
    feature_importances = feature_importances.sort_values(by='mean_importance', ascending=False)
    
    return pd.concat(all_predictions), feature_importances

# Update main execution:
if __name__ == "__main__":
    directory = r'C:\Users\harmo\OneDrive\Desktop\nba_predict\Predict_Games\Game_Data'
    target_columns = ['pts', 'reb', 'ast']  # Only these three stats
    
    print("Loading and preparing data...")
    master_df = load_and_prepare_data(directory)
    if master_df.empty:
        print("No data loaded. Please check the data directory and files.")
    
    master_df, label_encoders, feature_columns = prepare_features(master_df, target_columns)
    if not feature_columns:
        print("No feature columns identified. Please check the data preparation step.")
    
    print("\nTraining models...")
    matchup_col = next((col for col in master_df.columns if 'match' in col.lower()), None)
    models, scalers, metrics = train_models(master_df, target_columns, feature_columns, matchup_col)
    if not models:
        print("Model training failed. Please check the training process.")
    
    print("\n=== GENERATING PREDICTIONS FOR UPCOMING GAMES ===")
    predictions_df = generate_predictions_for_upcoming_games(
        master_df=master_df,
        models=models,
        scalers=scalers,
        feature_columns=feature_columns
    )
    
    if not predictions_df.empty:
        print("\n=== UPCOMING GAMES PREDICTIONS ===")
        print(predictions_df.round(1))
        predictions_df.to_csv('upcoming_games_predictions.csv', index=False)
        print("\nPredictions saved to 'upcoming_games_predictions.csv'")
    else:
        print("Failed to generate predictions!")

    # Example usage of backtest_with_feature_importance
    print("\n=== BACKTESTING WITH FEATURE IMPORTANCE ===")
    try:
        from xgboost import XGBRegressor  # Change to XGBRegressor
        
        model = XGBRegressor(  # Use XGBRegressor instead of XGBClassifier
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Remove any unwanted columns from predictors
        predictors = [col for col in feature_columns 
                     if col not in ['season', 'target', 'player_name', 'matchup']]
        
        # Try each target column
        for target_col in ['pts', 'reb', 'ast']:
            print(f"\nBacktesting for {target_col.upper()}:")
            predictions, feature_importances = backtest_with_feature_importance(
                data=master_df,
                model=model,
                predictors=predictors,
                target_col=target_col,
                start=1  # Start from the first season
            )
            
            print(f"\nPredictions Summary for {target_col.upper()}:")
            print(predictions.describe())
            print(f"\nTop 10 Most Important Features for {target_col.upper()}:")
            print(feature_importances.head(10))
            
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")


