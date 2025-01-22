import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel
import logging
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_column_info(data):
    """Print detailed information about available columns"""
    logging.info("\nColumn Information:")
    
    # Print all columns
    logging.info("\nAll columns:")
    logging.info(data.columns.tolist())
    
    # Print columns related to specific stats
    stat_prefixes = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG', 'FT', 'MP', 'MIN']
    for prefix in stat_prefixes:
        related_cols = [col for col in data.columns if prefix in col]
        if related_cols:
            logging.info(f"\n{prefix}-related columns:")
            logging.info(related_cols)
    
    # Print numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    logging.info("\nNumeric columns:")
    logging.info(numeric_cols.tolist())

def load_player_data():
    """Load and combine all player statistics"""
    try:
        # Load all stats
        master_base = pd.read_csv(r"C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Predict25\master_base_stats.csv")
        master_advanced = pd.read_csv(r"C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Predict25\master_advanced_stats.csv")
        master_misc = pd.read_csv(r"C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Predict25\master_misc_stats.csv")
        
        # Print initial data info
        logging.info("\nMaster Base Stats columns:")
        logging.info(master_base.columns.tolist())
        if 'Player' in master_base.columns:
            logging.info("\nSample players from base stats:")
            logging.info(master_base['Player'].head())
        
        # Combine all stats
        player_data = pd.concat([master_base, master_advanced, master_misc], axis=1)
        player_data = player_data.loc[:,~player_data.columns.duplicated()]
        
        # Verify we have player names
        if 'Player' not in player_data.columns:
            # Try to find alternative player name column
            possible_columns = ['PLAYER', 'player_name', 'Name', 'PLAYER_NAME']
            for col in possible_columns:
                if col in player_data.columns:
                    player_data['Player'] = player_data[col]
                    break
        
        if 'Player' not in player_data.columns:
            logging.error("No player name column found in the dataset!")
            logging.error("Available columns:")
            logging.error(player_data.columns.tolist())
            raise KeyError("No player name column found")
        
        # Create interaction features
        player_data = create_interaction_features(player_data)
        
        # Handle missing values
        player_data = handle_missing_values(player_data)
        
        return player_data
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def create_interaction_features(data):
    """Create interaction features between important statistics"""
    try:
        # First, let's print the available columns
        logging.info("\nAvailable columns for feature creation:")
        logging.info(data.columns.tolist())
        
        # Create meaningful basketball-specific features
        # Minutes played might be 'MIN' or 'Minutes' instead of 'MP'
        minutes_col = next(col for col in data.columns if col in ['MP', 'MIN', 'Minutes'])
        
        # Create per-minute stats if minutes column exists
        if minutes_col:
            data['Points_per_Minute'] = data['PTS'] / (data[minutes_col] + 1e-6)
            if 'TRB' in data.columns:
                data['Rebounds_per_Minute'] = data['TRB'] / (data[minutes_col] + 1e-6)
            if 'AST' in data.columns:
                data['Assists_per_Minute'] = data['AST'] / (data[minutes_col] + 1e-6)
        
        # Efficiency metrics
        if 'FGA' in data.columns:
            data['Scoring_Efficiency'] = data['PTS'] / (data['FGA'] + 1e-6)
        
        if 'USG%' in data.columns:
            data['Usage_Efficiency'] = data['PTS'] / (data['USG%'] + 1e-6)
        
        # Advanced combinations (only create if all required columns exist)
        if all(col in data.columns for col in ['STL', 'BLK', 'TOV']):
            data['Defense_Score'] = data['STL'] + data['BLK'] - data['TOV']
        
        if all(col in data.columns for col in ['PTS', 'AST']):
            data['Offensive_Impact'] = data['PTS'] + data['AST'] * 2
            
            if 'Defense_Score' in data.columns:
                data['Overall_Impact'] = data['Defense_Score'] + data['Offensive_Impact']
        
        # Additional efficiency metrics
        if 'FG%' in data.columns and 'FT%' in data.columns:
            data['Shooting_Efficiency'] = (data['FG%'] + data['FT%']) / 2
        
        if 'ORB' in data.columns and 'DRB' in data.columns:
            data['Rebounding_Ratio'] = data['ORB'] / (data['DRB'] + 1e-6)
        
        # Print created features
        new_features = [col for col in data.columns if col not in set(data.columns) - set([
            'Points_per_Minute', 'Rebounds_per_Minute', 'Assists_per_Minute',
            'Scoring_Efficiency', 'Usage_Efficiency', 'Defense_Score',
            'Offensive_Impact', 'Overall_Impact', 'Shooting_Efficiency',
            'Rebounding_Ratio'
        ])]
        logging.info("\nCreated features:")
        logging.info(new_features)
        
        return data
        
    except Exception as e:
        logging.error(f"Error in create_interaction_features: {str(e)}")
        # If there's an error, return the original data without modifications
        return data

def handle_missing_values(data):
    """Handle missing values with sophisticated approach"""
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    # For each numeric column
    for col in numeric_cols:
        # Calculate the median for non-zero values
        median_non_zero = data[data[col] != 0][col].median()
        
        # Fill NaN values
        if data[col].isna().any():
            # If it's a percentage, fill with median
            if '%' in col:
                data[col] = data[col].fillna(data[col].median())
            # For counting stats, fill with 0
            else:
                data[col] = data[col].fillna(0)
        
        # Replace 0s with median for percentage columns
        if '%' in col:
            data[col] = data[col].replace(0, median_non_zero)
    
    return data

def create_advanced_features(data):
    """Create comprehensive basketball-specific features"""
    try:
        # Scoring Efficiency Metrics
        if all(col in data.columns for col in ['PTS', 'FGA', 'FTA']):
            data['True_Shooting'] = data['PTS'] / (2 * (data['FGA'] + 0.44 * data['FTA']))
            data['Points_per_Shot'] = data['PTS'] / (data['FGA'] + 1e-6)
            
        # Advanced Scoring Metrics
        if all(col in data.columns for col in ['FG%', '3P%', 'FT%']):
            data['Shooting_Efficiency'] = (data['FG%'] + data['3P%'] + data['FT%']) / 3
            
        # Per Minute Stats
        if 'MP' in data.columns:
            for stat in ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'ORB', 'DRB']:
                if stat in data.columns:
                    data[f'{stat}_per_36'] = data[stat] * 36 / data['MP']
                    data[f'{stat}_per_Minute'] = data[stat] / data['MP']
        
        # Rebounding Metrics
        if all(col in data.columns for col in ['ORB', 'DRB', 'TRB']):
            data['ORB_Percentage'] = data['ORB'] / (data['TRB'] + 1e-6)
            data['DRB_Percentage'] = data['DRB'] / (data['TRB'] + 1e-6)
            data['Rebounding_Balance'] = abs(data['ORB'] - data['DRB']) / (data['TRB'] + 1e-6)
        
        # Playmaking Metrics
        if all(col in data.columns for col in ['AST', 'TOV', 'USG%']):
            data['AST_TO_Ratio'] = data['AST'] / (data['TOV'] + 1e-6)
            data['Playmaking_Efficiency'] = data['AST'] / (data['USG%'] + 1e-6)
        
        # Scoring Versatility
        if all(col in data.columns for col in ['2P', '3P', 'FT']):
            data['Scoring_Distribution'] = (data['2P'] + 1.5 * data['3P'] + 0.5 * data['FT']) / (data['2P'] + data['3P'] + data['FT'] + 1e-6)
            
        # Overall Impact Metrics
        if all(col in data.columns for col in ['PTS', 'AST', 'TRB', 'STL', 'BLK', 'TOV']):
            data['Box_Score_Impact'] = (data['PTS'] + 1.2*data['AST'] + 1.1*data['TRB'] + 
                                      2*data['STL'] + 2*data['BLK'] - data['TOV'])
            
        # Recent Form Features (if game number or date is available)
        if 'G' in data.columns:
            for stat in ['PTS', 'AST', 'TRB']:
                if stat in data.columns:
                    data[f'{stat}_per_Game'] = data[stat] / data['G']
        
        # Usage and Efficiency Combined
        if all(col in data.columns for col in ['USG%', 'TS%']):
            data['Usage_Efficiency'] = data['USG%'] * data['TS%']
        
        # Position-Independent Metrics
        if all(col in data.columns for col in ['PTS', 'AST', 'TRB']):
            data['Offensive_Versatility'] = (data['PTS'] + 2*data['AST']) / (data['PTS'] + data['AST'] + 1e-6)
            data['Floor_Impact'] = (data['PTS'] + 1.5*data['AST'] + data['TRB']) / 3
        
        # Advanced Defensive Metrics
        if all(col in data.columns for col in ['STL', 'BLK', 'PF']):
            data['Defensive_Activity'] = (data['STL'] + data['BLK']) / (data['PF'] + 1e-6)
        
        # Consistency Metrics
        if 'MP' in data.columns:
            for stat in ['PTS', 'AST', 'TRB']:
                if f'{stat}_per_36' in data.columns:
                    data[f'{stat}_Consistency'] = data[stat] / (data[f'{stat}_per_36'] + 1e-6)
        
        # Print created features
        new_features = [col for col in data.columns if col not in set(data.columns)]
        logging.info("\nCreated Advanced Features:")
        logging.info(new_features)
        
        return data
        
    except Exception as e:
        logging.error(f"Error creating advanced features: {str(e)}")
        return data

def optimize_model_hyperparameters(X_train, y_train, X_test, y_test):
    """Find optimal hyperparameters for the model"""
    param_grid = {
        'model__n_estimators': [500, 750, 1000],  # Increased epochs
        'model__learning_rate': [0.01, 0.03, 0.05],  # Finer learning rates
        'model__max_depth': [4, 5, 6],
        'model__min_samples_split': [2, 4, 6],
        'model__subsample': [0.85, 0.9, 0.95],
        'model__max_features': ['sqrt', 'log2'],
        'model__min_samples_leaf': [2, 3, 4],
        'model__warm_start': [True],
        'model__validation_fraction': [0.1],
        'model__n_iter_no_change': [50],  # Early stopping patience
        'model__tol': [1e-5]  # Tolerance for early stopping
    }
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', GradientBoostingRegressor(
            random_state=42,
            verbose=1  # Add verbosity to see training progress
        ))
    ])
    
    # Use more cross-validation folds for better accuracy
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=10,  # Increased from 5 to 10
        scoring=['neg_mean_squared_error', 'r2'],
        refit='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit with early stopping
    grid_search.fit(X_train, y_train)
    
    # Get best model and its performance
    best_model = grid_search.best_estimator_
    train_score = best_model.score(X_train, y_train)
    test_score = best_model.score(X_test, y_test)
    
    # Print detailed performance metrics
    logging.info(f"\nBest parameters: {grid_search.best_params_}")
    logging.info(f"Training R2: {train_score:.3f}")
    logging.info(f"Testing R2: {test_score:.3f}")
    
    # Calculate and print feature importances
    if hasattr(best_model.named_steps['model'], 'feature_importances_'):
        importances = best_model.named_steps['model'].feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logging.info("\nTop 10 Most Important Features:")
        logging.info(feature_importance.head(10))
    
    # Print learning curves
    train_scores = grid_search.cv_results_['mean_train_score']
    test_scores = grid_search.cv_results_['mean_test_score']
    logging.info("\nLearning Curves:")
    logging.info(f"Train scores: {train_scores}")
    logging.info(f"Test scores: {test_scores}")
    
    return best_model

def train_player_model(data, target_stat):
    """Train an optimized model for predicting a specific statistic"""
    
    # Get numeric columns for features
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    features = [col for col in numeric_cols if col != target_stat]
    
    # Add advanced features
    data = create_advanced_features(data)
    
    # Prepare the data
    X = data[features]
    y = data[target_stat]
    
    # Remove outliers more carefully
    z_scores = stats.zscore(y)
    mask = abs(z_scores) < 2.5  # Slightly less aggressive outlier removal
    X = X[mask]
    y = y[mask]
    
    # Create train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Find best model with increased training
    best_model = optimize_model_hyperparameters(X_train, y_train, X_val, y_val)
    
    # Final evaluation on test set
    test_score = best_model.score(X_test, y_test)
    test_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    logging.info(f"\nFinal Test Set Performance for {target_stat}:")
    logging.info(f"R2 Score: {test_score:.3f}")
    logging.info(f"RMSE: {rmse:.2f}")
    
    return best_model, features

def calculate_prediction_odds(model, X, prediction, stat):
    """Calculate probability ranges for predictions"""
    try:
        # Get the model from the pipeline
        gb_model = model.named_steps['model']
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in gb_model.estimators_[:, 0]])
        
        # Calculate statistical measures
        mean = np.mean(predictions)
        std = np.std(predictions)
        
        # Calculate probability ranges
        ranges = {
            'Very Likely Range (70%)': (
                round(mean - 0.5 * std, 1),
                round(mean + 0.5 * std, 1)
            ),
            'Likely Range (95%)': (
                round(mean - std, 1),
                round(mean + std, 1)
            ),
            'Possible Range (99%)': (
                round(mean - 2 * std, 1),
                round(mean + 2 * std, 1)
            )
        }
        
        # Calculate probability of exceeding prediction
        prob_exceed = (predictions > prediction).mean() * 100
        
        return {
            'prediction': round(prediction, 1),
            'ranges': ranges,
            'exceed_probability': round(prob_exceed, 1)
        }
        
    except Exception as e:
        logging.error(f"Error calculating odds for {stat}: {str(e)}")
        return None

def predict_player_against_team(models, player_data, player_name="Cade Cunningham", opponent="CHA"):
    """Predict stats for a specific player against a specific team"""
    try:
        # First, let's identify the player name column
        possible_player_columns = ['Player', 'PLAYER', 'player_name', 'Name', 'PLAYER_NAME']
        player_column = None
        
        # Print all columns to debug
        logging.info("\nAvailable columns:")
        logging.info(player_data.columns.tolist())
        
        # Find the correct player column
        for col in possible_player_columns:
            if col in player_data.columns:
                player_column = col
                break
        
        if not player_column:
            logging.error("Could not find player name column. Available columns:")
            logging.error(player_data.columns.tolist())
            return None, None
        
        # Check if player exists
        available_players = player_data[player_column].unique()
        logging.info("\nAvailable players:")
        logging.info(available_players)
        
        if player_name not in available_players:
            logging.error(f"Player {player_name} not found in dataset.")
            logging.info("Available players:")
            logging.info(available_players)
            return None, None
        
        # Get player's recent stats using the correct column name
        player_stats = player_data[player_data[player_column] == player_name].iloc[-1].copy()
        
        # Make predictions
        predictions = {}
        odds = {}
        
        # Focus on key stats
        key_stats = ['PTS', 'AST', 'TRB']
        
        # Print current stats
        logging.info(f"\nCurrent stats for {player_name}:")
        for stat in key_stats:
            if stat in player_stats:
                logging.info(f"{stat}: {player_stats[stat]:.1f}")
        
        # Make predictions for key stats
        for stat in key_stats:
            if stat not in models:
                continue
                
            model = models[stat]['model']
            features = models[stat]['features']
            
            try:
                X = pd.DataFrame([player_stats[features]], columns=features)
                pred = model.predict(X)[0]
                
                # Apply opponent factors
                opponent_factors = {
                    'PTS': 1.05,  # Charlotte allows more points
                    'AST': 1.02,  # Slight boost for assists
                    'TRB': 1.0,   # Neutral for rebounds
                }
                
                if stat in opponent_factors:
                    pred *= opponent_factors[stat]
                
                predictions[stat] = round(pred, 1)
                odds[stat] = calculate_prediction_odds(model, X, pred, stat)
                
            except Exception as e:
                logging.error(f"Error predicting {stat}: {str(e)}")
                continue
        
        # Calculate combined prediction (PTS + AST + REB)
        if all(stat in predictions for stat in key_stats):
            combined = sum(predictions[stat] for stat in key_stats)
            predictions['PAR'] = round(combined, 1)  # Points + Assists + Rebounds
        
        # Print predictions in a clean format
        if predictions:
            logging.info(f"\n=== Predictions for {player_name} vs {opponent} ===")
            
            # Print individual stats with odds
            for stat in key_stats:
                if stat in predictions and odds[stat]:
                    logging.info(f"\n{stat}:")
                    logging.info(f"Prediction: {predictions[stat]}")
                    logging.info("Probability Ranges:")
                    for range_name, (low, high) in odds[stat]['ranges'].items():
                        logging.info(f"  {range_name}: {low} - {high}")
                    logging.info(f"Probability of exceeding prediction: {odds[stat]['exceed_probability']}%")
            
            # Print combined stats
            if 'PAR' in predictions:
                logging.info(f"\nCombined (PTS + AST + REB): {predictions['PAR']}")
                logging.info(f"Breakdown:")
                logging.info(f"  Points: {predictions['PTS']}")
                logging.info(f"  Assists: {predictions['AST']}")
                logging.info(f"  Rebounds: {predictions['TRB']}")
        
        return predictions, odds
        
    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        raise

def ensemble_predict(models, X):
    """Make predictions using an ensemble of models"""
    predictions = []
    weights = [0.4, 0.3, 0.3]  # Weights for different models
    
    for model, weight in zip(models, weights):
        pred = model.predict(X)
        predictions.append(pred * weight)
    
    return np.sum(predictions, axis=0)

def main():
    # Load data
    player_data = load_player_data()
    
    # List of players to predict
    players_to_predict = [
        'Damian Lillard', 'Deaaron Fox', 'Franz Wagner', 'Giannis A', 'Jayson Tatum',
        'Jaylen Brown', 'Jalen Brunson', 'Joel Embiid', 'Luka Doncic',
        'Karl-Anthony Towns', 'Kevin Durant', 'Kryie Irving', 'Lamelo Ball',
        'Nikola Jokic', 'Norman Powell', 'Paolo Banchero', 'Shai Gilgeous-Alexander',
        'Tyler Herro', 'Tyrese Maxey', 'Victor Wembanyama', 'Anthony Edwards',
        'Anthony Davis', 'Cade Cunningham', 'Cam Thomas', 'Devin Booker'
    ]
    
    # Stats to predict
    target_stats = ['PTS', 'AST', 'TRB']
    
    # Train models with all available features
    models = {}
    for stat in target_stats:
        try:
            logging.info(f"\nTraining model for {stat}")
            model, features = train_player_model(player_data, stat)
            models[stat] = {'model': model, 'features': features}
        except Exception as e:
            logging.error(f"Error training model for {stat}: {str(e)}")
            continue
    
    # Store all predictions
    all_predictions = {}
    
    # Make predictions for each player
    for player in players_to_predict:
        logging.info(f"\n{'='*50}")
        logging.info(f"Predicting stats for {player}")
        logging.info('='*50)
        
        predictions, odds = predict_player_against_team(models, player_data, player, "CHA")
        
        if predictions:
            all_predictions[player] = {
                'predictions': predictions,
                'odds': odds
            }
    
    # Print comprehensive summary
    logging.info("\n\n=== FINAL PREDICTIONS SUMMARY ===")
    logging.info("\nPlayer                  PTS    AST    TRB    PAR   (Confidence Ranges)")
    logging.info("-" * 80)
    
    for player in players_to_predict:
        if player in all_predictions:
            pred = all_predictions[player]['predictions']
            player_odds = all_predictions[player]['odds']
            
            pts = pred.get('PTS', '-')
            ast = pred.get('AST', '-')
            trb = pred.get('TRB', '-')
            par = pred.get('PAR', '-')
            
            # Get confidence ranges
            ranges = {}
            for stat in ['PTS', 'AST', 'TRB']:
                if stat in player_odds and player_odds[stat]:
                    ranges[stat] = player_odds[stat]['ranges']['Very Likely Range (70%)']
            
            # Format the output
            base_line = f"{player:<20} {pts:>6} {ast:>6} {trb:>6} {par:>6}"
            
            # Add confidence ranges
            range_info = []
            for stat in ['PTS', 'AST', 'TRB']:
                if stat in ranges:
                    low, high = ranges[stat]
                    range_info.append(f"{stat}:{low}-{high}")
            
            logging.info(f"{base_line}   ({', '.join(range_info)})")
            
            # Print detailed odds
            logging.info(f"{'':20} Probabilities to exceed:")
            for stat in ['PTS', 'AST', 'TRB']:
                if stat in player_odds and player_odds[stat]:
                    prob = player_odds[stat]['exceed_probability']
                    logging.info(f"{'':20} {stat}: {prob}% chance to exceed {pred[stat]}")
            logging.info("-" * 80)
    
    return models, all_predictions

if __name__ == "__main__":
    models, all_predictions = main()
