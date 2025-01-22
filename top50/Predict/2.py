import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA

master_advanced = pd.read_csv(r'C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Predict\master_advanced_stats.csv')
master_base = pd.read_csv(r'C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Predict\master_base_stats.csv')
master_misc = pd.read_csv(r'C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Predict\master_misc_stats.csv')
master_scoring = pd.read_csv(r'C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Predict\master_scoring_stats.csv')
master_usage = pd.read_csv(r'C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Predict\master_usage_stats.csv')

master_df = pd.concat([master_advanced, master_base, master_misc, master_scoring, master_usage], axis=1)

# Remove duplicate columns
master_df = master_df.loc[:, ~master_df.columns.duplicated()]

# Print the column names to check for the player name and player_id columns
print("Column names:", master_df.columns)

# Check for missing values in each column and fill them with 0
master_df = master_df.fillna(0)

# Convert 'date' column to datetime if it exists
if 'date' in master_df.columns:
    master_df['date'] = pd.to_datetime(master_df['date'])

# Extract the year from 'By Year' column and create a new column 'Year'
if 'By Year' in master_df.columns:
    master_df['Year'] = pd.to_numeric(master_df['By Year'].str[:4])

# Identify the column causing the ValueError and exclude it from feature columns
exclude_columns = ['player_name', 'player_id', 'By Year']
for col in master_df.columns:
    try:
        master_df[col].astype(float)
    except ValueError:
        exclude_columns.append(col)

# Select all columns except the excluded columns for feature creation
feature_columns = [col for col in master_df.columns if col not in exclude_columns]

# Normalize the data (excluding the problematic columns)
scaler = StandardScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(master_df[feature_columns]), 
                             columns=feature_columns)

# Concatenate the principal components back with the player names, player_id, By Year, Year, and the excluded columns
final_df = pd.concat([master_df[exclude_columns + ['Year']], master_df[feature_columns]], axis=1)

# Keep only the specified columns in final_df
final_df = final_df[['player_name', 'player_id', 'By Year', 'TEAM']]

# Define the feature columns (keeping all original features)
feature_columns = ['By Year', 'GP', 'MIN', 'OffRtg', 'DefRtg', 'NetRtg', 'AST%',
       'AST/TO', 'AST Ratio', 'OREB%', 'DREB%', 'REB%', 'TO Ratio', 'eFG%',
       'TS%', 'USG%', 'PACE', 'PIE', 'FGM', 'FGA', 'FG%', '3PM', '3PA',
       '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'TOV', 'STL',
       'BLK', 'PF', 'FP', 'DD2', 'TD3', '+/-', 'FBPs', 'PITP', 
       'Opp', 'Opp.1', 'Opp.2', 'Opp.3', 'BLKA', 'PFD',
       '%FGA', '%FGA.1', '%PTS', '%PTS.1', '%PTS.2', '%PTS.3', '%PTS.4',
       '%PTS.5', '%PTS.6', '2FGM', '2FGM.1', '3FGM', '3FGM.1', 'FGM.1',
       '%FGM', '%3PM', '%3PA', '%FTM', '%FTA', '%OREB', '%DREB', '%REB',
       '%AST', '%TOV', '%STL', '%BLK', '%BLKA', '%PF', '%PFD', 'TEAM']

# Define the target variables we want to predict
target_columns = ['PTS', 'AST', 'REB','FGM','FGA','3PM','3PA']

# Keep only the necessary columns in final_df
final_df = master_df[feature_columns + target_columns + ['player_name', 'player_id']]

# Data preprocessing
final_df['By Year'] = pd.to_numeric(final_df['By Year'].str[:4])
final_df = pd.get_dummies(final_df, columns=['TEAM'])

# Prepare features (X) and targets (y)
X = final_df.drop(['player_name', 'player_id'] + target_columns, axis=1)
y = final_df[target_columns]

# Handle any remaining non-numeric columns and scale the features
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [800, 1000, 1200],      # More trees for better learning
    'max_depth': [25, 35, 45, None],        # Wider range of depth options
    'min_samples_split': [2, 3],            # Fine-tuned split thresholds
    'max_features': ['sqrt', 'log2', None], # All feature selection options
    'min_samples_leaf': [1, 2],             # Leaf size options
    'bootstrap': [True],                    # Enable bootstrapping
    'max_samples': [0.7, 0.8, 0.9]         # Different sample sizes for bootstrapping
}

# Dictionary to store optimized models and their predictions
models = {}
predictions = {}
feature_importances = {}

# Train and optimize models for each target variable
for stat in target_columns:
    print(f"\nOptimizing model for {stat}...")
    
    # Initial feature selection
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    selector = SelectFromModel(base_model, prefit=False)
    selector.fit(X_train, y_train[stat])
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Grid search with cross-validation
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        warm_start=True
    )
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=10,                           # 10-fold cross-validation
        n_jobs=-1,                      # Use all CPU cores
        scoring=['neg_mean_squared_error', 'r2'],
        refit='neg_mean_squared_error',
        verbose=2,
        return_train_score=True,
        error_score='raise'
    )
    
    # Fit the model
    grid_search.fit(X_train_selected, y_train[stat])
    
    # Store the best model
    models[stat] = grid_search.best_estimator_
    
    # Make predictions
    predictions[stat] = models[stat].predict(X_test_selected)
    
    # Store feature importance
    feature_importances[stat] = pd.DataFrame({
        'Feature': X.columns[selector.get_support()],
        'Importance': models[stat].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Print model performance metrics
    print(f"\nBest parameters for {stat}:", grid_search.best_params_)
    print(f"Cross-validation score: {-grid_search.best_score_:.2f}")
    print(f"R² Score: {r2_score(y_test[stat], predictions[stat]):.2f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test[stat], predictions[stat]):.2f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test[stat], predictions[stat])):.2f}")

# Create results dataframe
results_df = pd.DataFrame({
    'Player': final_df.loc[y_test.index, 'player_name'],
    'Actual PTS': y_test['PTS'],
    'Predicted PTS': predictions['PTS'],
    'Actual AST': y_test['AST'],
    'Predicted AST': predictions['AST'],
    'Actual REB': y_test['REB'],
    'Predicted REB': predictions['REB']
})

# Print sample predictions
print("\nSample Predictions:")
print(results_df.head(50))

# Plot feature importance for each stat
plt.figure(figsize=(15, 5))
for i, stat in enumerate(target_columns, 1):
    plt.subplot(1, 3, i)
    top_features = feature_importances[stat].head(10)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.title(f'Top 10 Important Features for {stat}')
    plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# Save the models and results
#for stat in target_columns:
#    models[stat].save(f'model_{stat}.pkl')
#results_df.to_csv('prediction_results.csv', index=False)

def analyze_predictions(results_df, target_columns=['PTS', 'AST', 'REB']):
    """
    Analyze prediction accuracy for each statistic.
    """
    print("\nModel Accuracy Analysis:")
    print("-" * 50)
    
    for stat in target_columns:
        actual = results_df[f'Actual {stat}']
        predicted = results_df[f'Predicted {stat}']
        
        # Calculate error metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual, predicted)
        
        # Calculate percentage error
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Calculate accuracy within different margins
        within_1 = np.mean(np.abs(actual - predicted) <= 1) * 100
        within_2 = np.mean(np.abs(actual - predicted) <= 2) * 100
        within_5 = np.mean(np.abs(actual - predicted) <= 5) * 100
        
        print(f"\n{stat} Predictions Analysis:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"R² Score: {r2:.2f}")
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        print(f"Predictions within 1 {stat}: {within_1:.1f}%")
        print(f"Predictions within 2 {stat}: {within_2:.1f}%")
        print(f"Predictions within 5 {stat}: {within_5:.1f}%")
        
        # Find best and worst predictions
        errors = np.abs(actual - predicted)
        best_idx = errors.nsmallest(3).index
        worst_idx = errors.nlargest(3).index
        
        print("\nBest Predictions:")
        for idx in best_idx:
            print(f"Player: {results_df.loc[idx, 'Player']}")
            print(f"Actual: {actual[idx]:.1f}, Predicted: {predicted[idx]:.1f}")
        
        print("\nWorst Predictions:")
        for idx in worst_idx:
            print(f"Player: {results_df.loc[idx, 'Player']}")
            print(f"Actual: {actual[idx]:.1f}, Predicted: {predicted[idx]:.1f}")

# Add this after creating results_df
analyze_predictions(results_df, target_columns=['PTS', 'AST', 'REB','BLK','STL','TOV','FGM','FGA','3PM','3PA'])
