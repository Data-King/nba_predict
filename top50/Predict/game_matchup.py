
def predict_game_matchup():
    """
    Predict stats for all players in a game matchup automatically.
    """
    # First, ensure we have the correct data
    print("\nLoading team and player data...")
    
    # Get unique teams and sort them
    available_teams = sorted(master_df['TEAM'].unique())
    
    while True:
        print("\n" + "="*50)
        print("NBA GAME PREDICTION TOOL")
        print("="*50)
        
        # Show teams in a grid format
        print("\nAVAILABLE TEAMS:")
        print("-" * 50)
        for i in range(0, len(available_teams), 5):
            row_teams = available_teams[i:i+5]
            print("  ".join(f"{team:^5}" for team in row_teams))
        
        try:
            # Get matchup input
            print("\nEnter matchup teams (or 'quit' to exit)")
            home_team = input("Home Team (e.g., BOS): ").strip().upper()
            
            if home_team.lower() == 'quit':
                break
                
            away_team = input("Away Team (e.g., LAL): ").strip().upper()
            
            # Validate teams
            if home_team not in available_teams:
                print(f"\n❌ Error: {home_team} not found. Please use team codes shown above.")
                continue
                
            if away_team not in available_teams:
                print(f"\n❌ Error: {away_team} not found. Please use team codes shown above.")
                continue
            
            # Print matchup header
            print(f"\n{'='*20} {home_team} vs {away_team} {'='*20}")
            
            # Process each team
            for team, label in [(home_team, "HOME"), (away_team, "AWAY")]:
                print(f"\n{team} ({label}) PREDICTIONS:")
                print("-" * 60)
                print(f"{'PLAYER':<25} | {'PTS':>6} {'AST':>6} {'REB':>6}")
                print("-" * 60)
                
                # Get team players
                team_players = master_df[master_df['TEAM'] == team]['player_name'].unique()
                
                if len(team_players) == 0:
                    print(f"No players found for {team}")
                    continue
                
                # Process each player
                for player in team_players:
                    try:
                        # Get latest stats
                        player_stats = master_df[master_df['player_name'] == player].iloc[-1]
                        
                        # Prepare features
                        features = pd.DataFrame([player_stats[feature_columns]])
                        features_scaled = scaler.transform(features)
                        
                        # Make predictions
                        predictions = {}
                        for stat in ['PTS', 'AST', 'REB']:
                            pred = models[stat].predict(features_scaled)[0]
                            predictions[stat] = max(0, round(pred, 1))
                        
                        # Print prediction
                        print(f"{player:<25} | {predictions['PTS']:>6.1f} {predictions['AST']:>6.1f} {predictions['REB']:>6.1f}")
                        
                    except Exception as e:
                        print(f"Error predicting for {player}: {str(e)}")
            
            # Ask about another prediction
            again = input("\nPredict another game? (y/n): ").lower().strip()
            if again != 'y':
                break
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again.")
            continue
    
    print("\nThanks for using the NBA Prediction Tool!")

def run_prediction_tool():
    print("\nWelcome to NBA Prediction Tool")
    print("=" * 50)
    
    while True:
        print("\n1. Make game predictions")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1 or 2): ")
        
        if choice == '1':
            predict_game_matchup()
        elif choice == '2':
            print("\nThanks for using the NBA Prediction Tool!")
            break
        else:
            print("\nInvalid choice. Please enter 1 or 2.")

# Call the menu function directly
run_prediction_tool()
