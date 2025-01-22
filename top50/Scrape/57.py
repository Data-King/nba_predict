
import pandas as pd
player_name = 'Mikal Bridges'
player_id = 57

def split_stats_dataframes(csv_path, player_name, player_id):
    # Read the CSV file
    with open(csv_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize variables
    current_section = None
    sections_data = {
        'base': {'headers': None, 'data': []},
        'advanced': {'headers': None, 'data': []},
        'misc': {'headers': None, 'data': []},
        'scoring': {'headers': None, 'data': []},
        'usage': {'headers': None, 'data': []}
    }
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Identify sections and their headers
        if 'Traditional Splits' in line:
            current_section = 'base'
        elif 'Advanced Splits' in line:
            current_section = 'advanced'
        elif 'Misc Splits' in line:
            current_section = 'misc'
        elif 'Scoring Splits' in line:
            current_section = 'scoring'
        elif 'Usage Splits' in line:
            current_section = 'usage'
        elif 'By Year' in line:
            if current_section:
                # Filter out empty columns from headers
                headers = [h.strip() for h in line.split(',')]
                sections_data[current_section]['headers'] = [h for h in headers if h]
        elif line.startswith('20') and current_section:  # Data rows start with year
            # Filter out empty values from data rows
            data = [d.strip() for d in line.split(',')]
            sections_data[current_section]['data'].append([d for d in data if d])
    
    # Create dataframes
    dataframes = {}
    for section in sections_data:
        if sections_data[section]['headers'] and sections_data[section]['data']:
            df = pd.DataFrame(sections_data[section]['data'], 
                            columns=sections_data[section]['headers'])
            
            # Convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    continue
                    
            # Remove empty columns and columns with all empty strings
            df = df.dropna(axis=1, how='all')
            df = df.loc[:, (df != '').any()]
            
            # Add player_name and player_id columns
            df['player_name'] = player_name
            df['player_id'] = player_id
            
            dataframes[section] = df
    
    return (dataframes['base'], 
            dataframes['advanced'], 
            dataframes['misc'], 
            dataframes['scoring'], 
            dataframes['usage'])


def save_dataframes_to_csv(base_df, advanced_df, misc_df, scoring_df, usage_df, player_name):
    """
    Save all dataframes to CSV files with player name in the filename.
    
    Args:
        base_df, advanced_df, misc_df, scoring_df, usage_df: pandas DataFrames
        player_name: string, name of the player for filename
    """
    # Create a clean player name for filenames (remove spaces)
    clean_name = player_name.replace(" ", "_")
    
    # Dictionary mapping dataframe types to their respective dataframes
    df_dict = {
        'base': base_df,
        'advanced': advanced_df,
        'misc': misc_df,
        'scoring': scoring_df,
        'usage': usage_df
    }
    
    # Save each dataframe
    for df_type, df in df_dict.items():
        filename = f'Data/{clean_name}_{df_type}_stats.csv'
        df.to_csv(filename, index=False)
        print(f"Saved {df_type} stats to {filename}")





if __name__ == "__main__":
    file_path = r'C:\Users\harmo\OneDrive\Desktop\nba_predict\top50\Data2\Mikal B.csv'
    base_df, advanced_df, misc_df, scoring_df, usage_df = split_stats_dataframes(file_path, player_name, player_id)
    
    # Print information about each dataframe
    print("\nBase Stats DataFrame:")
    print(base_df.columns.tolist())  # Print columns to verify empty ones are removed
    print(base_df)
    
    print("\nAdvanced Stats DataFrame:")
    print(advanced_df.columns.tolist())
    print(advanced_df)
    
    print("\nMisc Stats DataFrame:")
    print(misc_df.columns.tolist())
    print(misc_df)
    
    print("\nScoring Stats DataFrame:")
    print(scoring_df.columns.tolist())
    print(scoring_df)
    
    print("\nUsage Stats DataFrame:")
    print(usage_df.columns.tolist())
    print(usage_df)



    save_dataframes_to_csv(base_df, advanced_df, misc_df, scoring_df, usage_df, player_name)