import pandas as pd
import numpy as np

# Create the DataFrame with the specified structure
data = [
    ["Thu, Jan 16, 2025", "7:00p", "Indiana Pacers", "", "Detroit Pistons", "", "", "", "", "", "Little Caesars Arena", ""],
    ["Thu, Jan 16, 2025", "7:00p", "Phoenix Suns", "", "Washington Wizards", "", "", "", "", "", "Capital One Arena", ""],
    ["Thu, Jan 16, 2025", "7:30p", "Cleveland Cavaliers", "", "Oklahoma City Thunder", "", "", "", "", "", "Paycom Center", ""],
    ["Thu, Jan 16, 2025", "10:00p", "Los Angeles Clippers", "", "Portland Trail Blazers", "", "", "", "", "", "Moda Center", ""],
    ["Thu, Jan 16, 2025", "10:00p", "Houston Rockets", "", "Sacramento Kings", "", "", "", "", "", "Golden 1 Center", ""],
    ["Fri, Jan 17, 2025", "7:00p", "Orlando Magic", "", "Boston Celtics", "", "", "", "", "", "TD Garden", ""],
    ["Fri, Jan 17, 2025", "7:30p", "Minnesota Timberwolves", "", "New York Knicks", "", "", "", "", "", "Madison Square Garden (IV)", ""],
    ["Fri, Jan 17, 2025", "8:00p", "Charlotte Hornets", "", "Chicago Bulls", "", "", "", "", "", "United Center", ""],
    ["Fri, Jan 17, 2025", "8:00p", "Denver Nuggets", "", "Miami Heat", "", "", "", "", "", "Kaseya Center", ""],
    ["Fri, Jan 17, 2025", "8:00p", "Toronto Raptors", "", "Milwaukee Bucks", "", "", "", "", "", "Fiserv Forum", ""],
    ["Fri, Jan 17, 2025", "8:00p", "Utah Jazz", "", "New Orleans Pelicans", "", "", "", "", "", "Smoothie King Center", ""],
    ["Fri, Jan 17, 2025", "8:30p", "Oklahoma City Thunder", "", "Dallas Mavericks", "", "", "", "", "", "American Airlines Center", ""]
]

# Define column names
columns = ['Date', 'Start (ET)', 'Visitor/Neutral', 'PTS', 'Home/Neutral', 'PTS', '', '', 'Attend.', 'LOG', 'Arena', 'Notes']

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Drop columns where all values are NA or empty strings
df = df.replace('', np.nan)  # Replace empty strings with NaN
df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN

print(df.columns)  # To verify the remaining columns

# Save the cleaned DataFrame to CSV
df.to_csv('Jan week 5 2025 NBA - Sheet1.csv', index=False)