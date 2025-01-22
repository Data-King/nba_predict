import selenium.common.exceptions
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from io import StringIO
import time


game_seasons = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
months_dict = {
    'January': '01',
    'February': '02',
    'March': '03',
    'April': '04',
    'May': '05',
    'June': '06',
    'July': '07',
    'August': '08',
    'September': '09',
    'October': '10',
    'November': '11',
    'December': '12',
}


""" Initialize the webdriver and box scores dictionary """
basketball_refernece_root = 'https://www.basketball-reference.com'
chrome_options=webdriver.ChromeOptions()
chrome_options.add_experimental_option('detach', True)


driver = webdriver.Chrome(options=chrome_options)
box_scores = []
keys = []


def add_games_to_box_scores():
    """Adds box score link for every game to dictionary """
    for season in game_seasons:
        # Get months games were played
        driver.get(f'{basketball_refernece_root}/leagues/NBA_{season}_games.html')
        filter = driver.find_elements(By.CSS_SELECTOR, value='.filter a')
        links = [month.get_attribute('href') for month in filter]

        for link in links:
            # Update the link to get the box score
            month = link.split('/')[4].split('-')[1].split('.')[0]
            basketball_reference_link = link

            # open url in browser 
            driver.get(basketball_reference_link)

        #  Find table div which contains the box score
        try:
            table_div = driver.find_element(By.ID, value='div_schedule')
        except selenium.common.exceptions.NoSuchElementException:
            driver.get(basketball_reference_link)
            table_div = driver.find_element(By.ID, value='div_schedule')
        


        # Find  Anchor tags in the table div
        time.sleep(7)
        anchors = table_div.find_elements(By.CSS_SELECTOR, value='.center a')


add_games_to_box_scores()



""" Get the number of games in the box scores """
def get_number_of_games():
    sum = 0
    for key in keys:
        sum += len(box_scores[key])
    return sum

number_of_games = get_number_of_games()



def get_stats():
    games = []
    base_cols = None
    for key in keys:
        for i in range(0, len(box_scores[key])):
            # Max 20 requests per minute
            time.sleep(5)


            # basic & advanced team stats for specific game
            stats = []


            # Go to game box score link
            game_link = box_scores[key][i]
            driver.get(game_link)


            # Get Game Data
            try:
                header = driver.find_element(By.CSS_SELECTOR, value=' H1').text
                game_day = header.split(',')[1].split(' ')[2]
                game_month = months_dict[header.split(',')[1].split(' ')[1]]
                game_year = header.split[','][-1]
                game_season = key.split('-')[0]
            except IndexError:
                # Handle hos downtime with saving the progress to csv and try to continue after user approval
                input('There was an index error press enter to continue')

                full_df = pd.concat(games, axis=0)
                full_df = full_df.reset_index()
                full_df = full_df.drop('index', axis=1)
                full_df.to_csv(f'Basketball_data/all_game_stats_{game_seasons[0]}-{game_seasons[-1]}.csv', index = False)


                time.sleep(6)
                header=driver.find_element(By.CSS_SELECTOR, value=' H1').text
                game_day = header.split(',')[1].split(' ')[2]
                game_month = months_dict[header.split(',')[1].split(' ')[1]]
                game_year = header.split[','][-1]
                game_season = key.split('-')[0]


                
            # ===== GET LINE SCORES =====
            line_score_table = driver.find_element(By.ID, 'div_line_score').get_attribute('innerHTML')
            line_score_df = pd.read_html(StringIO(line_score_table))[0]

            # Adjust Dataframe
            line_score_df.columns = line_score_df.columns.droplevel()
            line_score_df = line_score_df.rename(columns={'Unnamed: 0_level_1': 'team', 'T': 'total'})
            line_score_df = line_score_df[['team', 'total']]

            # line_score_df.to_csv(
            #    f'Basketball_data/line_scores/{line_score_df["team"][0]}vs{line_score_df["team"][1]}_{key}-{game_day}.csv',
            #    index=False)

            # ===== GET BASIC & ADVANCED STATS =====
            teams = list(line_score_df['team'])
            #print(f'Gathering {teams[0]} vs {teams[1]} Data')
            for team in teams:
                advanced_id = f'div_box-{team}-game-advanced'
                basic_id = f'div_box-{team}-game-basic'

                # Find advanced stats table
                advanced_stats_table = driver.find_element(By.ID, advanced_id).get_attribute('innerHTML')
                advanced_stats_df = pd.read_html(StringIO(advanced_stats_table), index_col=0)[0]
                advanced_stats_df = advanced_stats_df.apply(pd.to_numeric, errors='coerce')
                advanced_stats_df.columns = advanced_stats_df.columns.droplevel()

                # Find basic stats table
                basic_stats_table = driver.find_element(By.ID, basic_id).get_attribute('innerHTML')
                basic_stats_df = pd.read_html(StringIO(basic_stats_table), index_col=0)[0]
                basic_stats_df = basic_stats_df.apply(pd.to_numeric, errors='coerce')
                basic_stats_df.columns = basic_stats_df.columns.droplevel()

                # Get total team stats for basic and advanced stats and concat.
                totals_df = pd.concat([basic_stats_df.iloc[-1, :], advanced_stats_df.iloc[-1, :]])
                totals_df.index = totals_df.index.str.lower()

                # Get Max scores for each stat & for each team (individual player)
                maxes_df = pd.concat([basic_stats_df.iloc[:-1, :].max(), advanced_stats_df.iloc[:-1, :].max()])
                maxes_df.index = maxes_df.index.str.lower() + '_max'

                stat = pd.concat([totals_df, maxes_df])

                if base_cols is None:
                    base_cols = list(stat.index.drop_duplicates(keep='first'))
                    base_cols = [b for b in base_cols if "bmp" not in b]

                stat = stat[base_cols]
                stats.append(stat)

            # Concat both stats
            stat_df = pd.concat(stats, axis=1).T

            # Create game df
            game = pd.concat([stat_df, line_score_df], axis=1)
            game['home'] = [0, 1]

            # Create Opponent columns
            game_opp = game.iloc[::-1].reset_index()
            game_opp.columns += '_opp'

            # Merge home + opponent columns
            full_game = pd.concat([game, game_opp], axis=1)

            full_game['season'] = game_season

            full_game['date'] = f'{game_year}-{game_month}-{game_day}'
            full_game['date'] = pd.to_datetime(full_game['date'])

            full_game['won'] = full_game['total'] > full_game['total_opp']

            # for every full game data we have 2 rows, from opponent teams perspective & from home teams perspective
            games.append(full_game)

            if len(games) % 100 == 0:
                print(f'{len(games)}/{number_of_games}')
            #full_game.to_csv(f'Basketball_data/game/teams_{teams[0]}-{teams[1]}_season_{game_season}_date_{game_year}-{game_month}-{game_day}.csv', index=False)
    
    return games




games = get_stats()
full_df = pd.concat(games, axis=0)
full_df = full_df.reset_index()
full_df = full_df.drop('index', axis=1)
full_df.to_csv(f'Historical Data/{game_seasons[0]}-{game_seasons[-1]}_nba_games.csv', index = False)

driver.quit()










