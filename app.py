from flask import Flask, jsonify
import pandas as pd
import joblib
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Table, Column, Integer, String, Float

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import logging
logging.basicConfig(level=logging.DEBUG)


import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import pandas as pd
import numpy as np
import sqlite3
import os

basedir = os.path.abspath(os.path.dirname(__file__))


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///' + os.path.join(basedir, '_app.db')

db = SQLAlchemy(app)
ma = Marshmallow(app)
from datetime import datetime


# Declare model
class GameData(db.Model):
    __tablename__ = 'game_table'
    id = db.Column(db.Integer, primary_key=True)
    team_name = db.Column(db.String(80))
    matchup = db.Column(db.String(120))
    game_date = db.Column(db.TIMESTAMP, default=datetime.now)   
    prediction = db.Column(db.String(80))

    def __init__(self, team_name, matchup, game_date, prediction) -> None:
        super(GameData, self).__init__()
        self.team_name = team_name
        self.matchup = matchup
        self.game_date = game_date
        self.prediction = prediction

    def __repr__(self):
        return f'<Prediction(team_name={self.team_name}, matchup={self.matchup}, prediction={self.prediction})>'

class NBADataSchema(ma.Schema):
    class Meta:
        fields = ('id', 'ticker', 'std_dev', 'image_url')

single_NBA_data_schema = NBADataSchema()
multiple_NBA_data_schema = NBADataSchema(many=True)

# Create tables/db file
with app.app_context():
    db.create_all()


# Main url
@app.route('/', methods=['GET'])
def create_main():
    all_data = GameData.query.all()
    print(all_data) 
    all_data_ser = multiple_NBA_data_schema.dump(all_data)
    all_data_df = pd.DataFrame(all_data_ser)
    print(all_data_df)
    return render_template('index.html', NBA_data=all_data_df)
    
@app.route('/prediction_finder.html')
def prediction_finder():
    return render_template('prediction_finder.html')

    


#   @app.route('/')
# Function to fetch and process data
def fetch_and_process_data(seasons=['2021-22', '2022-23']):
    try:
        all_games_df = pd.DataFrame()
        nba_teams = teams.get_teams()

        for season in seasons:
            gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
            games_dict = gamefinder.get_dict()
            games = games_dict['resultSets'][0]['rowSet']
            columns = games_dict['resultSets'][0]['headers']
            season_games_df = pd.DataFrame(games, columns=columns)
            all_games_df = pd.concat([all_games_df, season_games_df], ignore_index=True)

        nba_team_ids = [team['id'] for team in nba_teams]
        all_games_df = all_games_df[all_games_df['TEAM_ID'].isin(nba_team_ids)]
        all_games_df = all_games_df[['SEASON_ID', 'TEAM_ID', 'TEAM_NAME', 'MATCHUP', 'WL', 'PTS', 'PLUS_MINUS', 'GAME_DATE', 'FG_PCT']]
        all_games_df = all_games_df[all_games_df['SEASON_ID'].astype(str).str.startswith(('2'))]
        
        final_df = all_games_df
        # Convert GAME_DATE to datetime
        final_df['GAME_DATE'] = pd.to_datetime(final_df['GAME_DATE'])

        # Sort by TEAM_ID and GAME_DATE
        final_df.sort_values(by=['TEAM_ID', 'GAME_DATE'], inplace=True)
        final_df['Season'] = final_df['SEASON_ID'].astype(str).str[-4:]  # sets season column to season

        #average plus minus, pts, and points allowed for last 10 games
        final_df['avg_plus_minus_last_10'] = round(final_df.groupby('TEAM_ID')['PLUS_MINUS'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).mean()), 3)
        final_df['avg_points_last_10'] = round(final_df.groupby('TEAM_ID')['PTS'].transform(lambda x: x.shift().rolling(window=10, min_periods=1 ).mean()),3)
        final_df['avg_points_allowed_last_10'] = round(final_df['avg_points_last_10'] - final_df['avg_plus_minus_last_10'], 3)
        final_df.dropna(subset=['avg_plus_minus_last_10', 'avg_points_last_10', 'avg_points_allowed_last_10'], inplace=True)

        #opponents scored, found by subtracting pts from plus minus
        final_df['opponents_points_scored'] = final_df['PTS'] - final_df['PLUS_MINUS']

        #Home/Away Games
        final_df['Home_Game'] = final_df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)

        #back to back games
        final_df['Back_to_back'] = 0
        for team in final_df['TEAM_NAME'].unique():
            team_games = final_df[final_df['TEAM_NAME'] == team]
            prev_date = None
            for index,row in team_games.iterrows():
                if prev_date is not None and (row['GAME_DATE']-prev_date).days == 1 and prev_date:
                    final_df.at[index, 'Back_to_back'] = 1
                prev_date = row['GAME_DATE']

        #season plus minus, pts
        final_df['avg_points_season'] = round(final_df.groupby('TEAM_ID')['PTS'].transform(lambda x: x.rolling(window=82, min_periods=1).mean()), 3)
        final_df['avg_plus_minus_season'] = round(final_df.groupby('TEAM_ID')['PLUS_MINUS'].transform(lambda x: x.rolling(window=82, min_periods=1).mean()), 3)

        #win percentage
        final_df['Win'] = final_df['WL'].apply(lambda x: 1 if x == 'W' else 0)  # numerizes wins
        final_df['Loss'] = final_df['WL'].apply(lambda x: 1 if x == 'L' else 0) #numerizes losses
        final_df['cumulative_wins'] = final_df.groupby(['TEAM_ID', 'Season'])['Win'].cumsum()
        final_df['total_games'] = final_df.groupby(['TEAM_ID', 'Season']).cumcount() + 1
        final_df['Win_Percentage'] = round((final_df['cumulative_wins'] / final_df['total_games']) * 100, 3)

        #Opponent Win Record
        final_df['TEAM_ABB'] = final_df['MATCHUP'].apply(lambda x: x.split(' ')[0])
        final_df['OPPONENT_TEAM_NAME'] = final_df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
        final_df.sort_values(by=['TEAM_NAME', 'OPPONENT_TEAM_NAME', 'GAME_DATE'], inplace=True)
        final_df['Rolling_Win'] = final_df.groupby(['TEAM_NAME', 'OPPONENT_TEAM_NAME'])['Win'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).sum())
        final_df['Rolling_Loss'] = final_df.groupby(['TEAM_NAME', 'OPPONENT_TEAM_NAME'])['Loss'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).sum())
        # Calculate rolling win percentage
        final_df['Rolling_Win_Percentage'] = (final_df['Rolling_Win'] / (final_df['Rolling_Win'] + final_df['Rolling_Loss'])).round(3)
        final_df['Rolling_Win_Percentage'] = final_df['Rolling_Win_Percentage'].fillna(0.5)

        def get_opponent_feature(row, feature):
            # Filter by opponent team name, and the last game date before the current game
            opponent_data = final_df.loc[(final_df['TEAM_ABB'] == row['OPPONENT_TEAM_NAME']) & (final_df['GAME_DATE'] == row['GAME_DATE'])]
            if not opponent_data.empty:
                return opponent_data.iloc[-1][feature]
            else:
                return 0  # Default value if no previous data is found

        final_df['Opponent_Back_to_back'] = final_df.apply(get_opponent_feature, axis=1, feature = 'Back_to_back')
        final_df['Opponent_avg_plus_minus_last_10'] = final_df.apply(get_opponent_feature, axis=1, feature = 'avg_plus_minus_last_10')
        final_df['Opponent_Win_Percentage'] = final_df.apply(get_opponent_feature, axis=1, feature = 'Win_Percentage')
        final_df['Opponent_plus_minus_season'] = final_df.apply(get_opponent_feature, axis=1, feature = 'avg_plus_minus_season')
        #filtering out only the useful information and ordering it to be neater
        final_df = final_df[['TEAM_ID','Opponent_plus_minus_season', 'Opponent_Win_Percentage', 'Opponent_avg_plus_minus_last_10', 'Opponent_Back_to_back','TEAM_ABB', 'OPPONENT_TEAM_NAME','Home_Game', 'GAME_DATE','Season', 'TEAM_NAME', 'MATCHUP', 'WL','Win', 'PTS','opponents_points_scored', 'PLUS_MINUS',  'avg_plus_minus_last_10' , 'avg_points_last_10' ,'avg_points_allowed_last_10','avg_points_season','avg_plus_minus_season','cumulative_wins','total_games', 'Win_Percentage', 'Back_to_back', 'Rolling_Win_Percentage']]
        #making the games by game date instead of each team
        final_df.sort_values(by=['GAME_DATE'], inplace=True)
    except Exception as e:
        print(f"Data Error: {e}")
    
    return final_df

def get_fourty_first_game_for_team(team_id):
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, season_nullable="2023-24")
    games_dict = gamefinder.get_dict()
    games = games_dict['resultSets'][0]['rowSet']
    
    columns = games_dict['resultSets'][0]['headers']
    team_games_df = pd.DataFrame(games, columns=columns)
    first_game = team_games_df.loc[40] if not team_games_df.empty else None
    #first_game.to_csv(f"first_game ${team_id}", index = False)
    print("The teams 41st game should be on " + first_game["GAME_DATE"])
    if(first_game.empty):
        print("Empty")
    else:
        #print("Was able to find fourty first game" + str(first_game["TEAM_ID"]))
        first_game['TEAM_ABB'] = first_game['MATCHUP'].split(' ')[0]
        first_game['OPPONENT_TEAM_NAME'] = first_game['MATCHUP'].split(' ')[-1]
    return first_game




        

def generate_features_for_first_game(first_game_df, historical_data_df):
    try:
        
        if 'TEAM_ID' not in first_game_df:
            print("Hi lol")
            raise ValueError("first_game_df does not contain the 'TEAM_ID' column")
        
        home_team_id = first_game_df['TEAM_ID']  # Using iloc to safely access the first element
       
        home_team_data = historical_data_df[historical_data_df['TEAM_ID'] == home_team_id].tail(1)
        print(home_team_data)
        
        
        if home_team_data.empty:
            raise ValueError(f"No historical data found for home team ID {home_team_id}")
        
        features = [
            
            'Opponent_plus_minus_season', 'Opponent_Win_Percentage', 
            'Opponent_avg_plus_minus_last_10', 'Opponent_Back_to_back',
            'Back_to_back', 'Home_Game', 'avg_plus_minus_last_10', 
            'avg_points_last_10', 'avg_points_allowed_last_10', 
            'avg_points_season', 'avg_plus_minus_season', 
            'cumulative_wins', 'Win_Percentage', 'Rolling_Win_Percentage', 
            'total_games'
        ]
        
        if not set(features).issubset(home_team_data.columns):
            missing_features = set(features) - set(home_team_data.columns)
            raise ValueError(f"Missing required features in home_team_data: {missing_features}")
        
        home_features = home_team_data[features]
        
        return home_features
    
    except Exception as e:
        print(f"Error in generate_features_for_first_game: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

nba_teams_abbreviation_to_full = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
    }      
def predict_fourty_first_game_outcome(home_features):
    model = joblib.load('logistic_regression_model.pkl')
    home_prediction = model.predict(home_features)[0]
    
    return {
        'home_prediction': home_prediction,
        
    }


@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    data = fetch_and_process_data()
    return data.to_json()

@app.route('/train_model', methods=['GET'])
def train_model():
    final_df = fetch_and_process_data()
    features = ['Opponent_plus_minus_season', 'Opponent_Win_Percentage', 'Opponent_avg_plus_minus_last_10', 'Opponent_Back_to_back', 'Back_to_back', 'Home_Game', 'avg_plus_minus_last_10', 'avg_points_last_10', 'avg_points_allowed_last_10', 'avg_points_season', 'avg_plus_minus_season', 'cumulative_wins', 'Win_Percentage', 'Rolling_Win_Percentage', 'total_games']
    X = final_df[features]
    y = final_df['Win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    joblib.dump(model, 'logistic_regression_model.pkl')
    return jsonify({'message': 'Model trained successfully'})

"""@app.route('/first_game_predictions', methods=['GET'])
def show_first_game_predictions():
    try:
        logging.debug("Fetching and processing historical data.")
        historical_data_df = fetch_and_process_data()
        predictions = {}

        for team in teams:
            team_id = team['id']
            logging.debug(f"Getting the 41st game for team ID: {team_id}")
            first_game_df = get_fourty_first_game_for_team(team_id)
            
            if first_game_df is not None:
                logging.debug(f"Generating features for team ID: {team_id}")
                home_features, visitor_features = generate_features_for_first_game(first_game_df, historical_data_df)
                team_predictions = predict_fourty_first_game_outcome(home_features, visitor_features)
                predictions[team['full_name']] = team_predictions

        logging.debug("Rendering predictions template.")
        return render_template('predictions.html', predictions=predictions)
    
    except Exception as e:
        logging.error(f"Error in show_first_game_predictions: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

"""

@app.route('/first_game_predictions', methods=['GET'])
def show_first_game_predictions():
    try:
        print("requesting for teams")
        currTeam = request.args.get('team')
        print(f"Selected team: {currTeam}")
        historical_data_df = fetch_and_process_data()
        if historical_data_df.empty:
            print("Empty Error: historical_data_df is empty after fetch_and_process_data()")
            return jsonify({'error': 'No historical data available'}), 500
        
        predictions = {}
        
        nba_teams = teams.get_teams()  # Ensure teams are fetched correctly
        dates = {}
        #team_name_to_id = {team['full_name']: team['id'] for team in nba_teams}     
        print("hi lol")
        for team in nba_teams:
            team_id = team['id']
            first_game_df = get_fourty_first_game_for_team(team_id)
            
            if first_game_df is not None and not first_game_df.empty:
                home_features = generate_features_for_first_game(first_game_df, historical_data_df)
                if home_features.empty:
                    print(f"Error: home_features is empty for team {team['full_name']}")
                    continue
                    
                
                team_predictions = predict_fourty_first_game_outcome(home_features)
                prediction_value = team_predictions['home_prediction']
                finalString = "win" if prediction_value == 1 else "lose"
                predictions[team['full_name']] = finalString
                opponent_id = first_game_df['OPPONENT_TEAM_NAME']
                opponent_team = nba_teams_abbreviation_to_full.get(opponent_id)

                date = first_game_df["GAME_DATE"]
                #dates[currTeam] = date
                #.split("-"[]) + first_game_df["GAME_DATE"].split("-"[2])

                print(f"Found the data for {team['full_name']}")
                if team['full_name'] == currTeam:
                    break
            else:
                print(f"No first game data found for team {team['full_name']}")
        #dates.to_csv("dates.csv")
        if not predictions:
            print("Error: No predictions generated")
            return jsonify({'error': 'No predictions available'}), 500
        print("trying to get the prediction for the " + currTeam)
        prediction = predictions.get(currTeam)
        print(currTeam + " " + str(prediction))
        
        return jsonify({'team': currTeam, 'prediction': prediction, 'opponent': opponent_team, 'date': date})
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'prediction error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port = 8000)

'''
@app.route('/first_game_predictions', methods=['GET'])
def show_first_game_predictions():
    try:
        historical_data_df = fetch_and_process_data()
        if historical_data_df.empty:
            print("Error: historical_data_df is empty after fetch_and_process_data()")
            return jsonify({'error': 'No historical data available'}), 500
        
        predictions = {}
        
        nba_teams = teams.get_teams()
        
        for team in nba_teams:
            team_id = team['id']
            print(f"Processing team: {team['full_name']} (ID: {team_id})")
            
            first_game_df = get_fourty_first_game_for_team(team_id)
            if first_game_df is not None and not first_game_df.empty:
                print(f"First game data found for team {team['full_name']}")
                
                home_features = generate_features_for_first_game(first_game_df, historical_data_df)
                if home_features.empty:
                    print(f"Error: home_features is empty for team {team['full_name']}")
                    continue
                
                team_predictions = predict_fourty_first_game_outcome(home_features)
                predictions[team['full_name']] = team_predictions
            else:
                print(f"No first game data found for team {team['full_name']}")
        
        if not predictions:
            print("Error: No predictions generated")
            return jsonify({'error': 'No predictions available'}), 500

        return render_template('predictions.html', predictions=predictions)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

'''
