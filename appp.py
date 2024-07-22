'''import os
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
import pandas as pd
import joblib
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///' + os.path.join(basedir, '_app.db')
db = SQLAlchemy(app)
ma = Marshmallow(app)

# Declare model
class GameData(db.Model):
    __tablename__ = 'game_table'
    id = db.Column(db.Integer, primary_key=True)
    team_name = db.Column(db.String(80))
    matchup = db.Column(db.String(120))
    game_date = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    prediction = db.Column(db.String(80))

    def __init__(self, team_name, matchup, game_date, prediction):
        self.team_name = team_name
        self.matchup = matchup
        self.game_date = game_date
        self.prediction = prediction

class NBADataSchema(ma.Schema):
    class Meta:
        fields = ('id', 'team_name', 'matchup', 'game_date', 'prediction')

single_NBA_data_schema = NBADataSchema()
multiple_NBA_data_schema = NBADataSchema(many=True)

# Create tables/db file
with app.app_context():
    db.create_all()

# Main url
@app.route('/', methods=['GET'])
def create_main():
    all_data = GameData.query.all()
    all_data_ser = multiple_NBA_data_schema.dump(all_data)
    all_data_df = pd.DataFrame(all_data_ser)
    return render_template('index.html', NBA_data=all_data_df)

# Function to fetch and process data
def fetch_and_process_data(seasons=['2021-22', '2022-23', '2023-24']):
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
    final_df['GAME_DATE'] = pd.to_datetime(final_df['GAME_DATE'])
    final_df.sort_values(by=['TEAM_ID', 'GAME_DATE'], inplace=True)
    final_df['Season'] = final_df['SEASON_ID'].astype(str).str[-4:]
    final_df['avg_plus_minus_last_10'] = round(final_df.groupby('TEAM_ID')['PLUS_MINUS'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).mean()), 3)
    final_df['avg_points_last_10'] = round(final_df.groupby('TEAM_ID')['PTS'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).mean()), 3)
    final_df['avg_points_allowed_last_10'] = round(final_df['avg_points_last_10'] - final_df['avg_plus_minus_last_10'], 3)
    final_df.dropna(subset=['avg_plus_minus_last_10', 'avg_points_last_10', 'avg_points_allowed_last_10'], inplace=True)
    final_df['opponents_points_scored'] = final_df['PTS'] - final_df['PLUS_MINUS']
    final_df['Home_Game'] = final_df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
    final_df['Back_to_back'] = 0
    for team in final_df['TEAM_NAME'].unique():
        team_games = final_df[final_df['TEAM_NAME'] == team]
        prev_date = None
        for index, row in team_games.iterrows():
            if prev_date is not None and (row['GAME_DATE']-prev_date).days == 1:
                final_df.at[index, 'Back_to_back'] = 1
            prev_date = row['GAME_DATE']

    final_df['avg_points_season'] = round(final_df.groupby('TEAM_ID')['PTS'].transform(lambda x: x.rolling(window=82, min_periods=1).mean()), 3)
    final_df['avg_plus_minus_season'] = round(final_df.groupby('TEAM_ID')['PLUS_MINUS'].transform(lambda x: x.rolling(window=82, min_periods=1).mean()), 3)
    final_df['Win'] = final_df['WL'].apply(lambda x: 1 if x == 'W' else 0)
    final_df['Loss'] = final_df['WL'].apply(lambda x: 1 if x == 'L' else 0)
    final_df['cumulative_wins'] = final_df.groupby(['TEAM_ID', 'Season'])['Win'].cumsum()
    final_df['total_games'] = final_df.groupby(['TEAM_ID', 'Season']).cumcount() + 1
    final_df['Win_Percentage'] = round((final_df['cumulative_wins'] / final_df['total_games']) * 100, 3)
    final_df['TEAM_ABB'] = final_df['MATCHUP'].apply(lambda x: x.split(' ')[0])
    final_df['OPPONENT_TEAM_NAME'] = final_df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
    final_df.sort_values(by=['TEAM_NAME', 'OPPONENT_TEAM_NAME', 'GAME_DATE'], inplace=True)
    final_df['Rolling_Win'] = final_df.groupby(['TEAM_NAME', 'OPPONENT_TEAM_NAME'])['Win'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).sum())
    final_df['Rolling_Loss'] = final_df.groupby(['TEAM_NAME', 'OPPONENT_TEAM_NAME'])['Loss'].transform(lambda x: x.shift().rolling(window=10, min_periods=1).sum())
    final_df['Rolling_Win_Percentage'] = (final_df['Rolling_Win'] / (final_df['Rolling_Win'] + final_df['Rolling_Loss'])).round(3)
    final_df['Rolling_Win_Percentage'] = final_df['Rolling_Win_Percentage'].fillna(0.5)

    def get_opponent_feature(row, feature):
        opponent_data = final_df.loc[(final_df['TEAM_ABB'] == row['OPPONENT_TEAM_NAME']) & (final_df['GAME_DATE'] == row['GAME_DATE'])]
        if not opponent_data.empty:
            return opponent_data.iloc[-1][feature]
        else:
            return 0

    final_df['Opponent_Back_to_back'] = final_df.apply(get_opponent_feature, axis=1, feature='Back_to_back')
    final_df['Opponent_avg_plus_minus_last_10'] = final_df.apply(get_opponent_feature, axis=1, feature='avg_plus_minus_last_10')
    final_df['Opponent_Win_Percentage'] = final_df.apply(get_opponent_feature, axis=1, feature='Win_Percentage')
    final_df['Opponent_plus_minus_season'] = final_df.apply(get_opponent_feature, axis=1, feature='avg_plus_minus_season')
    final_df = final_df[['Opponent_plus_minus_season', 'Opponent_Win_Percentage', 'Opponent_avg_plus_minus_last_10', 'Opponent_Back_to_back', 'TEAM_ABB', 'OPPONENT_TEAM_NAME', 'Home_Game', 'GAME_DATE', 'Season', 'TEAM_NAME', 'MATCHUP', 'WL', 'Win', 'PTS', 'opponents_points_scored', 'PLUS_MINUS', 'avg_plus_minus_last_10', 'avg_points_last_10', 'avg_points_allowed_last_10', 'avg_points_season', 'avg_plus_minus_season', 'cumulative_wins', 'total_games', 'Win_Percentage', 'Back_to_back', 'Rolling_Win_Percentage']]
    final_df.sort_values(by=['GAME_DATE'], inplace=True)
    return final_df

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
    
    return jsonify({'message': 'Model trained successfully', 'accuracy': accuracy})

@app.route('/make_predictions', methods=['GET'])
def make_predictions():
    final_df = fetch_and_process_data()
    features = ['Opponent_plus_minus_season', 'Opponent_Win_Percentage', 'Opponent_avg_plus_minus_last_10', 'Opponent_Back_to_back', 'Back_to_back', 'Home_Game', 'avg_plus_minus_last_10', 'avg_points_last_10', 'avg_points_allowed_last_10', 'avg_points_season', 'avg_plus_minus_season', 'cumulative_wins', 'Win_Percentage', 'Rolling_Win_Percentage', 'total_games']
    X = final_df[features]
    model = joblib.load('logistic_regression_model.pkl')
    X = X.apply(pd.to_numeric, errors='coerce')
    predictions = model.predict(X)
    
    final_df['Predictions'] = predictions
    for index, row in final_df.iterrows():
        game_data = GameData(team_name=row['TEAM_NAME'], matchup=row['MATCHUP'], game_date=row['GAME_DATE'], prediction=str(predictions[index]))
        db.session.add(game_data)
    db.session.commit()
    
    predictions_html = final_df.to_html(classes='table table-striped', index=False)
    return render_template('predictions.html', predictions=predictions_html)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
'''

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
    


    
if __name__ == '__main__':
    app.run(debug=True, port = 8001)

#   @app.route('/')
# Function to fetch and process data
def fetch_and_process_data(seasons=['2021-22', '2022-23', '2023-24']):
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
        final_df = final_df[['Opponent_plus_minus_season', 'Opponent_Win_Percentage', 'Opponent_avg_plus_minus_last_10', 'Opponent_Back_to_back','TEAM_ABB', 'OPPONENT_TEAM_NAME','Home_Game', 'GAME_DATE','Season', 'TEAM_NAME', 'MATCHUP', 'WL','Win', 'PTS','opponents_points_scored', 'PLUS_MINUS',  'avg_plus_minus_last_10' , 'avg_points_last_10' ,'avg_points_allowed_last_10','avg_points_season','avg_plus_minus_season','cumulative_wins','total_games', 'Win_Percentage', 'Back_to_back', 'Rolling_Win_Percentage']]
        #making the games by game date instead of each team
        final_df.sort_values(by=['GAME_DATE'], inplace=True)
    except Exception as e:
        print(f"Error: {e}")
    
    return final_df

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

@app.route('/make_predictions', methods=['GET'])
def make_predictions():
    final_df = fetch_and_process_data()
    features = ['Opponent_plus_minus_season', 'Opponent_Win_Percentage', 'Opponent_avg_plus_minus_last_10', 'Opponent_Back_to_back', 'Back_to_back', 'Home_Game', 'avg_plus_minus_last_10', 'avg_points_last_10', 'avg_points_allowed_last_10', 'avg_points_season', 'avg_plus_minus_season', 'cumulative_wins', 'Win_Percentage', 'Rolling_Win_Percentage', 'total_games']
    X = final_df[features]
    model = joblib.load('logistic_regression_model.pkl')
    X = X.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, coerce errors to NaN
    predictions = model.predict(X)
    
    final_df['Predictions'] = predictions
    for index, row in final_df.iterrows():
        game_data = GameData(team_name=row['TEAM_NAME'], matchup=row['MATCHUP'], game_date=row['GAME_DATE'], prediction=str(predictions[index]))
        db.session.add(game_data)
    db.session.commit()
    predictions_html = predictions.to_html(classes='table table-striped', index=False)
    return render_template('predictions.html', predictions=predictions_html)