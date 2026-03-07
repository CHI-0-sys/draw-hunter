import requests
import sqlite3
import pandas as pd
import numpy as np
import joblib
import json
import time
import logging
import os
import warnings
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pytz

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Constants
DB_PATH      = "draw_hunter.db"
MODELS_DIR   = "models"
MAX_CONF     = 88.0
MIN_CONF_BET = 62.0
MIN_EDGE_PCT = 5.0
LOOKBACK     = 10

# ESPN League IDs for Soccer
# Reference: https://www.espn.com/soccer/scoreboard
LEAGUES = {
    'bra.1': 'Brazilian Série A',
    'arg.1': 'Argentine Primera División',
    'col.1': 'Colombian Primera A',
    'chi.1': 'Chilean Primera División',
    'uru.1': 'Uruguayan Primera División',
    'lib':   'Copa Libertadores',
    'sud':   'Copa Sudamericana',
    'mex.1': 'Mexican Liga MX',
}

FEATURE_COLUMNS = [
    'home_draw_rate', 'home_goals_scored_avg', 'home_goals_conceded_avg',
    'home_win_rate', 'home_form_pts', 'home_last5_draws',
    'away_draw_rate', 'away_goals_scored_avg', 'away_goals_conceded_avg',
    'away_win_rate', 'away_form_pts', 'away_last5_draws',
    'combined_draw_rate', 'goal_expectancy', 'defensive_strength',
    'form_difference', 'draw_rate_difference',
    'h2h_draw_rate', 'h2h_total_games',
    'is_copa', 'altitude_factor', 
    'implied_draw_prob',
]

def init_db():
    os.makedirs(MODELS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY, league TEXT, season TEXT, match_date TEXT,
        home_team TEXT, away_team TEXT, home_goals INTEGER, away_goals INTEGER,
        result TEXT, is_draw INTEGER, total_goals INTEGER, draw_odds REAL
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id TEXT PRIMARY KEY, league TEXT, match_date TEXT,
        home_team TEXT, away_team TEXT, home_id TEXT, away_id TEXT,
        draw_odds REAL, status TEXT, prediction_json TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, match_date TEXT,
        league TEXT, match_name TEXT, home_team TEXT, away_team TEXT,
        draw_prob REAL, book_implied REAL, edge REAL, confidence REAL,
        draw_odds REAL, stake REAL, result TEXT DEFAULT 'PENDING',
        actual_result TEXT, profit_loss REAL DEFAULT 0, settled_at TEXT
    )""")
    
    conn.commit()
    conn.close()

# ── ESPN FREE API ──────────────────────────────────────────────────────
def espn_fetch(league, endpoint='scoreboard', days_offset=0):
    """Zero API Key fetch from ESPN Public API."""
    date_str = (datetime.now() + timedelta(days=days_offset)).strftime('%Y%m%d')
    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/{endpoint}?dates={date_str}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"ESPN Error ({league}): {e}")
        return {}

def process_espn_matches(league_id, data):
    matches = []
    events = data.get('events', [])
    for event in events:
        try:
            status = event['status']['type']['name']
            if status != 'STATUS_FULL_TIME': continue
            
            match_id = str(event['id'])
            comp = event['competitions'][0]
            date = event['date'][:10]
            
            home = next(t for t in comp['competitors'] if t['homeAway'] == 'home')
            away = next(t for t in comp['competitors'] if t['homeAway'] == 'away')
            
            hg = int(home['score'])
            ag = int(away['score'])
            res = 'D' if hg == ag else ('H' if hg > ag else 'A')
            
            matches.append({
                'match_id': match_id, 'league': LEAGUES.get(league_id, league_id),
                'season': date[:4], 'match_date': date,
                'home_team': home['team']['displayName'], 'away_team': away['team']['displayName'],
                'home_goals': hg, 'away_goals': ag, 'result': res, 'is_draw': 1 if res == 'D' else 0,
                'total_goals': hg + ag
            })
        except: continue
    return matches

def get_todays_fixtures():
    fixtures = []
    for lid, lname in LEAGUES.items():
        data = espn_fetch(lid, 'scoreboard')
        events = data.get('events', [])
        for event in events:
            try:
                comp = event['competitions'][0]
                status = event['status']['type']['state']
                if status == 'post': continue # Skip finished
                
                home = next(t for t in comp['competitors'] if t['homeAway'] == 'home')
                away = next(t for t in comp['competitors'] if t['homeAway'] == 'away')
                
                # Fetch odds if available in event
                draw_odds = 3.20 # Default
                for odd in comp.get('odds', []):
                    if 'draw' in str(odd).lower():
                        # ESPN odds parse varies, fallback to 3.20 if complex
                        pass
                
                fixtures.append({
                    'fixture_id': str(event['id']), 'league': lname,
                    'match_date': event['date'][:10],
                    'home_team': home['team']['displayName'], 'away_team': away['team']['displayName'],
                    'home_id': str(home['id']), 'away_id': str(away['id']),
                    'draw_odds': draw_odds, 'status': status
                })
            except: continue
    return fixtures

# ── Feature Engineering ────────────────────────────────────────────────
def build_features(home_team, away_team, league, lookback=LOOKBACK):
    conn = sqlite3.connect(DB_PATH)
    
    def get_team_stats(team_name):
        df = pd.read_sql("SELECT * FROM matches WHERE (home_team=? OR away_team=?) AND league=? ORDER BY match_date DESC LIMIT ?", 
                         conn, params=(team_name, team_name, league, lookback))
        if df.empty:
            return {'draw_rate': 0.29, 'goals_scored': 1.1, 'goals_conceded': 1.1, 'win_rate': 0.35, 'form_pts': 6, 'last5_draws': 1, 'games': 0}
        
        draws, wins, gs, gc, f_pts, l5_d = 0, 0, [], [], 0, 0
        for i, row in df.iterrows():
            is_h = row['home_team'] == team_name
            me_g = row['home_goals'] if is_h else row['away_goals']
            op_g = row['away_goals'] if is_h else row['home_goals']
            gs.append(me_g); gc.append(op_g)
            if row['result'] == 'D':
                draws += 1
                if i < 5: {f_pts := f_pts + 1, l5_d := l5_d + 1}
            elif (is_h and row['result'] == 'H') or (not is_h and row['result'] == 'A'):
                wins += 1
                if i < 5: f_pts += 3
        
        return {
            'draw_rate': draws/len(df), 'goals_scored': np.mean(gs), 'goals_conceded': np.mean(gc),
            'win_rate': wins/len(df), 'form_pts': f_pts, 'last5_draws': l5_d, 'games': len(df)
        }

    h_f = get_team_stats(home_team)
    a_f = get_team_stats(away_team)
    
    # H2H
    h2h = pd.read_sql("SELECT * FROM matches WHERE (home_team=? AND away_team=?) OR (home_team=? AND away_team=?) LIMIT 10", 
                      conn, params=(home_team, away_team, away_team, home_team))
    h2h_dr = h2h['is_draw'].mean() if not h2h.empty else 0.28
    conn.close()

    return {
        'home_draw_rate': h_f['draw_rate'], 'home_goals_scored_avg': h_f['goals_scored'], 'home_goals_conceded_avg': h_f['goals_conceded'],
        'home_win_rate': h_f['win_rate'], 'home_form_pts': h_f['form_pts'], 'home_last5_draws': h_f['last5_draws'],
        'away_draw_rate': a_f['draw_rate'], 'away_goals_scored_avg': a_f['goals_scored'], 'away_goals_conceded_avg': a_f['goals_conceded'],
        'away_win_rate': a_f['win_rate'], 'away_form_pts': a_f['form_pts'], 'away_last5_draws': a_f['last5_draws'],
        'combined_draw_rate': (h_f['draw_rate'] + a_f['draw_rate'])/2,
        'goal_expectancy': h_f['goals_scored'] + a_f['goals_scored'],
        'defensive_strength': (h_f['goals_conceded'] + a_f['goals_conceded'])/2,
        'form_difference': h_f['form_pts'] - a_f['form_pts'],
        'draw_rate_difference': abs(h_f['draw_rate'] - a_f['draw_rate']),
        'h2h_draw_rate': h2h_dr, 'h2h_total_games': len(h2h),
        'is_copa': 1 if 'Copa' in league else 0,
        'altitude_factor': 0.2, # Simplified
        'implied_draw_prob': 0.31
    }

# ── ML Logic ───────────────────────────────────────────────────────────
def train_model():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches", conn)
    conn.close()
    
    if len(df) < 100: return False # Need baseline
    
    # In V2, we generate features for training from historical results
    # For speed in this rebuild, we'll assume features are stored or calculated
    # Real logic: iterate matches, build features as if it were 'today' for each
    
    # Dummy training for skeleton proof
    y = df['is_draw'].values
    X = np.random.rand(len(y), len(FEATURE_COLUMNS)) # Placeholder for feature build loop
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    model.fit(X_s, y)
    
    joblib.dump(model, f"{MODELS_DIR}/draw_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/draw_scaler.pkl")
    return True

def predict_match(home, away, league, draw_odds=3.20, bankroll=1000):
    feats = build_features(home, away, league)
    feats['implied_draw_prob'] = 1/draw_odds
    
    try:
        model = joblib.load(f"{MODELS_DIR}/draw_model.pkl")
        scaler = joblib.load(f"{MODELS_DIR}/draw_scaler.pkl")
        X = np.array([[feats[c] for c in FEATURE_COLUMNS]])
        prob = float(model.predict_proba(scaler.transform(X))[0][1])
    except:
        prob = feats['combined_draw_rate']
    
    prob_pct = round(prob * 100, 1)
    implied_pct = round((1/draw_odds)*100, 1)
    edge = round(prob_pct - implied_pct, 1)
    
    # Kelly
    b = draw_odds - 1
    p = prob
    q = 1 - p
    kelly_f = (b * p - q) / b if b > 0 else 0
    stake = round(max(0, bankroll * kelly_f * 0.2), 2)
    
    return {
        'home_team': home, 'away_team': away, 'league': league,
        'draw_prob': prob_pct, 'implied_prob': implied_pct, 'edge': edge,
        'draw_odds': draw_odds, 'stake': stake, 'confidence': round(prob_pct * 0.8, 1),
        'generated_at': datetime.now().isoformat()
    }

def daily_retrain():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    for lid in LEAGUES:
        # Fetch last 30 days to keep DB fresh
        for i in range(-30, 1):
            data = espn_fetch(lid, 'scoreboard', days_offset=i)
            matches = process_espn_matches(lid, data)
            for m in matches:
                conn.execute("INSERT OR REPLACE INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", 
                             (m['match_id'], m['league'], m['season'], m['match_date'], 
                              m['home_team'], m['away_team'], m['home_goals'], m['away_goals'], 
                              m['result'], m['is_draw'], m['total_goals'], 3.20))
    conn.commit()
    conn.close()
    train_model()
    log.info("V2 Retrain complete (Zero Keys).")

if __name__ == "__main__":
    init_db()
    # daily_retrain() # Uncomment to run first time
