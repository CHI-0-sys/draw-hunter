import requests
import io
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
MIN_CONF_BET  = 52.0    # minimum confidence to show any pick
MIN_EDGE_PCT  = 3.0     # minimum edge to flag as VALUE
LOOKBACK      = 10

# Tier thresholds
TIER_ELITE    = 10.0    # 🔥🔥 edge >= 10%
TIER_STRONG   = 7.0     # 🔥   edge >= 7%
TIER_VALUE    = 4.0     # ✅   edge >= 4%
TIER_LEAN     = 2.0     # ⚠️   edge >= 2%  (show but no stake)
TIER_INFO     = 0.0     # 📊   any positive edge (info only)
ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# ESPN League IDs for Soccer
ESPN_LEAGUES = {
    'bra.1': 'Brazilian Série A',
    'arg.1': 'Argentine Primera División',
    'col.1': 'Colombian Primera A',
    'chi.1': 'Chilean Primera División',
    'uru.1': 'Uruguayan Primera División',
    'lib':   'Copa Libertadores',
    'sud':   'Copa Sudamericana',
    'mex.1': 'Mexican Liga MX',
}

CSV_SOURCES = {
    'bra.1': 'https://www.football-data.co.uk/new/BRA.csv',
    'arg.1': 'https://www.football-data.co.uk/new/ARG.csv',
    'usa.1':  'https://www.football-data.co.uk/new/USA.csv',
}

ALTITUDE_MAP = {
    'bol.1': 1.0,   # High (La Paz)
    'ecu.1': 0.8,   # High (Quito)
    'col.1': 0.7,   # Med-High (Bogota)
    'per.1': 0.6,   # Med (Cuzco/Puno)
    'mex.1': 0.5,   # Med (CDMX)
    'bra.1': 0.1,   # Low
    'arg.1': 0.05,  # Low
    'chi.1': 0.1,   # Low
    'uru.1': 0.05,  # Low
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

def espn_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"ESPN Request Error: {e}")
        return {}

def store_team_history(records: list):
    conn = sqlite3.connect(DB_PATH)
    for r in records:
        conn.execute("""
            INSERT OR REPLACE INTO team_history 
            (record_id, league, team_id, team_name, match_date, goals_for, goals_against, result, is_draw, is_home, opponent)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (r['record_id'], r['league'], r['team_id'], r['team_name'], r['match_date'], r['goals_for'], r['goals_against'], r['result'], r['is_draw'], r['is_home'], r['opponent']))
    conn.commit()
    conn.close()

def get_h2h_draw_rate(home, away):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("""
        SELECT AVG(is_draw), COUNT(*) FROM matches 
        WHERE (home_team LIKE ? AND away_team LIKE ?) 
           OR (home_team LIKE ? AND away_team LIKE ?)
    """, (f'%{home}%', f'%{away}%', f'%{away}%', f'%{home}%')).fetchone()
    conn.close()
    if row and row[1] > 0:
        return round(row[0], 3), row[1]
    return 0.28, 0

def kelly_stake(prob, odds, bankroll):
    b = odds - 1
    p = prob / 100
    q = 1 - p
    kelly_f = (b * p - q) / b if b > 0 else 0
    return round(max(0, bankroll * kelly_f * 0.2), 2)

def init_db():
    os.makedirs(MODELS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY, league TEXT, season TEXT, match_date TEXT,
        home_team TEXT, away_team TEXT, home_goals INTEGER, away_goals INTEGER,
        result TEXT, is_draw INTEGER, total_goals INTEGER, 
        draw_odds REAL, home_odds REAL, away_odds REAL, source TEXT,
        home_draw_rate REAL, away_draw_rate REAL,
        home_goals_scored_avg REAL, away_goals_scored_avg REAL,
        home_goals_conceded_avg REAL, away_goals_conceded_avg REAL,
        home_win_rate REAL, away_win_rate REAL,
        home_form_pts INTEGER, away_form_pts INTEGER,
        home_last5_draws INTEGER, away_last5_draws INTEGER,
        combined_draw_rate REAL, goal_expectancy REAL, defensive_strength REAL,
        form_difference REAL, draw_rate_difference REAL,
        h2h_draw_rate REAL, h2h_total_games INTEGER,
        is_copa INTEGER, altitude_factor REAL, is_derby INTEGER, implied_draw_prob REAL
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fixtures (
        fixture_id TEXT PRIMARY KEY, league TEXT, league_name TEXT, match_date TEXT,
        home_team TEXT, away_team TEXT, home_team_id TEXT, away_team_id TEXT,
        country TEXT, status TEXT, draw_odds REAL, prediction_json TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, match_date TEXT,
        league TEXT, match_name TEXT, home_team TEXT, away_team TEXT,
        draw_prob REAL, book_implied REAL, edge REAL, confidence REAL,
        draw_odds REAL, stake REAL, result TEXT DEFAULT 'PENDING',
        actual_result TEXT, profit_loss REAL DEFAULT 0, settled_at TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS team_history (
        record_id TEXT PRIMARY KEY, league TEXT, team_id TEXT, team_name TEXT,
        match_date TEXT, goals_for INTEGER, goals_against INTEGER,
        result TEXT, is_draw INTEGER, is_home INTEGER, opponent TEXT
    )""")
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, 
        auc_roc REAL, brier_score REAL, draw_precision REAL
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

def get_todays_fixtures(timezone: str = "Africa/Lagos") -> list:
    """
    Fetch today's South American fixtures from ESPN free API.
    Tries today AND tomorrow to catch late-night kickoffs.
    No API key needed.
    """
    import pytz
    from datetime import datetime, timedelta

    TZ       = pytz.timezone(timezone)
    sa_tz    = pytz.timezone('America/Sao_Paulo')
    now_sa   = datetime.now(sa_tz)

    # Try today and tomorrow (covers early morning Africa = previous day SA)
    dates_to_try = [
        now_sa.strftime('%Y%m%d'),
        (now_sa + timedelta(days=1)).strftime('%Y%m%d'),
        (now_sa - timedelta(days=1)).strftime('%Y%m%d'),
    ]

    all_fixtures = []

    for league_code, league_name in ESPN_LEAGUES.items():
        for date_str in dates_to_try[:2]:  # today + tomorrow only
            try:
                url  = f"{ESPN_BASE}/{league_code}/scoreboard"
                data = espn_get(url, params={'dates': date_str, 'limit': 100})

                if not data or 'events' not in data:
                    continue

                for event in data.get('events', []):
                    try:
                        # Skip already completed
                        if event['status']['type'].get('completed', False):
                            continue

                        comps = event['competitions'][0]
                        competitors = comps.get('competitors', [])

                        if len(competitors) < 2:
                            continue

                        # Safe home/away extraction
                        home_c = next(
                            (t for t in competitors if t.get('homeAway') == 'home'),
                            competitors[0]
                        )
                        away_c = next(
                            (t for t in competitors if t.get('homeAway') == 'away'),
                            competitors[1]
                        )

                        home_name = home_c.get('team', {}).get('displayName', 'Home')
                        away_name = away_c.get('team', {}).get('displayName', 'Away')
                        home_id   = home_c.get('team', {}).get('id', '0')
                        away_id   = away_c.get('team', {}).get('id', '0')
                        home_abbr = home_c.get('team', {}).get('abbreviation',
                                    home_name[:3].upper())
                        away_abbr = away_c.get('team', {}).get('abbreviation',
                                    away_name[:3].upper())

                        # Convert kickoff to Africa local
                        time_local = event['status']['type'].get('shortDetail', 'TBD')
                        try:
                            start  = event.get('date', '')
                            if start:
                                from datetime import datetime as dt
                                utc_dt = dt.strptime(start[:16], '%Y-%m-%dT%H:%M')
                                utc_dt = pytz.utc.localize(utc_dt)
                                time_local = utc_dt.astimezone(TZ).strftime('%I:%M %p %Z')
                        except Exception:
                            pass

                        # Country from venue
                        venue   = comps.get('venue', {})
                        country = venue.get('address', {}).get('country', '')

                        # Avoid duplicates
                        fixture_id = str(event.get('id', f"{home_id}_{away_id}_{date_str}"))
                        if any(f['fixture_id'] == fixture_id for f in all_fixtures):
                            continue

                        all_fixtures.append({
                            'fixture_id':   fixture_id,
                            'league':       league_code,
                            'league_name':  league_name,
                            'match_date':   date_str[:4] + '-' + date_str[4:6] + '-' + date_str[6:],
                            'home_team':    home_name,
                            'away_team':    away_name,
                            'home_abbr':    home_abbr,
                            'away_abbr':    away_abbr,
                            'home_team_id': str(home_id),
                            'away_team_id': str(away_id),
                            'country':      country,
                            'status':       event['status']['type'].get('shortDetail', ''),
                            'time_local':   time_local,
                            'draw_odds':    None,
                        })

                    except Exception as e:
                        log.warning(f"Event parse error {league_code}: {e}")

            except Exception as e:
                log.error(f"ESPN fixtures error {league_code}: {e}")

    log.info(f"Total SA fixtures: {len(all_fixtures)}")
    return all_fixtures

def fetch_espn_team_history(team_id: str, league_code: str,
                              limit: int = 20) -> list:
    """
    Fetch completed match history for a team from ESPN.
    Tries the schedule endpoint — falls back to scoreboard search.
    """
    url  = f"{ESPN_BASE}/{league_code}/teams/{team_id}/schedule"
    data = espn_get(url)

    if not data or 'events' not in data:
        # Fallback: try without league code
        url2  = f"https://site.api.espn.com/apis/site/v2/sports/soccer/teams/{team_id}/schedule"
        data  = espn_get(url2)

    if not data or 'events' not in data:
        log.warning(f"No schedule data for team {team_id} in {league_code}")
        return []

    results = []
    for event in data.get('events', []):
        try:
            comp = event['competitions'][0]

            if not comp.get('status', {}).get('type', {}).get('completed', False):
                continue

            competitors = comp.get('competitors', [])
            if len(competitors) < 2:
                continue

            home_c = next((t for t in competitors if t.get('homeAway') == 'home'), competitors[0])
            away_c = next((t for t in competitors if t.get('homeAway') == 'away'), competitors[1])

            is_home = str(home_c.get('team', {}).get('id', '')) == str(team_id)
            us      = home_c if is_home else away_c
            them    = away_c if is_home else home_c

            # Score — handle missing
            try:
                gf = int(us.get('score', 0) or 0)
                ga = int(them.get('score', 0) or 0)
            except (ValueError, TypeError):
                continue

            if gf > ga:   result = 'W'
            elif gf < ga: result = 'L'
            else:         result = 'D'

            match_date = event.get('date', '')[:10]
            if not match_date:
                continue

            results.append({
                'record_id':      f"{event.get('id', '')}_{team_id}",
                'league':         league_code,
                'team_id':        str(team_id),
                'team_name':      us.get('team', {}).get('displayName', ''),
                'match_date':     match_date,
                'goals_for':      gf,
                'goals_against':  ga,
                'result':         result,
                'is_draw':        1 if result == 'D' else 0,
                'is_home':        1 if is_home else 0,
                'opponent':       them.get('team', {}).get('displayName', ''),
            })

            if len(results) >= limit:
                break

        except Exception as e:
            log.warning(f"History parse error team {team_id}: {e}")
            continue

    log.info(f"Team {team_id} ({league_code}): {len(results)} historical games")
    return results

def get_team_form(team_name: str, league_code: str,
                   lookback: int = LOOKBACK) -> dict:
    """
    Get rolling form stats.
    Priority: team_history DB → matches DB → ESPN live fetch → defaults
    """
    SA_DEFAULT = {
        'draw_rate': 0.29, 'goals_scored_avg': 1.2,
        'goals_conceded_avg': 1.1, 'win_rate': 0.33,
        'form_pts': 7, 'last5_draws': 1, 'games': 0,
    }

    conn = sqlite3.connect(DB_PATH)

    # Try team_history (ESPN data)
    rows = conn.execute("""
        SELECT goals_for, goals_against, result, is_draw, is_home
        FROM team_history
        WHERE (team_name LIKE ? OR team_name LIKE ?)
          AND league=?
        ORDER BY match_date DESC LIMIT ?
    """, (f'%{team_name}%', f'{team_name[:5]}%', league_code, lookback)).fetchall()

    # Try matches table (CSV data) if not enough
    if len(rows) < 3:
        rows_m = conn.execute("""
            SELECT
                CASE WHEN home_team LIKE ? THEN home_goals ELSE away_goals END,
                CASE WHEN home_team LIKE ? THEN away_goals ELSE home_goals END,
                result,
                is_draw,
                CASE WHEN home_team LIKE ? THEN 1 ELSE 0 END
            FROM matches
            WHERE (home_team LIKE ? OR away_team LIKE ?)
              AND league=?
            ORDER BY match_date DESC LIMIT ?
        """, (
            f'%{team_name}%', f'%{team_name}%', f'%{team_name}%',
            f'%{team_name}%', f'%{team_name}%',
            league_code, lookback
        )).fetchall()

        if len(rows_m) > len(rows):
            rows = rows_m

    conn.close()

    if not rows:
        log.info(f"No DB data for {team_name} — using SA averages")
        return SA_DEFAULT

    draws = wins = 0
    goals_scored   = []
    goals_conceded = []
    form_pts = last5_draws = 0

    for i, row in enumerate(rows):
        try:
            gf      = int(row[0] or 0)
            ga      = int(row[1] or 0)
            res     = str(row[2] or 'L').strip().upper()
            is_home = int(row[4] or 0)

            goals_scored.append(gf)
            goals_conceded.append(ga)

            # Result codes: team_history uses W/D/L, matches uses H/D/A
            is_win = (res == 'W') or (res == 'H' and is_home) or (res == 'A' and not is_home)
            is_draw_r = (res == 'D')

            if is_draw_r:
                draws += 1
                if i < 5:
                    form_pts    += 1
                    last5_draws += 1
            elif is_win:
                wins += 1
                if i < 5:
                    form_pts += 3

        except Exception as e:
            log.warning(f"Form row error: {e}")
            continue

    n = max(len(rows), 1)
    return {
        'draw_rate':           round(draws / n, 3),
        'goals_scored_avg':    round(np.mean(goals_scored), 2) if goals_scored else 1.2,
        'goals_conceded_avg':  round(np.mean(goals_conceded), 2) if goals_conceded else 1.1,
        'win_rate':            round(wins / n, 3),
        'form_pts':            form_pts,
        'last5_draws':         last5_draws,
        'games':               n,
    }

# ── Feature Engineering ────────────────────────────────────────────────
def build_features(home_team, away_team, league_code, country='', draw_odds=3.20):
    """
    Build feature set for draw prediction.
    """
    h_f = get_team_form(home_team, league_code)
    a_f = get_team_form(away_team, league_code)
    
    h2h_dr, h2h_n = get_h2h_draw_rate(home_team, away_team)
    
    # Context
    is_copa = 1 if 'lib' in league_code or 'sud' in league_code else 0
    altitude = 0.5 if country in ['Bolivia', 'Ecuador', 'Peru'] else 0.1
    is_derby = 0 # Could add lookup
    
    implied_draw_prob = round(1 / draw_odds, 4) if draw_odds and draw_odds > 1 else 0.30

    return {
        'home_draw_rate': h_f['draw_rate'], 
        'home_goals_scored_avg': h_f['goals_scored_avg'], 
        'home_goals_conceded_avg': h_f['goals_conceded_avg'],
        'home_win_rate': h_f['win_rate'], 
        'home_form_pts': h_f['form_pts'], 
        'home_last5_draws': h_f['last5_draws'],
        'away_draw_rate': a_f['draw_rate'], 
        'away_goals_scored_avg': a_f['goals_scored_avg'], 
        'away_goals_conceded_avg': a_f['goals_conceded_avg'],
        'away_win_rate': a_f['win_rate'], 
        'away_form_pts': a_f['form_pts'], 
        'away_last5_draws': a_f['last5_draws'],
        'combined_draw_rate': round((h_f['draw_rate'] + a_f['draw_rate']) / 2, 3),
        'goal_expectancy': round(h_f['goals_scored_avg'] + a_f['goals_scored_avg'], 2),
        'defensive_strength': round((h_f['goals_conceded_avg'] + a_f['goals_conceded_avg']) / 2, 2),
        'form_difference': h_f['form_pts'] - a_f['form_pts'],
        'draw_rate_difference': round(abs(h_f['draw_rate'] - a_f['draw_rate']), 3),
        'h2h_draw_rate': h2h_dr, 
        'h2h_total_games': h2h_n,
        'is_copa': is_copa, 
        'altitude_factor': altitude,
        'implied_draw_prob': implied_draw_prob,
        '_home_games': h_f['games'],
        '_away_games': a_f['games']
    }

def fetch_csv_training_data(league_code: str) -> pd.DataFrame:
    """
    Download free CSV from football-data.co.uk.
    Handles all column name variants and date formats.
    """
    url = CSV_SOURCES.get(league_code)
    if not url:
        log.warning(f"No CSV source for {league_code}")
        return pd.DataFrame()

    try:
        log.info(f"Downloading: {url}")
        r = requests.get(url, timeout=30,
                         headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()

        # Try multiple encodings
        for enc in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(
                    io.StringIO(r.content.decode(enc)),
                    on_bad_lines='skip'
                )
                break
            except Exception:
                continue
        else:
            log.error(f"Could not decode CSV for {league_code}")
            return pd.DataFrame()

        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        log.info(f"CSV columns found: {list(df.columns)}")

        # Normalize column names — map variants to standard names
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ('date',):                          col_map[c] = 'Date'
            elif cl in ('hometeam', 'home team', 'home'): col_map[c] = 'HomeTeam'
            elif cl in ('awayteam', 'away team', 'away'): col_map[c] = 'AwayTeam'
            elif cl in ('fthg', 'hg', 'home goals', 'homegoals'): col_map[c] = 'FTHG'
            elif cl in ('ftag', 'ag', 'away goals', 'awaygoals'): col_map[c] = 'FTAG'
            elif cl in ('ftr', 'res', 'result'):          col_map[c] = 'FTR'
            elif cl in ('b365d', 'bbd', 'drawodds'):      col_map[c] = 'B365D'
            elif cl in ('b365h',):                        col_map[c] = 'B365H'
            elif cl in ('b365a',):                        col_map[c] = 'B365A'

        df = df.rename(columns=col_map)

        # Must have these columns
        required = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            log.error(f"CSV {league_code} missing columns: {missing}")
            log.error(f"Available: {list(df.columns)}")
            return pd.DataFrame()

        # Filter completed matches
        df = df[df['FTR'].isin(['H', 'D', 'A'])].copy()
        df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
        df = df[df['HomeTeam'].str.strip() != '']
        df = df[df['AwayTeam'].str.strip() != '']

        log.info(f"CSV {league_code}: {len(df)} valid rows after filtering")
        return df

    except Exception as e:
        log.error(f"CSV download error {league_code}: {e}")
        return pd.DataFrame()

def store_csv_matches(df: pd.DataFrame, league_code: str):
    """
    Store CSV matches with rolling features.
    Builds running team history from the CSV itself — no DB needed first.
    """
    if df.empty:
        log.warning(f"Empty dataframe for {league_code}")
        return

    conn    = sqlite3.connect(DB_PATH)
    stored  = skipped = 0

    # Running history cache — built as we go through sorted rows
    team_cache = {}   # team_name -> list of {gf, ga, is_draw, win}

    # Parse and sort by date first
    rows_parsed = []
    for _, row in df.iterrows():
        try:
            raw_date  = str(row.get('Date', '')).strip()
            home_team = str(row.get('HomeTeam', '')).strip()
            away_team = str(row.get('AwayTeam', '')).strip()
            result    = str(row.get('FTR', '')).strip().upper()

            if result not in ('H', 'D', 'A'):
                continue
            if not home_team or not away_team:
                continue

            # Parse date — handles DD/MM/YY and DD/MM/YYYY
            match_date = None
            for fmt in ('%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y'):
                try:
                    from datetime import datetime as _dt
                    match_date = _dt.strptime(raw_date, fmt).strftime('%Y-%m-%d')
                    break
                except Exception:
                    continue

            if not match_date:
                skipped += 1
                continue

            hg = int(float(row.get('FTHG', 0) or 0))
            ag = int(float(row.get('FTAG', 0) or 0))

            draw_odds = home_odds = away_odds = None
            try:
                v = row.get('B365D')
                draw_odds = float(v) if v and str(v).strip() not in ('', 'nan') else None
                v = row.get('B365H')
                home_odds = float(v) if v and str(v).strip() not in ('', 'nan') else None
                v = row.get('B365A')
                away_odds = float(v) if v and str(v).strip() not in ('', 'nan') else None
            except Exception:
                pass

            rows_parsed.append({
                'date': match_date, 'home': home_team, 'away': away_team,
                'hg': hg, 'ag': ag, 'result': result,
                'draw_odds': draw_odds, 'home_odds': home_odds,
                'away_odds': away_odds,
            })
        except Exception as e:
            log.warning(f"Row parse error: {e}")
            skipped += 1

    # Sort chronologically
    rows_parsed.sort(key=lambda x: x['date'])
    log.info(f"CSV {league_code}: {len(rows_parsed)} rows to store")

    def get_rolling(team):
        """Compute rolling stats from cache."""
        hist = team_cache.get(team, [])
        if not hist:
            return {
                'draw_rate': 0.29, 'goals_scored_avg': 1.2,
                'goals_conceded_avg': 1.1, 'win_rate': 0.33,
                'form_pts': 7, 'last5_draws': 1,
            }
        recent = hist[-LOOKBACK:]
        n = len(recent)
        draws = wins = 0
        gf_list = []
        ga_list = []
        form_pts = last5_draws = 0

        for i, g in enumerate(reversed(recent)):
            gf_list.append(g['gf'])
            ga_list.append(g['ga'])
            if g['is_draw']:
                draws += 1
                if i < 5:
                    form_pts    += 1
                    last5_draws += 1
            elif g['win']:
                wins += 1
                if i < 5:
                    form_pts += 3

        return {
            'draw_rate':           round(draws / n, 3),
            'goals_scored_avg':    round(float(np.mean(gf_list)), 2),
            'goals_conceded_avg':  round(float(np.mean(ga_list)), 2),
            'win_rate':            round(wins / n, 3),
            'form_pts':            form_pts,
            'last5_draws':         last5_draws,
        }

    # Derby detection
    derby_pairs = [
        ('Flamengo','Fluminense'), ('Flamengo','Vasco'),
        ('Boca','River'), ('Nacional','Peñarol'),
        ('Colo-Colo','Universidad'),
        ('Arsenal','Chelsea'), ('Arsenal','Tottenham'),
        ('Liverpool','Everton'), ('Manchester','Manchester'),
        ('Milan','Inter'), ('Juventus','Torino'),
        ('Barcelona','Espanyol'), ('Real Madrid','Atletico'),
        ('Galatasaray','Fenerbahce'),
    ]

    for r in rows_parsed:
        try:
            hf = get_rolling(r['home'])
            af = get_rolling(r['away'])

            is_copa  = 1 if 'libertadores' in league_code or 'sudamericana' in league_code else 0
            country  = ALTITUDE_MAP.get(league_code, None)
            altitude = 0.1

            is_derby = 0
            for t1, t2 in derby_pairs:
                if ((t1.lower() in r['home'].lower() and t2.lower() in r['away'].lower()) or
                    (t2.lower() in r['home'].lower() and t1.lower() in r['away'].lower())):
                    is_derby = 1
                    break

            implied = round(1 / r['draw_odds'], 4) if r['draw_odds'] and r['draw_odds'] > 1 else 0.30

            mid = f"{league_code}_{r['date']}_{r['home']}_{r['away']}"
            mid = mid.replace(' ', '_')[:120]

            conn.execute("""
                INSERT OR REPLACE INTO matches (
                    match_id, league, season, match_date,
                    home_team, away_team, home_goals, away_goals,
                    result, is_draw, total_goals,
                    draw_odds, home_odds, away_odds, source,
                    home_draw_rate, away_draw_rate,
                    home_goals_scored_avg, away_goals_scored_avg,
                    home_goals_conceded_avg, away_goals_conceded_avg,
                    home_win_rate, away_win_rate,
                    home_form_pts, away_form_pts,
                    home_last5_draws, away_last5_draws,
                    combined_draw_rate, goal_expectancy,
                    defensive_strength, form_difference,
                    draw_rate_difference, h2h_draw_rate,
                    h2h_total_games, is_copa, altitude_factor,
                    is_derby, implied_draw_prob
                ) VALUES (
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                    ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                )
            """, (
                mid, league_code, r['date'][:4], r['date'],
                r['home'], r['away'], r['hg'], r['ag'],
                r['result'], 1 if r['result']=='D' else 0,
                r['hg'] + r['ag'],
                r['draw_odds'], r['home_odds'], r['away_odds'],
                'football-data.co.uk',
                hf['draw_rate'], af['draw_rate'],
                hf['goals_scored_avg'], af['goals_scored_avg'],
                hf['goals_conceded_avg'], af['goals_conceded_avg'],
                hf['win_rate'], af['win_rate'],
                hf['form_pts'], af['form_pts'],
                hf['last5_draws'], af['last5_draws'],
                round((hf['draw_rate'] + af['draw_rate']) / 2, 3),
                round(hf['goals_scored_avg'] + af['goals_scored_avg'], 2),
                round((hf['goals_conceded_avg'] + af['goals_conceded_avg']) / 2, 2),
                hf['form_pts'] - af['form_pts'],
                round(abs(hf['draw_rate'] - af['draw_rate']), 3),
                0.29, 0,   # h2h defaults — computed at predict time
                is_copa, altitude, is_derby, implied
            ))

            # Update running cache AFTER insert — no data leakage
            for team, gf, ga, home_flag in [
                (r['home'], r['hg'], r['ag'], True),
                (r['away'], r['ag'], r['hg'], False),
            ]:
                win = (r['result']=='H' and home_flag) or (r['result']=='A' and not home_flag)
                if team not in team_cache:
                    team_cache[team] = []
                team_cache[team].append({
                    'gf': gf, 'ga': ga,
                    'is_draw': r['result']=='D',
                    'win': win,
                })

            stored += 1
            if stored % 200 == 0:
                conn.commit()
                log.info(f"  {league_code}: stored {stored} so far...")

        except Exception as e:
            log.warning(f"Store row error: {e}")
            skipped += 1

    conn.commit()
    conn.close()
    log.info(f"CSV {league_code}: ✅ stored={stored} skipped={skipped}")

def predict_draw_prob(features: dict) -> tuple:
    """
    Predict draw probability.
    Uses trained model if available.
    Falls back to weighted statistical estimate if model not trained yet.
    Always returns a real number — never crashes.
    """
    try:
        model     = joblib.load(f'{MODELS_DIR}/draw_model.pkl')
        scaler    = joblib.load(f'{MODELS_DIR}/draw_scaler.pkl')
        feat_cols = joblib.load(f'{MODELS_DIR}/draw_features.pkl')

        X   = np.array([[float(features.get(c, 0) or 0) for c in feat_cols]])
        X_s = scaler.transform(X)
        prob = float(model.predict_proba(X_s)[0][1])

        distance   = abs(prob - 0.50)
        g_factor   = min(1.0, (features.get('_home_games', 5) +
                                features.get('_away_games', 5)) / 16)
        confidence = min(MAX_CONF, round(
            (distance * 100 * 0.65 + prob * 100 * 0.35) * max(g_factor, 0.4), 1
        ))
        return round(prob * 100, 1), max(45.0, confidence)

    except FileNotFoundError:
        # Model not trained yet — use weighted average of draw signals
        log.info("Model not trained — using statistical fallback")
        cdr  = float(features.get('combined_draw_rate', 0.29) or 0.29)
        h2h  = float(features.get('h2h_draw_rate', 0.29) or 0.29)
        imp  = float(features.get('implied_draw_prob', 0.30) or 0.30)
        goal = float(features.get('goal_expectancy', 2.2) or 2.2)

        # Low scoring = more draws
        goal_bonus = max(0, (2.5 - goal) * 0.04)

        # Weighted average
        prob = round((cdr * 0.4 + h2h * 0.3 + imp * 0.2 + 0.29 * 0.1 + goal_bonus) * 100, 1)
        prob = max(15.0, min(60.0, prob))
        return prob, 52.0

    except Exception as e:
        log.error(f"predict_draw_prob error: {e}")
        return 29.0, 50.0

def predict_match(fixture: dict, bankroll: float = 1000) -> dict:
    """
    Full draw prediction for one fixture.
    Never crashes — always returns a result dict.
    """
    home    = fixture.get('home_team', 'Home')
    away    = fixture.get('away_team', 'Away')
    league  = fixture.get('league', 'bra.1')
    country = fixture.get('country', '')

    # Auto-fetch ESPN history if not in DB
    home_form_check = get_team_form(home, league)
    away_form_check = get_team_form(away, league)

    if home_form_check['games'] < 3:
        log.info(f"Fetching ESPN history: {home}")
        try:
            recs = fetch_espn_team_history(fixture.get('home_team_id', '0'), league, 20)
            if recs:
                store_team_history(recs)
        except Exception as e:
            log.warning(f"ESPN fetch failed for {home}: {e}")

    if away_form_check['games'] < 3:
        log.info(f"Fetching ESPN history: {away}")
        try:
            recs = fetch_espn_team_history(fixture.get('away_team_id', '0'), league, 20)
            if recs:
                store_team_history(recs)
        except Exception as e:
            log.warning(f"ESPN fetch failed for {away}: {e}")

    # Build features
    draw_odds = fixture.get('draw_odds') or 3.20
    try:
        features = build_features(home, away, league, country, draw_odds)
    except Exception as e:
        log.error(f"build_features error: {e}")
        features = {c: 0.0 for c in FEATURE_COLUMNS}
        features['implied_draw_prob'] = round(1 / draw_odds, 4)
        features['combined_draw_rate'] = 0.29

    # Predict
    try:
        draw_prob, confidence = predict_draw_prob(features)
    except Exception as e:
        log.error(f"predict error: {e}")
        draw_prob, confidence = 29.0, 50.0

    # Edge calculation
    implied_prob = round(1 / max(draw_odds, 1.01), 4)
    edge_pct     = round((draw_prob / 100) - implied_prob, 4) * 100
    has_value    = edge_pct >= MIN_EDGE_PCT and draw_prob >= MIN_CONF_BET

    if edge_pct >= TIER_ELITE:
        edge_label = "🔥🔥 ELITE VALUE"
        has_value  = True
    elif edge_pct >= TIER_STRONG:
        edge_label = "🔥 STRONG VALUE"
        has_value  = True
    elif edge_pct >= TIER_VALUE:
        edge_label = "✅ GOOD VALUE"
        has_value  = True
    elif edge_pct >= TIER_LEAN:
        edge_label = "⚠️ LEAN VALUE"
        has_value  = confidence >= 50.0   # lean picks only if decent confidence
    elif edge_pct >= TIER_INFO:
        edge_label = "📊 MARGINAL"
        has_value  = False
    else:
        edge_label = "❌ NO VALUE"
        has_value  = False

    # Stake — only on value picks with real confidence
    stake = 0.0
    if has_value and edge_pct >= TIER_VALUE and confidence >= MIN_CONF_BET:
        stake = kelly_stake(draw_prob, draw_odds, bankroll)

    # Data quality note
    home_games = features.get('_home_games', 0)
    away_games = features.get('_away_games', 0)
    if home_games == 0 and away_games == 0:
        data_note = "⚠️ No team history — using SA averages. Run /retrain for better predictions."
    elif home_games < 3 or away_games < 3:
        data_note = f"⚠️ Limited data ({home_games}/{away_games} games). Run /retrain for better data."
    else:
        data_note = ""

    result = {
        **fixture,
        'draw_prob':    draw_prob,
        'confidence':   confidence,
        'draw_odds':    draw_odds,
        'implied_prob': round(implied_prob * 100, 1),
        'edge_pct':     round(edge_pct, 2),
        'edge_label':   edge_label,
        'has_value':    has_value,
        'stake':        stake,
        'data_note':    data_note,
        'features':     {k: v for k, v in features.items() if not k.startswith('_')},
        'generated_at': datetime.now().isoformat(),
    }

    # Save to DB
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO fixtures
            (fixture_id, league, league_name, match_date, home_team, away_team,
             home_team_id, away_team_id, country, status, draw_odds, prediction_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            fixture.get('fixture_id', f"{home}_{away}"),
            league, fixture.get('league_name', ''),
            fixture.get('match_date', ''), home, away,
            fixture.get('home_team_id', ''), fixture.get('away_team_id', ''),
            country, fixture.get('status', ''), draw_odds, json.dumps(result)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"Could not save prediction: {e}")

    return result

def train_draw_model():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM matches WHERE combined_draw_rate IS NOT NULL", conn)
    conn.close()
    
    if len(df) < 100: 
        log.warning(f"Not enough training data: {len(df)}/100")
        return False, None
    
    y = df['is_draw'].values
    X = df[FEATURE_COLUMNS].values
    
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, objective='binary:logistic')
    model.fit(X_train, y_train)
    
    # Eval
    from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))
    prec = float(precision_score(y_test, (y_prob > 0.35).astype(int)))
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO model_performance (date, auc_roc, brier_score, draw_precision) VALUES (?,?,?,?)",
                 (datetime.now().isoformat(), auc, brier, prec))
    conn.commit()
    conn.close()
    
    joblib.dump(model, f"{MODELS_DIR}/draw_model.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/draw_scaler.pkl")
    joblib.dump(FEATURE_COLUMNS, f"{MODELS_DIR}/draw_features.pkl")
    return True, model

def find_best_draws(predictions: list) -> dict:
    """
    Rank predictions into tiers.
    Returns dict with tier lists — not just a flat list.
    """
    elite   = []
    strong  = []
    value   = []
    lean    = []
    info    = []

    for p in predictions:
        ep = p.get('edge_pct', 0)
        dp = p.get('draw_prob', 0)
        cf = p.get('confidence', 0)

        # Skip very low draw probability regardless of edge
        if dp < 20:
            continue

        score = round(ep * (dp / 100) * (cf / 100), 4)
        p['_score'] = score

        if ep >= TIER_ELITE and cf >= 50:
            elite.append(p)
        elif ep >= TIER_STRONG and cf >= 50:
            strong.append(p)
        elif ep >= TIER_VALUE and cf >= 48:
            value.append(p)
        elif ep >= TIER_LEAN and cf >= 46:
            lean.append(p)
        elif ep >= TIER_INFO and dp >= 28:
            info.append(p)

    for tier in (elite, strong, value, lean, info):
        tier.sort(key=lambda x: x.get('_score', 0), reverse=True)

    return {
        'elite':  elite,
        'strong': strong,
        'value':  value,
        'lean':   lean,
        'info':   info,
        'all_value': elite + strong + value,
    }

def daily_retrain():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    for lid in ESPN_LEAGUES:
        # Fetch last 30 days to keep DB fresh
        for i in range(-30, 1):
            data = espn_fetch(lid, 'scoreboard', days_offset=i)
            matches = process_espn_matches(lid, data)
            for m in matches:
                conn.execute("INSERT OR REPLACE INTO matches (match_id, league, season, match_date, home_team, away_team, home_goals, away_goals, result, is_draw, total_goals, draw_odds) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", 
                             (m['match_id'], m['league'], m['season'], m['match_date'], 
                              m['home_team'], m['away_team'], m['home_goals'], m['away_goals'], 
                              m['result'], m['is_draw'], m['total_goals'], 3.20))
    conn.commit()
    conn.close()
    train_draw_model()
    log.info("V2 Retrain complete (Zero Keys).")

if __name__ == "__main__":
    init_db()
