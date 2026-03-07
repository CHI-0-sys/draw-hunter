import sqlite3
import json
import logging
from datetime import datetime, timedelta

DB_PATH = "draw_hunter.db"
log = logging.getLogger(__name__)

def log_draw_pick(match_name, league, draw_prob, implied_prob,
                   edge_pct, confidence, draw_odds, stake):
    """Log draw pick to prediction_log."""
    conn = sqlite3.connect(DB_PATH)
    try:
        home, away = match_name.split(' vs ')
    except ValueError:
        home, away = match_name, ""
        
    conn.execute("""
        INSERT INTO prediction_log
        (created_at, match_date, league, match, home_team, away_team,
         draw_prob, book_implied, edge, confidence, draw_odds, stake, result)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(),
        datetime.now().strftime('%Y-%m-%d'),
        league, match_name, home, away,
        draw_prob, implied_prob, edge_pct, confidence, draw_odds, stake, 'PENDING'
    ))
    conn.commit()
    conn.close()

def settle_pick(pick_id, actual_result):
    """
    actual_result: 'H', 'D', or 'A'
    WIN if actual_result == 'D', else LOSS
    """
    conn = sqlite3.connect(DB_PATH)
    pick = conn.execute("SELECT draw_odds, stake FROM prediction_log WHERE id = ?", (pick_id,)).fetchone()
    if not pick:
        conn.close()
        return False
    
    draw_odds, stake = pick
    is_win = (actual_result == 'D')
    result_status = 'WIN' if is_win else 'LOSS'
    profit_loss = (stake * (draw_odds - 1)) if is_win else -stake
    
    conn.execute("""
        UPDATE prediction_log
        SET result = ?, actual_result = ?, profit_loss = ?, settled_at = ?
        WHERE id = ?
    """, (result_status, actual_result, profit_loss, datetime.now().isoformat(), pick_id))
    conn.commit()
    conn.close()
    return True

def get_stats(days=None, league=None, last_n=None) -> dict:
    """Return: total, wins, losses, win_rate, roi, total_profit, streak"""
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT result, profit_loss, stake, league FROM prediction_log WHERE result != 'PENDING'"
    params = []
    
    if days:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        query += " AND settled_at >= ?"
        params.append(cutoff)
    if league:
        query += " AND league = ?"
        params.append(league)
    
    query += " ORDER BY settled_at DESC"
    if last_n:
        query += f" LIMIT {last_n}"
        
    df = conn.execute(query, params).fetchall()
    conn.close()
    
    if not df:
        return {
            'total': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'roi': 0, 'total_profit': 0, 'streak': 0
        }
    
    wins = sum(1 for r in df if r[0] == 'WIN')
    total = len(df)
    total_profit = sum(r[1] for r in df)
    total_staked = sum(r[2] for r in df)
    
    streak = 0
    for r in df:
        if r[0] == 'WIN': streak += 1
        else: break
        
    return {
        'total': total,
        'wins': wins,
        'losses': total - wins,
        'win_rate': round(wins / total * 100, 1) if total > 0 else 0,
        'roi': round(total_profit / total_staked * 100, 1) if total_staked > 0 else 0,
        'total_profit': round(total_profit, 2),
        'streak': streak
    }

def get_pending_picks() -> list:
    conn = sqlite3.connect(DB_PATH)
    picks = conn.execute("SELECT id, match, league, draw_odds, stake FROM prediction_log WHERE result = 'PENDING'").fetchall()
    conn.close()
    return picks

def format_stats_message(stats, title="📊 DRAW HUNTER RECORD") -> str:
    return f"""{title}
━━━━━━━━━━━━━━━━━━━━
✅ Wins: {stats['wins']}
❌ Losses: {stats['losses']}
📈 Win Rate: {stats['win_rate']}%
💰 Total Profit: ${stats['total_profit']}
📊 ROI: {stats['roi']}%
🔥 Current Streak: {stats['streak']}
━━━━━━━━━━━━━━━━━━━━"""

def get_league_breakdown() -> dict:
    conn = sqlite3.connect(DB_PATH)
    leagues = conn.execute("SELECT DISTINCT league FROM prediction_log WHERE result != 'PENDING'").fetchall()
    breakdown = {}
    for (league,) in leagues:
        breakdown[league] = get_stats(league=league)
    conn.close()
    return breakdown
