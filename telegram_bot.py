import os
import logging
import asyncio
import pytz
from datetime import datetime, time as dtime
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from dotenv import load_dotenv

import football_engine as engine
import tracker

# Load environment variables
load_dotenv()

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
BANKROLL       = float(os.environ.get("BANKROLL", "1000"))
TIMEZONE       = os.environ.get("TIMEZONE", "Africa/Lagos")
CHAT_ID        = os.environ.get("CHAT_ID", "")
MIN_ODDS       = float(os.environ.get("MIN_ODDS", "2.80"))
TZ             = pytz.timezone(TIMEZONE)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def fmt_draw_prediction(res):
    return f"""⚽ {res['home_team']} vs {res['away_team']}
{res['league']} — {datetime.fromisoformat(res['generated_at']).strftime('%I:%M %p %Z')}

📊 DRAW ANALYSIS
━━━━━━━━━━━━━━━━━━━━
🤖 Model draw prob  : {res['draw_prob']}%
📖 Book implied     : {res['implied_prob']}%
📐 Edge             : {res['edge']['edge_pct']}% {res['edge']['edge_emoji']} {res['edge']['edge_label']}

🎯 Confidence: {res['confidence']}% {res['conf_label']}
💰 Draw odds: {res['draw_odds']}
💵 Suggested stake: ${res['stake']} of ${BANKROLL}

📈 DRAW FACTORS
  Home draw rate (L10): {round(res['features']['home_draw_rate']*100)}%
  Away draw rate (L10): {round(res['features']['away_draw_rate']*100)}%
  H2H draw rate: {round(res['features']['h2h_draw_rate']*100)}%
  Goal expectancy: {res['features']['goal_expectancy']}
  Copa format: {'Yes' if res['features']['is_copa'] else 'No'}
  Altitude: {'High' if res['features']['altitude_factor'] > 0.5 else 'Low'}

⚠️ Verify draw odds at your sportsbook"""

def fmt_daily_report(preds, best_picks):
    msg = f"⚽ DRAW HUNTER — {datetime.now().strftime('%b %d, %Y')}\n━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"{len(preds)} matches analyzed\n{len(best_picks)} value draws found\n\n"
    msg += "🔥 VALUE DRAWS TODAY:\n\n"
    
    for i, p in enumerate(best_picks[:5], 1):
        msg += f"{i}. {p['home_team']} vs {p['away_team']}\n"
        msg += f"   {p['league']}\n"
        msg += f"   Draw: {p['draw_prob']}% | Edge: {p['edge']['edge_pct']}% | Conf: {p['confidence']}% {p['edge']['edge_emoji']}\n"
        msg += f"   Odds: {p['draw_odds']} | Stake: ${p['stake']}\n\n"
    
    if best_picks:
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"🏆 BEST PICK: {best_picks[0]['home_team']} vs {best_picks[0]['away_team']} — DRAW\n"
        msg += f"   Edge: {best_picks[0]['edge']['edge_pct']}% | Stake: ${best_picks[0]['stake']}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n⚠️ Always verify odds at Sportybet before betting"
    return msg

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """👋 Welcome to Draw Hunter Bot V1!
South American Football Draw Prediction Specialist.

Leagues Covered:
🇧🇷 Brazilian Serie A
🇦🇷 Argentine Primera División
🇨🇴 Colombian Liga BetPlay
🇨🇱 Chilean Primera División
🇺🇾 Uruguayan Primera División
🏆 Copa Libertadores & Sudamericana

Commands:
/today - Today's value picks
/match [home] vs [away] - Predict specific match
/record - All-time stats
/status - Model performance
/bankroll [amount] - Update bankroll
/retrain - Force model update

Schedule:
08:00 AM - Results fetch & model retrain
12:00 PM - Daily prediction report"""
    await update.message.reply_text(msg)

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("Fetching fixtures and running models...")
    fixtures = engine.get_todays_fixtures()
    if not fixtures:
        await status_msg.edit_text("No fixtures found for today.")
        return
    
    preds = []
    for f in fixtures:
        p = engine.predict_match(f['home_team'], f['away_team'], f['home_team_id'], f['away_team_id'], f['league_name'], bankroll=BANKROLL)
        preds.append(p)
    
    best = engine.find_best_draws(preds)
    await status_msg.edit_text(fmt_daily_report(preds, best))
    
    for p in best:
        tracker.log_draw_pick(f"{p['home_team']} vs {p['away_team']}", p['league'], p['draw_prob'], p['implied_prob'], p['edge']['edge_pct'], p['confidence'], p['draw_odds'], p['stake'])

async def match_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /match Team A vs Team B")
        return
    
    match_str = " ".join(context.args)
    if " vs " not in match_str.lower():
        await update.message.reply_text("Please use 'vs' to separate teams. Example: /match Flamengo vs Palmeiras")
        return
    
    home, away = [t.strip() for t in match_str.split(" vs ")]
    res = engine.predict_match(home, away, "0", "0", "Unknown", bankroll=BANKROLL)
    await update.message.reply_text(fmt_draw_prediction(res))

async def record(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = tracker.get_stats()
    await update.message.reply_text(tracker.format_stats_message(stats))

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import sqlite3
    conn = sqlite3.connect(engine.DB_PATH)
    mp = conn.execute("SELECT * FROM model_performance ORDER BY id DESC LIMIT 1").fetchone()
    count = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()
    
    if not mp:
        await update.message.reply_text(f"Model not trained yet. Matches in DB: {count}")
        return
    
    msg = f"""📊 DRAW HUNTER STATUS
━━━━━━━━━━━━━━━━━━━━
🎯 Matches in DB: {count}
📈 Model AUC-ROC: {mp[3]}
🎯 Draw Precision: {mp[5]}
🔄 Last Retrain: {mp[1][:16]}
━━━━━━━━━━━━━━━━━━━━"""
    await update.message.reply_text(msg)

async def set_bankroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global BANKROLL
    if not context.args:
        await update.message.reply_text(f"Current bankroll: ${BANKROLL}")
        return
    try:
        new_val = float(context.args[0])
        os.environ["BANKROLL"] = str(new_val)
        BANKROLL = new_val
        await update.message.reply_text(f"Bankroll updated to ${new_val}")
    except ValueError:
        await update.message.reply_text("Invalid amount.")

async def force_retrain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Starting daily retrain process. This will fetch 2 seasons of data...")
    engine.daily_retrain()
    await update.message.reply_text("Retrain complete. Model updated.")

async def my_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Your Chat ID: {update.effective_chat.id}")

async def daily_job_retrain(context: ContextTypes.DEFAULT_TYPE):
    engine.daily_retrain()
    if CHAT_ID:
        await context.bot.send_message(chat_id=CHAT_ID, text="✅ Daily retrain complete. Model updated with latest results.")

async def daily_job_predictions(context: ContextTypes.DEFAULT_TYPE):
    fixtures = engine.get_todays_fixtures()
    if not fixtures: return
    preds = []
    for f in fixtures:
        p = engine.predict_match(f['home_team'], f['away_team'], f['home_team_id'], f['away_team_id'], f['league_name'], bankroll=BANKROLL)
        preds.append(p)
    best = engine.find_best_draws(preds)
    if CHAT_ID:
        await context.bot.send_message(chat_id=CHAT_ID, text=fmt_daily_report(preds, best))
        for p in best:
            tracker.log_draw_pick(f"{p['home_team']} vs {p['away_team']}", p['league'], p['draw_prob'], p['implied_prob'], p['edge']['edge_pct'], p['confidence'], p['draw_odds'], p['stake'])

def main():
    if not TELEGRAM_TOKEN:
        print("Error: TELEGRAM_TOKEN not found in .env")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("today", today))
    application.add_handler(CommandHandler("match", match_predict))
    application.add_handler(CommandHandler("record", record))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("bankroll", set_bankroll))
    application.add_handler(CommandHandler("retrain", force_retrain))
    application.add_handler(CommandHandler("mychatid", my_chat_id))
    
    # Scheduler
    job_queue = application.job_queue
    job_queue.run_daily(daily_job_retrain, time=dtime(8, 0, tzinfo=TZ))
    job_queue.run_daily(daily_job_predictions, time=dtime(12, 0, tzinfo=TZ))
    
    print("✅ Draw Hunter V2 Started (Zero API Keys Mode)")
    application.run_polling()

if __name__ == "__main__":
    main()
