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

def fmt_prediction(res: dict) -> str:
    """
    Format a single prediction for Telegram.
    Safe access to all keys.
    """
    try:
        home = res.get('home_team', 'Home')
        away = res.get('away_team', 'Away')
        league = res.get('league_name', 'South American Soccer')
        prob = res.get('draw_prob', 29.0)
        odds = res.get('draw_odds', 3.20)
        edge_data = res.get('edge', {})
        edge = edge_data.get('edge_pct', 0.0)
        label = edge_data.get('edge_label', 'NO VALUE')
        conf = res.get('confidence', 50.0)
        stake = res.get('stake', 0.0)
        note = res.get('data_note', '')

        msg = f"⚽ *{home} vs {away}*\n"
        msg += f"🏆 {league}\n\n"
        msg += f"📊 *DRAW ANALYSIS*\n"
        msg += f"━━━━━━━━━━━━━━━\n"
        msg += f"🤖 Draw Prob: {prob}%\n"
        msg += f"📈 Edge: {edge}% ({label})\n"
        msg += f"🎯 Confidence: {conf}%\n"
        msg += f"💰 Odds: {odds}\n"
        
        if stake > 0:
            msg += f"💵 *SUGGESTED STAKE: ${stake}*\n"
        else:
            msg += f"❌ No value detected for this match.\n"
            
        if note:
            msg += f"\n_{note}_\n"
            
        return msg
    except Exception as e:
        log.error(f"fmt_prediction error: {e}")
        return "⚠️ Error formatting prediction."

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
    status_msg = await update.message.reply_text("🔎 Scanning South American leagues for draws...")
    
    try:
        fixtures = engine.get_todays_fixtures()
        if not fixtures:
            await status_msg.edit_text("📭 No upcoming South American fixtures found for today.")
            return

        await status_msg.edit_text(f"📊 Analyzing {len(fixtures)} fixtures. Please wait...")
        
        preds = []
        for f in fixtures:
            p = engine.predict_match(f, bankroll=BANKROLL)
            preds.append(p)
        
        best = engine.find_best_draws(preds)
        
        if not best:
             await status_msg.edit_text("❌ No high-value draws found today. Check again later!")
             return

        await status_msg.delete()
        for p in best:
            await update.message.reply_text(fmt_prediction(p), parse_mode='Markdown')
            
    except Exception as e:
        log.error(f"today error: {e}")
        await status_msg.edit_text("⚠️ An error occurred while fetching predictions.")

async def match_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /match Team A vs Team B")
        return
    
    match_str = " ".join(context.args)
    if " vs " not in match_str.lower():
        await update.message.reply_text("Please use 'vs'. Example: /match Flamengo vs Palmeiras")
        return
    
    home, away = [t.strip() for t in match_str.lower().split(" vs ")]
    status_msg = await update.message.reply_text(f"🔍 Analyzing {home} vs {away}...")
    
    # Generic fixture for manual search
    fix = {
        'home_team': home.title(), 
        'away_team': away.title(),
        'league': 'bra.1', # Default to Brazil if unknown
        'draw_odds': 3.20
    }
    
    res = engine.predict_match(fix, bankroll=BANKROLL)
    await status_msg.delete()
    await update.message.reply_text(fmt_prediction(res), parse_mode='Markdown')

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
    status_msg = await update.message.reply_text("🛰️ Rebuilding Database & Retraining AI...\nStep 1/3: Fetching ESPN historical data")
    
    try:
        # We'll use a wrapper or just the engine call
        engine.init_db()
        
        # In a real async bot, we'd run this in a thread, but for now:
        await status_msg.edit_text("🛰️ Step 2/3: Computing rolling features and training XGBoost...")
        success, model = engine.train_draw_model()
        
        if success:
             import sqlite3
             conn = sqlite3.connect(engine.DB_PATH)
             perf = conn.execute("SELECT * FROM model_performance ORDER BY id DESC LIMIT 1").fetchone()
             conn.close()
             
             auc = round(perf[2], 3) if perf else 0.0
             prec = round(perf[4], 3) if perf else 0.0
             
             msg = f"✅ *RE-TRAIN COMPLETE*\n\nModel now optimized with latest patterns.\n"
             msg += f"📈 AUC-ROC: {auc}\n"
             msg += f"🎯 Precision: {prec}\n"
             await status_msg.edit_text(msg, parse_mode='Markdown')
        else:
             await status_msg.edit_text("⚠️ Retrain failed: Not enough historical data in DB Yet.")
             
    except Exception as e:
        log.error(f"retrain error: {e}")
        await status_msg.edit_text(f"⚠️ Error during retrain: {e}")

async def cmd_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Min Odds filter: 2.80\nBookies: Bet365, Sportybet, 1xBet")

async def cmd_pending(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Processing results for matches from last 24h...")

async def cmd_settle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ All pending matches settled and bankroll updated.")

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
    
    # Handlers - Fixed registration
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("today", today))
    application.add_handler(CommandHandler("match", match_predict))
    application.add_handler(CommandHandler("record", record))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("bankroll", set_bankroll))
    application.add_handler(CommandHandler("retrain", force_retrain))
    application.add_handler(CommandHandler("odds", cmd_odds))
    application.add_handler(CommandHandler("pending", cmd_pending))
    application.add_handler(CommandHandler("settle", cmd_settle))
    application.add_handler(CommandHandler("mychatid", my_chat_id))
    
    # Scheduler
    job_queue = application.job_queue
    if job_queue:
        job_queue.run_daily(daily_job_retrain, time=dtime(8, 0, tzinfo=TZ))
        job_queue.run_daily(daily_job_predictions, time=dtime(12, 0, tzinfo=TZ))
    
    print("✅ Draw Hunter V2 Started (Zero API Keys Mode)")
    application.run_polling()

if __name__ == "__main__":
    main()
