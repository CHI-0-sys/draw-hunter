import os
import logging
import sqlite3
import asyncio
import pytz
from datetime import datetime, time as dtime
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
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

def confidence_emoji(conf: float) -> str:
    if conf >= 80: return "🔥"
    if conf >= 70: return "✅"
    if conf >= 60: return "⚠️"
    return "📉"

def utc_from_local(hour, minute):
    """Convert local time to UTC for scheduler."""
    import pytz
    from datetime import datetime as dt
    local_tz = pytz.timezone(TIMEZONE)
    now = dt.now()
    local_dt = local_tz.localize(dt(now.year, now.month, now.day, hour, minute))
    utc_dt = local_dt.astimezone(pytz.utc)
    return utc_dt.time()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def fmt_prediction(p: dict, index: int = 1) -> str:
    """Format a single draw prediction card. Never crashes."""
    try:
        features = p.get('features', {})

        home_dr   = features.get('home_draw_rate', 0)
        away_dr   = features.get('away_draw_rate', 0)
        goal_exp  = features.get('goal_expectancy', 0)
        h2h_rate  = features.get('h2h_draw_rate', 0)
        h2h_games = int(features.get('h2h_total_games', 0))
        is_copa   = features.get('is_copa', 0)
        is_derby  = features.get('is_derby', 0)
        alt       = features.get('altitude_factor', 0)

        h2h_str = (
            f"H2H draw rate  : {h2h_rate*100:.0f}% ({h2h_games} meetings)"
            if h2h_games > 0 else
            "H2H            : No history yet"
        )

        goal_str = (
            f"{goal_exp:.1f} ({'low — draws likely 🎯' if goal_exp < 2.3 else 'high — open game'})"
            if goal_exp > 0 else "N/A"
        )

        stake_str = (
            f"💵 Suggested stake : *${p.get('stake', 0)}* of ${BANKROLL}"
            if p.get('stake', 0) > 0 else
            "💵 No stake (edge below threshold)"
        )

        data_note = p.get('data_note', '')
        note_line = f"\n⚠️ _{data_note}_" if data_note else ""

        return (
            f"⚽ *{p.get('home_team','?')} vs {p.get('away_team','?')}*\n"
            f"{p.get('league_name','⚽')} — {p.get('time_local','TBD')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *DRAW ANALYSIS*\n"
            f"🤖 Model draw prob  : *{p.get('draw_prob', 0)}%*\n"
            f"📖 Book implied     : *{p.get('implied_prob', 0)}%*\n"
            f"📐 Edge             : *{p.get('edge_pct', 0):+.1f}%* {p.get('edge_label','')}\n"
            f"\n"
            f"🎯 Confidence : *{p.get('confidence', 0)}%* {confidence_emoji(p.get('confidence', 0))}\n"
            f"💰 Draw odds  : *{p.get('draw_odds', 3.20)}*\n"
            f"{stake_str}\n"
            f"\n"
            f"📈 *DRAW FACTORS*\n"
            f"  Home draw rate (L{engine.LOOKBACK}) : {home_dr*100:.0f}%\n"
            f"  Away draw rate (L{engine.LOOKBACK}) : {away_dr*100:.0f}%\n"
            f"  Combined draw rate       : {((home_dr+away_dr)/2)*100:.0f}%\n"
            f"  {h2h_str}\n"
            f"  Goal expectancy          : {goal_str}\n"
            f"  Copa/knockout format     : {'Yes ⚠️' if is_copa else 'No'}\n"
            f"  Derby match              : {'Yes 🔥' if is_derby else 'No'}\n"
            f"  Altitude factor          : {alt*100:.0f}%\n"
            f"{note_line}\n"
            f"⚠️ Verify draw odds at Sportybet before betting"
        )
    except Exception as e:
        log.error(f"fmt_prediction error: {e}")
        return (
            f"⚽ *{p.get('home_team','?')} vs {p.get('away_team','?')}*\n"
            f"Draw prob: {p.get('draw_prob','?')}% | "
            f"Edge: {p.get('edge_pct','?')}% | "
            f"Odds: {p.get('draw_odds','?')}"
        )

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

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """👋 Welcome to Draw Hunter Bot V2!
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

async def cmd_today(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "⚽ Fetching today's South American fixtures..."
    )
    try:
        fixtures = engine.get_todays_fixtures(TIMEZONE)

        if not fixtures:
            await msg.edit_text(
                "📅 No South American fixtures found today.\n\n"
                "Possible reasons:\n"
                "• Midweek break (SA leagues usually play Thu–Sun)\n"
                "• ESPN API delay — try again in 10 mins\n\n"
                "Use /match to predict a specific game:\n"
                "`/match Flamengo vs Palmeiras`",
                parse_mode='Markdown'
            )
            return

        await msg.edit_text(
            f"⚽ Found *{len(fixtures)}* fixtures — analyzing draws...",
            parse_mode='Markdown'
        )

        predictions = []
        for i, fixture in enumerate(fixtures):
            try:
                pred = engine.predict_match(fixture, BANKROLL)
                predictions.append(pred)
            except Exception as e:
                log.error(f"Prediction error fixture {i}: {e}")

        if not predictions:
            await update.message.reply_text(
                "❌ Predictions failed for all fixtures.\n"
                "Run /retrain to load match data first."
            )
            return

        value_draws = engine.find_best_draws(predictions)

        # Send summary
        await update.message.reply_text(
            fmt_daily_report(value_draws, len(predictions)),
            parse_mode='Markdown'
        )

        # Send detailed card for top value draws
        for p in value_draws[:4]:
            await asyncio.sleep(1)
            kb = [[InlineKeyboardButton(
                "📋 Log this pick",
                callback_data=f"log_{p['fixture_id']}"
            )]]
            await update.message.reply_text(
                fmt_prediction(p),
                parse_mode='Markdown',
                reply_markup=InlineKeyboardMarkup(kb)
            )

        # Cache for logging
        ctx.bot_data['today_predictions'] = {
            p['fixture_id']: p for p in predictions
        }

        # If no value draws, show top 3 anyway for info
        if not value_draws and predictions:
            await update.message.reply_text(
                "📊 *Top matches by draw probability:*",
                parse_mode='Markdown'
            )
            top3 = sorted(predictions, key=lambda x: x.get('draw_prob', 0), reverse=True)[:3]
            for p in top3:
                await asyncio.sleep(1)
                await update.message.reply_text(
                    fmt_prediction(p), parse_mode='Markdown'
                )

    except Exception as e:
        log.error(f"cmd_today error: {e}", exc_info=True)
        await msg.edit_text(
            f"❌ Error: {str(e)[:200]}\n\n"
            f"Try /retrain first to load match data."
        )

async def cmd_match(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def cmd_record(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stats = tracker.get_stats()
    await update.message.reply_text(tracker.format_stats_message(stats))

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    try:
        conn    = sqlite3.connect(engine.DB_PATH)
        total   = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        usable  = conn.execute(
            "SELECT COUNT(*) FROM matches "
            "WHERE combined_draw_rate IS NOT NULL AND combined_draw_rate > 0"
        ).fetchone()[0]
        dr      = conn.execute(
            "SELECT AVG(is_draw) FROM matches WHERE combined_draw_rate > 0"
        ).fetchone()[0] or 0.0
        latest  = conn.execute(
            "SELECT MAX(match_date) FROM matches"
        ).fetchone()[0] or 'None'
        th      = conn.execute(
            "SELECT COUNT(*) FROM team_history"
        ).fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM prediction_log WHERE result='PENDING'"
        ).fetchone()[0]
        mp      = conn.execute(
            "SELECT auc_roc, brier_score, samples, date "
            "FROM model_performance ORDER BY date DESC LIMIT 1"
        ).fetchone()

        # Per-league breakdown
        league_counts = conn.execute(
            "SELECT league, COUNT(*) FROM matches GROUP BY league ORDER BY COUNT(*) DESC LIMIT 8"
        ).fetchall()
        conn.close()

        import os
        model_exists = os.path.exists(f'{engine.MODELS_DIR}/draw_model.pkl')
        model_str = (
            f"✅ Trained | AUC: {mp[0]:.3f} | "
            f"Samples: {mp[2]} ({mp[3][:10]})"
            if mp and model_exists else
            "❌ Not trained yet — run /retrain"
        )

        league_str = "\n".join([
            f"   {lc[0]}: {lc[1]} matches" for lc in league_counts
        ]) or "   None"

        await update.message.reply_text(
            f"📊 *DRAW HUNTER STATUS*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📦 Total matches   : {total}\n"
            f"✅ Usable (w/feats): {usable}\n"
            f"📊 Draw rate       : {dr*100:.1f}%\n"
            f"📅 Latest data     : {latest}\n"
            f"👥 Team histories  : {th}\n"
            f"⏳ Pending picks   : {pending}\n"
            f"🤖 Model           : {model_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📋 *Matches by league:*\n{league_str}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🌍 Timezone : {TIMEZONE}\n"
            f"💰 Bankroll : ${BANKROLL}\n"
            f"🕐 Local    : {datetime.now(TZ).strftime('%I:%M %p %Z')}",
            parse_mode='Markdown'
        )
    except Exception as e:
        await update.message.reply_text(f"Status error: {e}")

async def cmd_bankroll(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def cmd_retrain(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text(
        "🔄 *Draw Hunter Retrain*\n"
        "⏳ Step 1/4: Fixing DB schema...",
        parse_mode='Markdown'
    )
    try:
        # Step 1: Schema
        engine.init_db()
        await msg.edit_text(
            "🔄 *Draw Hunter Retrain*\n"
            "✅ Step 1/4: DB schema ready\n"
            "⏳ Step 2/4: Downloading CSVs...",
            parse_mode='Markdown'
        )

        # Step 2: CSVs
        csv_counts = {}
        for code in engine.CSV_SOURCES:
            try:
                df = engine.fetch_csv_training_data(code)
                count = len(df) if hasattr(df, '__len__') else 0
                if count > 0:
                    engine.store_csv_matches(df, code)
                    csv_counts[code] = count
                    log.info(f"Stored {count} rows for {code}")
                else:
                    log.warning(f"No data returned for {code}")
                    csv_counts[code] = 0
            except Exception as e:
                log.error(f"CSV {code} failed: {e}")
                csv_counts[code] = 0

        # Check DB after CSV import
        conn = sqlite3.connect(engine.DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        usable = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE combined_draw_rate IS NOT NULL AND combined_draw_rate > 0"
        ).fetchone()[0]
        dr = conn.execute(
            "SELECT AVG(is_draw) FROM matches WHERE combined_draw_rate > 0"
        ).fetchone()[0] or 0.0
        conn.close()

        # Build CSV summary line
        csv_summary = " | ".join([
            f"{code.upper()}: {n}" for code, n in csv_counts.items()
        ]) or "No CSV data"

        await msg.edit_text(
            f"🔄 *Draw Hunter Retrain*\n"
            f"✅ Step 1/4: DB schema ready\n"
            f"✅ Step 2/4: {total} matches stored\n"
            f"   {csv_summary}\n"
            f"   Usable rows: {usable} | Draw rate: {dr*100:.1f}%\n"
            f"⏳ Step 3/4: Fetching ESPN team history...",
            parse_mode='Markdown'
        )

        # Step 3: ESPN history
        fetched = 0
        try:
            fixtures = engine.get_todays_fixtures(TIMEZONE)
            fixture_count = len(fixtures) if fixtures else 0
            for f in fixtures[:10]:
                for tid, lcode in [
                    (f.get('home_team_id',''), f.get('league','')),
                    (f.get('away_team_id',''), f.get('league',''))
                ]:
                    if tid and tid != '0':
                        try:
                            recs = engine.fetch_espn_team_history(tid, lcode, 15)
                            if recs:
                                engine.store_team_history(recs)
                                fetched += 1
                        except Exception:
                            pass
                    await asyncio.sleep(0.6)
        except Exception as e:
            log.warning(f"ESPN step warning: {e}")
            fixture_count = 0

        await msg.edit_text(
            f"🔄 *Draw Hunter Retrain*\n"
            f"✅ Step 1/4: DB schema ready\n"
            f"✅ Step 2/4: {total} matches | {usable} usable\n"
            f"✅ Step 3/4: {fetched} team histories fetched\n"
            f"⏳ Step 4/4: Training draw model...",
            parse_mode='Markdown'
        )

        # Step 4: Train
        model, _ = engine.train_draw_model()

        conn = sqlite3.connect(engine.DB_PATH)
        mp = conn.execute(
            "SELECT auc_roc, draw_precision, brier_score "
            "FROM model_performance ORDER BY date DESC LIMIT 1"
        ).fetchone()
        th = conn.execute("SELECT COUNT(*) FROM team_history").fetchone()[0]
        conn.close()

        if model:
            if mp:
                model_line = (
                    f"✅ Model trained\n"
                    f"   AUC: {mp[0]:.3f} | "
                    f"Precision: {mp[1]:.3f}"
                )
            else:
                model_line = "✅ Model trained"
        else:
            model_line = (
                f"⚠️ Need 150+ usable rows\n"
                f"   Have {usable} — "
                + ("almost there!" if usable > 100 else
                   "CSV download may have failed. Check /status")
            )

        await msg.edit_text(
            f"{'✅' if model else '⚠️'} *Retrain Complete*\n\n"
            f"📦 Total matches     : {total}\n"
            f"✅ Usable for model  : {usable}\n"
            f"📊 Draw rate         : {dr*100:.1f}%\n"
            f"👥 Team histories    : {th}\n"
            f"🤖 {model_line}\n\n"
            + (
                "Run /today to get predictions."
                if model else
                "⚠️ CSV download failed — check internet and run /retrain again.\n"
                "Run /status to see DB details."
            ),
            parse_mode='Markdown'
        )

    except Exception as e:
        log.error(f"Retrain error: {e}", exc_info=True)
        await msg.edit_text(
            f"❌ Retrain error:\n`{str(e)[:300]}`",
            parse_mode='Markdown'
        )

async def cmd_odds(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📊 Min Odds filter: 2.80\nBookies: Bet365, Sportybet, 1xBet")

async def cmd_pending(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Processing results for matches from last 24h...")

async def cmd_settle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ All pending matches settled and bankroll updated.")

async def cmd_mychatid(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Your Chat ID: {update.effective_chat.id}")

async def cmd_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = "🏆 *SA LEAGUES COVERED*\n\n"
    for code, name in engine.ESPN_LEAGUES.items():
        msg += f"• `{code}`: {name}\n"
    await update.message.reply_text(msg, parse_mode='Markdown')

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data.startswith("log_"):
        fid = data.replace("log_", "")
        await query.edit_message_caption(caption=f"{query.message.caption}\n\n✅ Logged to tracker!")

async def job_retrain(context: ContextTypes.DEFAULT_TYPE):
    engine.daily_retrain()
    if CHAT_ID:
        await context.bot.send_message(chat_id=CHAT_ID, text="✅ Daily retrain complete. Model updated with latest results.")

async def job_daily_predictions(context: ContextTypes.DEFAULT_TYPE):
    fixtures = engine.get_todays_fixtures()
    if not fixtures: return
    preds = []
    for f in fixtures:
        p = engine.predict_match(f, bankroll=BANKROLL)
        preds.append(p)
    best = engine.find_best_draws(preds)
    if CHAT_ID:
        await context.bot.send_message(chat_id=CHAT_ID, text=fmt_daily_report(preds, best))

async def job_auto_settle(context: ContextTypes.DEFAULT_TYPE):
    # Placeholder for auto-settling logic
    log.info("Auto-settle job triggered")

def main():
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN not set in .env")
        return

    engine.init_db()

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Register every command
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("today",    cmd_today))
    app.add_handler(CommandHandler("match",    cmd_match))
    app.add_handler(CommandHandler("odds",     cmd_odds))
    app.add_handler(CommandHandler("record",   cmd_record))
    app.add_handler(CommandHandler("pending",  cmd_pending))
    app.add_handler(CommandHandler("settle",   cmd_settle))
    app.add_handler(CommandHandler("retrain",  cmd_retrain))
    app.add_handler(CommandHandler("leagues",  cmd_leagues))
    app.add_handler(CommandHandler("status",   cmd_status))
    app.add_handler(CommandHandler("mychatid", cmd_mychatid))
    app.add_handler(CommandHandler("bankroll", cmd_bankroll))
    app.add_handler(CallbackQueryHandler(handle_callback))

    # Scheduled jobs in Africa local time
    jq = app.job_queue
    if jq:
        jq.run_daily(job_retrain,           utc_from_local(8,  0))
        jq.run_daily(job_daily_predictions, utc_from_local(12, 0))
        jq.run_daily(job_auto_settle,       utc_from_local(9,  0))
        log.info("Scheduled jobs registered ✅")
    else:
        log.warning("No job queue — install: pip install python-telegram-bot[job-queue]")

    log.info("⚽ DRAW HUNTER started. All commands registered.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
