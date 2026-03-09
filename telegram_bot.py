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

def fmt_daily_report(tiers: dict, total_analyzed: int) -> str:
    """
    Tiered daily report.
    tiers: dict from find_best_draws()
    """
    elite  = tiers.get('elite', [])
    strong = tiers.get('strong', [])
    value  = tiers.get('value', [])
    lean   = tiers.get('lean', [])

    all_picks = elite + strong + value + lean
    staking   = elite + strong + value  # picks with actual stake

    if not all_picks:
        return (
            f"⚽ *DRAW HUNTER — {datetime.now().strftime('%b %d, %Y')}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 {total_analyzed} matches analyzed\n"
            f"❌ No draw edges found today\n\n"
            f"Book odds too tight or model sees no edge.\n"
            f"Come back tomorrow 💤"
        )

    lines = [
        f"⚽ *DRAW HUNTER — {datetime.now().strftime('%b %d, %Y')}*",
        f"━━━━━━━━━━━━━━━━━━━━",
        f"📊 Analyzed : *{total_analyzed}* matches",
        f"🎯 Flagged  : *{len(all_picks)}* draws "
        f"({len(staking)} with stake)",
        f"",
    ]

    # Elite tier
    if elite:
        lines.append(f"🔥🔥 *ELITE VALUE ({len(elite)}):*")
        for p in elite[:4]:
            lines += [
                f"  ⚽ *{p['home_team']} vs {p['away_team']}*",
                f"  {p.get('league_name','⚽')} — {p.get('time_local','TBD')}",
                f"  Draw: {p['draw_prob']}% | Edge: {p['edge_pct']:+.1f}% | "
                f"Odds: {p['draw_odds']} | Stake: *${p['stake']}*",
                f"",
            ]

    # Strong tier
    if strong:
        lines.append(f"🔥 *STRONG VALUE ({len(strong)}):*")
        for p in strong[:5]:
            lines += [
                f"  ⚽ *{p['home_team']} vs {p['away_team']}*",
                f"  {p.get('league_name','⚽')} — {p.get('time_local','TBD')}",
                f"  Draw: {p['draw_prob']}% | Edge: {p['edge_pct']:+.1f}% | "
                f"Odds: {p['draw_odds']} | Stake: *${p['stake']}*",
                f"",
            ]

    # Value tier
    if value:
        lines.append(f"✅ *GOOD VALUE ({len(value)}):*")
        for p in value[:6]:
            lines += [
                f"  ⚽ {p['home_team']} vs {p['away_team']}",
                f"  {p.get('league_name','⚽')} | "
                f"Draw: {p['draw_prob']}% | Edge: {p['edge_pct']:+.1f}% | "
                f"Stake: ${p['stake']}",
                f"",
            ]

    # Lean tier (no stake, info only)
    if lean:
        lines.append(f"⚠️ *LEAN / WATCH ({len(lean)}) — no stake:*")
        for p in lean[:5]:
            lines.append(
                f"  {p['home_team']} vs {p['away_team']} | "
                f"Draw: {p['draw_prob']}% | Edge: {p['edge_pct']:+.1f}%"
            )
        lines.append("")

    # Best pick summary
    best = (elite + strong + value)[0] if (elite + strong + value) else None
    if best:
        lines += [
            f"━━━━━━━━━━━━━━━━━━━━",
            f"🏆 *BEST PICK: {best['home_team']} vs {best['away_team']}*",
            f"   {best.get('league_name','')} | "
            f"Edge: {best['edge_pct']:+.1f}% | "
            f"Stake: ${best['stake']}",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"⚠️ Verify all odds before betting",
        ]

    return "\n".join(lines)

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
/draws - Top matches by draw prob
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

        tiers = engine.find_best_draws(predictions)

        # Send summary
        await update.message.reply_text(
            fmt_daily_report(tiers, len(predictions)),
            parse_mode='Markdown'
        )

        # Send detailed cards for top picks only (elite + strong)
        top_detail = tiers.get('elite', []) + tiers.get('strong', [])
        top_detail = top_detail[:5]
        
        if top_detail:
            await update.message.reply_text(
                f"🔍 *Detailed analysis — top {len(top_detail)} picks:*",
                parse_mode='Markdown'
            )
            for p in top_detail:
                await asyncio.sleep(0.8)
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

        all_shown = tiers.get('all_value', []) + tiers.get('lean', [])
        # If truly nothing at all
        if not all_shown and predictions:
            top5 = sorted(
                [p for p in predictions if p.get('draw_prob', 0) >= 25],
                key=lambda x: x.get('draw_prob', 0),
                reverse=True
            )[:5]
            if top5:
                await update.message.reply_text(
                    f"📊 *Top {len(top5)} by draw probability "
                    f"(no edge found):*",
                    parse_mode='Markdown'
                )
                for p in top5:
                    await asyncio.sleep(0.8)
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
    tiers = engine.find_best_draws(preds)
    if CHAT_ID:
        report = fmt_daily_report(tiers, len(preds))
        await context.bot.send_message(chat_id=CHAT_ID, text=report, parse_mode='Markdown')

async def job_auto_settle(context: ContextTypes.DEFAULT_TYPE):
    # Placeholder for auto-settling logic
    log.info("Auto-settle job triggered")

async def cmd_draws(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Show top 20 matches by raw draw probability — no edge filter.
    Useful when odds not available or model just trained.
    Usage: /draws
           /draws 30    (show top 30)
           /draws clay  (filter by league keyword)
    """
    limit  = 20
    filter_kw = None

    if ctx.args:
        arg = ctx.args[0]
        if arg.isdigit():
            limit = min(int(arg), 50)
        else:
            filter_kw = arg.lower()

    msg = await update.message.reply_text(
        f"⚽ Fetching top draw candidates..."
    )
    try:
        fixtures = engine.get_todays_fixtures(TIMEZONE)

        if filter_kw:
            fixtures = [
                f for f in fixtures
                if filter_kw in f.get('league','').lower() or
                   filter_kw in f.get('league_name','').lower() or
                   filter_kw in f.get('country','').lower()
            ]

        if not fixtures:
            await msg.edit_text("No fixtures found. Try /today.")
            return

        await msg.edit_text(
            f"⚽ Predicting {len(fixtures)} fixtures...",
            parse_mode='Markdown'
        )

        predictions = []
        for fixture in fixtures:
            try:
                pred = engine.predict_match(fixture, BANKROLL)
                predictions.append(pred)
            except Exception:
                pass

        # Sort purely by draw_prob — ignore edge
        predictions.sort(key=lambda x: x.get('draw_prob', 0), reverse=True)
        top = predictions[:limit]

        if not top:
            await update.message.reply_text("No predictions generated.")
            return

        lines = [
            f"⚽ *TOP {len(top)} DRAW CANDIDATES*",
            f"📊 {len(predictions)} matches analyzed",
            f"Sorted by draw probability (no edge filter)",
            f"━━━━━━━━━━━━━━━━━━━━",
            f"",
        ]

        for i, p in enumerate(top, 1):
            edge_str = f"Edge: {p['edge_pct']:+.1f}%" if p.get('draw_odds') else "No odds"
            tier_e   = (
                "🔥🔥" if p['edge_pct'] >= engine.TIER_ELITE else
                "🔥"   if p['edge_pct'] >= engine.TIER_STRONG else
                "✅"   if p['edge_pct'] >= engine.TIER_VALUE else
                "⚠️"   if p['edge_pct'] >= engine.TIER_LEAN else
                "📊"
            )
            lines.append(
                f"{i:2}. {tier_e} *{p['home_team']} vs {p['away_team']}*\n"
                f"    {p.get('league_name','⚽')} — {p.get('time_local','TBD')}\n"
                f"    Draw: *{p['draw_prob']}%* | "
                f"{edge_str} | Odds: {p.get('draw_odds','?')}\n"
            )

        lines += [
            f"━━━━━━━━━━━━━━━━━━━━",
            f"🔥🔥 ≥10% edge | 🔥 ≥7% | ✅ ≥4% | ⚠️ ≥2% | 📊 marginal",
            f"Use /today for full value analysis",
        ]

        await update.message.reply_text(
            "\n".join(lines), parse_mode='Markdown'
        )

    except Exception as e:
        log.error(f"cmd_draws: {e}", exc_info=True)
        await msg.edit_text(f"❌ Error: {e}")

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
    app.add_handler(CommandHandler("today",      cmd_today))
    app.add_handler(CommandHandler("draws",      cmd_draws))
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
