# DRAW-HUNTER BOT V2
### South American Football Draw Prediction Bot (ZERO API KEYS)

```bash
# Mac/Linux — run this once inside your draw_hunter/ folder
pyenv shell 3.12.6 && pip install requests pandas numpy scipy scikit-learn xgboost joblib pytz "python-telegram-bot[job-queue]" python-dotenv && echo "✅ Draw Hunter ready"
```

For Mac with libomp (needed by XGBoost):
```bash
brew install libomp && pyenv shell 3.12.6 && pip install requests pandas numpy scipy scikit-learn xgboost joblib pytz "python-telegram-bot[job-queue]" python-dotenv && echo "✅ Draw Hunter ready"
```

## 🚀 ZERO-KEY SETUP
1. Create a `.env` file with **only** your Telegram credentials:
   ```env
   TELEGRAM_TOKEN=your_token
   CHAT_ID=your_id
   ```
2. Run the bot:
   ```bash
   python telegram_bot.py
   ```
3. Use `/retrain` inside Telegram to populate the database via ESPN.

## 🇧🇷 Leagues Covered (ESPN Free)
- Brazilian Serie A
- Argentine Primera División
- Colombian Primera A
- Chilean Primera División
- Uruguayan Primera División
- Copa Libertadores & Sudamericana
- Mexican Liga MX

## 🤖 Commands
- `/today`: Analyze today's fixtures using ESPN live data.
- `/match Team A vs Team B`: Custom prediction.
- `/retrain`: Fetch last 30 days of data and train Model.
- `/bankroll [amount]`: Update your total bankroll for sizing.

## 🛠 Tech
- **ESP API**: Zero key public endpoints.
- **ML**: XGBoost on historical draw patterns.
- **sizing**: Dynamic Kelly Criterion (20% fraction).
# draw-hunter
