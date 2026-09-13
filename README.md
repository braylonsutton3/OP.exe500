OP.exe — 1-Minute FVG Scalper + Multi-Timeframe Context


Primary 1-minute strategy

The top of the app is the main decision engine:

Mark active, unfilled 1-minute FVGs and their midpoints.

Prefer the nearest qualifying FVG midpoint as the active target.

If the midpoint is above current price, the candidate direction is LONG.

If the midpoint is below current price, the candidate direction is SHORT.

CORE FVGs use the normal confirmation rule:

LONG: two bullish candles, or hammer + bullish follow-through.

SHORT: two bearish candles, or shooting star + bearish follow-through.

MICRO FVGs may use a reduced-target, no-full-confirmation entry only when price is already very close to the midpoint and the projected target dollars are below the editable micro-target cap.

Maximum planned risk is editable and defaults to $300.

The midpoint of the target FVG is the take-profit level.

Intermediate FVG profit-taking

Use Track this trade after a LONG/SHORT signal if you want OP.exe to manage the setup logically.

OP.exe stores the entry, direction, and original FVG midpoint. If a new qualifying FVG forms between the entry and original target while the trade develops:

its midpoint becomes a temporary profit-taking level;

if price reaches that new midpoint while the tracked trade is in profit, OP.exe displays TAKE PROFIT / PAUSE;

after that exit, wait for a fresh valid setup before re-entering toward the original FVG midpoint.

The app does not read your brokerage position, so the Track button is how it knows which trade you want monitored.

Supporting data retained from full OP.exe

The 1-minute scalp is the primary signal, but the same page also keeps supporting analysis for 5m, 15m, 30m, 4h, 1d, and 1w, including market structure and catalyst/news context.

Depending on available data, the supporting engine can use concepts such as BOS, CHoCH, liquidity sweeps, FVG context, SMA/EMA, VWAP, RSI, ADX, displacement, volume expansion, volatility regime, ORB structure, and prior levels.

Market data

Supported futures: OP.exe prefers TopstepX / ProjectX bars when valid credentials are configured and the contract can be resolved.

Stocks, ETFs, crypto, and other Yahoo-compatible symbols use Yahoo Finance data.

Yahoo/fallback data can be delayed or proxy data. The 1-minute engine checks bar freshness and forces WAIT when the data exceeds the configured maximum age.

Futures fallback symbols may be continuous contracts and can differ from the exact execution contract.

GitHub files

Create a new repository containing:

main.py
requirements.txt
README.md

Streamlit Community Cloud

Set main.py as the app entry file.

For TopstepX / ProjectX access, add these under Streamlit Secrets rather than committing them to GitHub:

TOPSTEP_USERNAME = "your username"
TOPSTEP_API_KEY = "your API key"

Local run

pip install -r requirements.txt
streamlit run main.py

Important

OP.exe is a rules-based decision-support and research tool. LONG, SHORT, WAIT, FVG targets, catalysts, and risk calculations do not guarantee market outcomes. Slippage and gaps can cause realized losses to exceed a planned stop amount.
