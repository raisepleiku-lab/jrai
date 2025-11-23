import logging
import os
import json
import requests
import time  # dùng cho volume delta

from telegram import (
    Update,
    constants,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    BotCommand,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)

# ========= LOGGING =========
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ========= TELEGRAM TOKEN =========
# 👉 THAY TOKEN CỦA MÀY VÀO ĐÂY (lấy từ BotFather)
TOKEN = os.getenv("8340989991:AAFbc5IiM5onGkvJDdzTrVzBgvseMrD-8xA")

# ========= CONFIG =========
# Khung thời gian cho report
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# Các symbol yêu thích
FAV_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "TRUMP": "TRUMPUSDT",
    "BTCDOM": "BTCDOMUSDT",
    "STRK": "STRKUSDT",
    "XRP": "XRPUSDT",
    "TAO": "TAOUSDT",
    "ICP": "ICPUSDT",
    "VIRTUAL": "VIRTUALUSDT",
}

ALERTS_FILE = "alerts.json"
alerts = []


# ========= BASIC UTILS =========
def normalize_symbol(s: str):
    """Chuẩn hoá BTC -> BTCUSDT, BTCDOM -> BTCDOMUSDT, v.v."""
    s = s.upper()
    return FAV_SYMBOLS.get(s, s)


def load_alerts():
    """Load alert từ file JSON."""
    global alerts
    if os.path.exists(ALERTS_FILE):
        try:
            alerts = json.load(open(ALERTS_FILE))
        except Exception:
            alerts = []
    else:
        alerts = []


def save_alerts():
    """Lưu alert ra file JSON."""
    try:
        json.dump(alerts, open(ALERTS_FILE, "w"))
    except Exception:
        pass


def fmt_num(n, d=4):
    """Format số cho đẹp."""
    return f"{n:.{d}f}" if n is not None else "N/A"


# ========= BINANCE DATA & INDICATORS =========
def get_klines(symbol="BTCUSDT", interval="1h", limit=100):
    """
    Lấy nến từ Binance.

    Logic:
    - Nếu là BTCDOMUSDT: thử Futures (fapi) với User-Agent, nếu Binance trả 418 thì báo lỗi dễ hiểu.
    - Các symbol ...USDT khác:
        + Thử Futures (fapi) trước.
        + Nếu fapi lỗi thì fallback về Spot (/api/v3/klines).
    - Còn lại (không phải ...USDT): dùng Spot luôn.
    """
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {"User-Agent": "Mozilla/5.0"}

    # Case đặc biệt: BTCDOMUSDT (dominance index)
    if symbol == "BTCDOMUSDT":
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params=params,
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 418:
                # IP server bị Binance từ chối cho BTCDOM
                raise RuntimeError(
                    "Binance trả 418 cho BTCDOMUSDT trên server này (IP bị chặn). "
                    "Tạm thời bot không lấy được nến BTCDOM trên futures."
                )
            else:
                raise

    # Các cặp ...USDT khác: ưu tiên lấy nến Futures (fapi)
    if symbol.endswith("USDT"):
        try:
            r = requests.get(
                "https://fapi.binance.com/fapi/v1/klines",
                params=params,
                headers=headers,
                timeout=10,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            # Có thể symbol đó không có futures hoặc bị chặn → thử Spot
            pass
        except Exception:
            # Lỗi network gì đó → thử Spot
            pass

    # Fallback: dùng Spot /api/v3/klines
    r = requests.get(
        "https://api.binance.com/api/v3/klines",
        params=params,
        headers=headers,
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def calc_ma(values, length):
    if len(values) < length:
        return None
    return sum(values[-length:]) / length


def calc_ema(values, length):
    if len(values) < length:
        return None
    k = 2 / (length + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def calc_rsi(closes, length=14):
    if len(closes) < length + 1:
        return None
    gains, losses = [], []
    for i in range(-length, 0):
        ch = closes[i] - closes[i - 1]
        gains.append(ch if ch > 0 else 0)
        losses.append(-ch if ch < 0 else 0)
    ag = sum(gains) / length
    al = sum(losses) / length
    if al == 0:
        return 100
    rs = ag / al
    return 100 - 100 / (1 + rs)


def calc_atr(highs, lows, closes, length=14):
    if len(highs) < length + 1 or len(lows) < length + 1 or len(closes) < length + 1:
        return None
    trs = []
    for i in range(-length, 0):
        h = highs[i]
        l = lows[i]
        prev_c = closes[i - 1]
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    return sum(trs) / length


def get_indicators(symbol, tf):
    """Lấy full bộ thông số cho report (OHLC, MA, RSI, ATR, Vol...)."""
    data = get_klines(symbol, tf, 100)
    opens = [float(x[1]) for x in data]
    highs = [float(x[2]) for x in data]
    lows = [float(x[3]) for x in data]
    closes = [float(x[4]) for x in data]
    vols = [float(x[5]) for x in data]

    last_open = opens[-1]
    last_high = highs[-1]
    last_low = lows[-1]
    last_close = closes[-1]
    last_vol = vols[-1]
    prev_close = closes[-2] if len(closes) >= 2 else None

    change_pct = None
    if prev_close and prev_close != 0:
        change_pct = (last_close - prev_close) / prev_close * 100

    range_val = last_high - last_low
    range_pct = (range_val / last_close * 100) if last_close != 0 else None
    body_pct = (abs(last_close - last_open) / range_val * 100) if range_val != 0 else None

    atr14 = calc_atr(highs, lows, closes, 14)
    ma20 = calc_ma(closes, 20)
    ma50 = calc_ma(_
