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
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
)

from groq import Groq

# ========= GROQ API (LLaMA 3.1 70B) =========
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ========= LOGGING =========
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ========= TELEGRAM TOKEN =========
# 👉 THAY TOKEN CỦA MÀY VÀO ĐÂY (lấy từ BotFather)
TOKEN = "8340989991:AAFbc5IiM5onGkvJDdzTrVzBgvseMrD-8xA"

# ========= CONFIG =========
# Khung thời gian cho report
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# Bổ sung BTCDOM + STRK, XRP, TAO, ICP, VIRTUAL
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


# ========= INDICATORS / DATA TỪ BINANCE =========
def get_klines(symbol="BTCUSDT", interval="1h", limit=100):
    """
    Lấy nến từ Binance.

    Logic:
    - Nếu là BTCDOMUSDT: thử Futures (fapi) với User-Agent, nếu Binance trả 418 thì báo lỗi dễ hiểu.
    - Các symbol ...USDT khác:
        + Thử Futures (fapi) trước (nến futures thường đủ dùng cho phân tích).
        + Nếu fapi lỗi (không list futures / 418 / các kiểu) thì fallback về Spot (/api/v3/klines).
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
                # IP Render bị Binance từ chối cho BTCDOM
                raise RuntimeError(
                    "Binance trả 418 cho BTCDOMUSDT trên server này (IP Render bị chặn). "
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
    """Lấy full bộ thông số cho report & plan (OHLC, MA, RSI, ATR, Vol...)."""
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
    ma50 = calc_ma(closes, 50)
    ema20 = calc_ema(closes, 20)
    ema50 = calc_ema(closes, 50)
    rsi14 = calc_rsi(closes, 14)
    vol_ma20 = calc_ma(vols, 20)

    range_pos_14 = None
    if len(highs) >= 14 and len(lows) >= 14:
        hh = max(highs[-14:])
        ll = min(lows[-14:])
        if hh != ll:
            range_pos_14 = (last_close - ll) / (hh - ll) * 100

    return {
        "price": last_close,
        "open": last_open,
        "high": last_high,
        "low": last_low,
        "prev_close": prev_close,
        "change_pct": change_pct,
        "range_pct": range_pct,
        "body_pct": body_pct,
        "ma20": ma20,
        "ma50": ma50,
        "ema20": ema20,
        "ema50": ema50,
        "rsi14": rsi14,
        "atr14": atr14,
        "vol": last_vol,
        "vol_ma20": vol_ma20,
        "range_pos_14": range_pos_14,
    }


def fmt_num(n, d=4):
    """Format số cho đẹp."""
    return f"{n:.{d}f}" if n is not None else "N/A"


def get_price(symbol):
    """
    Lấy giá hiện tại.
    - BTCDOMUSDT: futures ticker
    - Còn lại: spot ticker
    """
    if symbol == "BTCDOMUSDT":
        url = "https://fapi.binance.com/fapi/v1/ticker/price"
    else:
        url = "https://api.binance.com/api/v3/ticker/price"

    r = requests.get(url, params={"symbol": symbol}, timeout=10)
    r.raise_for_status()
    return float(r.json()["price"])


def get_swing_levels(symbol, interval="1h", lookback=40):
    """Tính swing high & swing low gần nhất."""
    data = get_klines(symbol, interval, lookback)
    highs = [float(x[2]) for x in data]
    lows = [float(x[3]) for x in data]
    closes = [float(x[4]) for x in data]
    return max(highs), min(lows), closes[-1]


# ========= THÊM: FUNDING, OI, ORDERBOOK, DELTA =========
def get_funding_rates(symbol):
    """
    Funding rate:
    - USDT-M: fapi/v1/fundingRate
    - COIN-M: dapi/v1/fundingRate (chỉ map cho BTC, ETH)
    """
    fr_usdt = None
    fr_coin = None

    # USDT-M
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": symbol, "limit": 1},
            timeout=10,
        )
        data = r.json()
        if data:
            fr_usdt = float(data[0]["fundingRate"])
    except Exception as e:
        logger.warning("Funding USDT-M error %s", e)

    # COIN-M: chỉ map BTC, ETH
    try:
        base = symbol.replace("USDT", "")
        coin_map = {
            "BTC": "BTCUSD_PERP",
            "ETH": "ETHUSD_PERP",
        }
        cm_sym = coin_map.get(base)
        if cm_sym:
            r2 = requests.get(
                "https://dapi.binance.com/dapi/v1/fundingRate",
                params={"symbol": cm_sym, "limit": 1},
                timeout=10,
            )
            d2 = r2.json()
            if d2:
                fr_coin = float(d2[0]["fundingRate"])
    except Exception as e:
        logger.warning("Funding COIN-M error %s", e)

    return fr_usdt, fr_coin


def get_open_interest_stats(symbol):
    """
    OI:
    - OI tổng: fapi/v1/openInterest
    - OI 5m/15m/1h: dùng openInterestHist, tính thay đổi so với kỳ trước.
    """
    base_hist = "https://fapi.binance.com/futures/data/openInterestHist"
    oi_changes = {}

    for period in ["5m", "15m", "1h"]:
        try:
            r = requests.get(
                base_hist,
                params={"symbol": symbol, "period": period, "limit": 2},
                timeout=10,
            )
            data = r.json()
            if isinstance(data, list) and len(data) >= 2:
                last = float(data[-1]["sumOpenInterest"])
                prev = float(data[-2]["sumOpenInterest"])
                diff = last - prev
                pct = (diff / prev * 100) if prev != 0 else None
                oi_changes[period] = (last, diff, pct)
            else:
                oi_changes[period] = (None, None, None)
        except Exception as e:
            logger.warning("OI hist %s error %s", period, e)
            oi_changes[period] = (None, None, None)

    oi_total = None
    try:
        r2 = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": symbol},
            timeout=10,
        )
        d2 = r2.json()
        if "openInterest" in d2:
            oi_total = float(d2["openInterest"])
    except Exception as e:
        logger.warning("OI total error %s", e)

    return oi_changes, oi_total


def get_orderbook_imbalance(symbol, limit=50):
    """
    Orderbook imbalance:
    - Sum qty bid & ask trong top 50 levels
    """
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/depth",
            params={"symbol": symbol, "limit": limit},
            timeout=10,
        )
        d = r.json()
        bids = d.get("bids", [])
        asks = d.get("asks", [])
        bid_vol = sum(float(x[1]) for x in bids)
        ask_vol = sum(float(x[1]) for x in asks)
        net = bid_vol - ask_vol
        return bid_vol, ask_vol, net
    except Exception as e:
        logger.warning("Orderbook error %s", e)
        return None, None, None


def get_volume_delta(symbol, minutes):
    """
    Volume delta trong X phút:
    - Dùng fapi/v1/aggTrades
    - isBuyerMaker = True  => sell
    - isBuyerMaker = False => buy
    """
    try:
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - minutes * 60 * 1000
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/aggTrades",
            params={"symbol": symbol, "startTime": start_ts, "endTime": end_ts},
            timeout=10,
        )
        data = r.json()
        buy_vol = 0.0
        sell_vol = 0.0
        if isinstance(data, list):
            for t in data:
                qty = float(t["q"])
                is_buyer_maker = t["m"]
                if is_buyer_maker:
                    sell_vol += qty
                else:
                    buy_vol += qty
        net = buy_vol - sell_vol
        return buy_vol, sell_vol, net
    except Exception as e:
        logger.warning("Delta error %s", e)
        return None, None, None


# ========= AI LLaMA 3.1 70B (Groq) =========
def ai_trade_view(
    symbol,
    side,
    tf,
    entry,
    price,
    sl,
    tp1,
    tp2,
    rsi,
    ma20,
    ma50,
    sh,
    slv,
):
    prompt = f"""
Phân tích crypto tham khảo:
- Symbol: {symbol}
- Phe: {side.upper()}
- Khung thời gian: {tf}
- Entry: {entry}
- Giá hiện tại: {price}
- SL: {sl}
- TP1: {tp1}
- TP2: {tp2}
- Swing high (kháng cự gần): {sh}
- Swing low (hỗ trợ gần): {slv}
- RSI: {rsi}
- MA20: {ma20}
- MA50: {ma50}

Yêu cầu:
- Viết 6–10 dòng bằng tiếng Việt, giọng thân thiện, kỹ thuật dễ hiểu.
- Không phím kèo, không all-in, không hứa chắc thắng.
- Chỉ ra:
  • Xu hướng nghiêng về bull/bear/sideway dựa trên MA và RSI.
  • Gợi ý cách nhìn vùng entry này: đu đỉnh, mua đáy, hay vùng giữa range.
  • Khi nào nên coi setup này là fail (mất hỗ trợ/kháng cự nào).
  • 1–2 lưu ý về quản lý rủi ro (giảm size, vào từng phần, v.v.).
"""

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia phân tích crypto, chỉ phân tích kỹ thuật THAM KHẢO, không cho lời khuyên đầu tư.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=500,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error("AI error: %s", e)
        return f"(AI lỗi: {e})"


# ========= HELP / MENU TEXT =========
def get_help_text():
    return (
        "📌 *Các lệnh chính:*\n"
        "/start – mở menu chính\n"
        "/help – xem lệnh nhanh\n"
        "/report BTC – report đa khung (5m, 15m, 1h, 4h, 1d)\n\n"
        "Lệnh long/short (có AI):\n"
        "  /longbtc [entry] [tf]\n"
        "  /shortbtc [entry] [tf]\n"
        "  /longeth [entry] [tf]\n"
        "  /shorteth [entry] [tf]\n"
        "  /longsol [entry] [tf]\n"
        "  /shortsol [entry] [tf]\n"
        "  /longtrump [entry] [tf]\n"
        "  /shorttrump [entry] [tf]\n\n"
        "Ví dụ:\n"
        "  /longbtc           → kế hoạch long BTC 1h\n"
        "  /longbtc 62000     → đánh giá lệnh long BTC entry 62000 (1h)\n"
        "  /shorteth 3500 4h  → đánh giá lệnh short ETH entry 3500 (4h)\n\n"
        "Có thể dùng /report với:\n"
        "  BTC, ETH, SOL, TRUMP, BTCDOM,\n"
        "  STRK, XRP, TAO, ICP, VIRTUAL\n\n"
        "Alert giá:\n"
        "  /alert BTC 1h below 60000\n"
        "  /alert BTC 1h above 65000\n"
    )


def get_main_menu_text():
    return "🏠 *Menu crypto bot (Groq LLaMA 3.1 70B)*\nChọn chức năng:"


def build_main_menu_kb():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📈 Long", callback_data="MENU_LONG"),
                InlineKeyboardButton("📉 Short", callback_data="MENU_SHORT"),
            ],
            [
                InlineKeyboardButton("📊 Report", callback_data="MENU_REPORT"),
            ],
            [
                InlineKeyboardButton("📖 Help", callback_data="SHOW_HELP"),
            ],
        ]
    )


def build_long_menu_kb():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("BTC 1h", callback_data="PLAN|long|BTC|1h"),
                InlineKeyboardButton("ETH 1h", callback_data="PLAN|long|ETH|1h"),
            ],
            [
                InlineKeyboardButton("SOL 1h", callback_data="PLAN|long|SOL|1h"),
                InlineKeyboardButton("TRUMP 1h", callback_data="PLAN|long|TRUMP|1h"),
            ],
            [
                InlineKeyboardButton("⬅ Quay lại", callback_data="MENU_MAIN"),
            ],
        ]
    )


def build_short_menu_kb():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("BTC 1h", callback_data="PLAN|short|BTC|1h"),
                InlineKeyboardButton("ETH 1h", callback_data="PLAN|short|ETH|1h"),
            ],
            [
                InlineKeyboardButton("SOL 1h", callback_data="PLAN|short|SOL|1h"),
                InlineKeyboardButton("TRUMP 1h", callback_data="PLAN|short|TRUMP|1h"),
            ],
            [
                InlineKeyboardButton("⬅ Quay lại", callback_data="MENU_MAIN"),
            ],
        ]
    )


def build_report_menu_kb():
    # Thêm STRK, XRP, TAO, ICP, VIRTUAL vào menu REPORT
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("BTC", callback_data="REPORT|BTC"),
                InlineKeyboardButton("ETH", callback_data="REPORT|ETH"),
            ],
            [
                InlineKeyboardButton("SOL", callback_data="REPORT|SOL"),
                InlineKeyboardButton("TRUMP", callback_data="REPORT|TRUMP"),
            ],
            [
                InlineKeyboardButton("BTCDOM", callback_data="REPORT|BTCDOM"),
                InlineKeyboardButton("STRK", callback_data="REPORT|STRK"),
            ],
            [
                InlineKeyboardButton("XRP", callback_data="REPORT|XRP"),
                InlineKeyboardButton("TAO", callback_data="REPORT|TAO"),
            ],
            [
                InlineKeyboardButton("ICP", callback_data="REPORT|ICP"),
                InlineKeyboardButton("VIRTUAL", callback_data="REPORT|VIRTUAL"),
            ],
            [
                InlineKeyboardButton("⬅ Quay lại", callback_data="MENU_MAIN"),
            ],
        ]
    )


# ========= BUILD REPORT FULL (OHLC + FUNDING + OI + ORDERBOOK + DELTA) =========
def build_full_report_text(symbol: str) -> str:
    lines = [f"📊 Report *{symbol}*:"]
    # 1) OHLC + indicators theo từng timeframe
    for tf in TIMEFRAMES:
        try:
            ind = get_indicators(symbol, tf)
            lines.append(
                f"\n⏱ *{tf}*\n"
                f"• O/H/L/C: `{fmt_num(ind['open'])}` / `{fmt_num(ind['high'])}` / `{fmt_num(ind['low'])}` / `{fmt_num(ind['price'])}`\n"
                f"• Thay đổi vs close trước: `{fmt_num(ind['change_pct'], 2)}%`\n"
                f"• Biên độ (H-L)/C: `{fmt_num(ind['range_pct'], 2)}%`, Body%: `{fmt_num(ind['body_pct'], 2)}%`\n"
                f"• Volume: `{fmt_num(ind['vol'], 2)}`, Vol MA20: `{fmt_num(ind['vol_ma20'], 2)}`\n"
                f"• MA20 / MA50: `{fmt_num(ind['ma20'])}` / `{fmt_num(ind['ma50'])}`\n"
                f"• EMA20 / EMA50: `{fmt_num(ind['ema20'])}` / `{fmt_num(ind['ema50'])}`\n"
                f"• RSI14: `{fmt_num(ind['rsi14'], 2)}`, ATR14: `{fmt_num(ind['atr14'], 2)}`\n"
                f"• Vị trí trong range 14 nến: `{fmt_num(ind['range_pos_14'], 2)}%` (0% = đáy, 100% = đỉnh)"
            )
        except Exception as e:
            lines.append(f"\n⏱ {tf}: lỗi {e}")

    # 2) Funding
    fr_usdt, fr_coin = get_funding_rates(symbol)
    lines.append("\n———\n\n🧾 *Funding Rate:*")
    lines.append(
        f"• Funding rate hiện tại (USDT-M): `{fmt_num(fr_usdt, 6)}`"
    )
    lines.append(
        f"• Funding rate hiện tại (COIN-M): `{fmt_num(fr_coin, 6)}` (BTC/ETH mới có, alt thường = N/A)"
    )

    # 3) Open Interest
    oi_changes, oi_total = get_open_interest_stats(symbol)
    lines.append("\n📈 *Open Interest (USDT-M futures):*")
    for period in ["5m", "15m", "1h"]:
        last, diff, pct = oi_changes.get(period, (None, None, None))
        lines.append(
            f"• OI {period}: `{fmt_num(last, 2)}` | Δ: `{fmt_num(diff, 2)}` ({fmt_num(pct, 2)}%)"
        )
    lines.append(f"• OI tổng: `{fmt_num(oi_total, 2)}`")

    # 4) Orderbook imbalance
    bid_vol, ask_vol, net = get_orderbook_imbalance(symbol)
    lines.append("\n📚 *Orderbook Imbalance (USDT-M top 50 levels):*")
    lines.append(f"• Buy wall (bids): `{fmt_num(bid_vol, 2)}`")
    lines.append(f"• Sell wall (asks): `{fmt_num(ask_vol, 2)}`")
    lines.append(f"• Net imbalance (buy - sell): `{fmt_num(net, 2)}`")

    # 5) Volume Delta
    lines.append("\n⚔️ *Volume Delta (USDT-M):*")
    for mins, label in [(5, "5m"), (15, "15m"), (60, "1h")]:
        b, s, n = get_volume_delta(symbol, mins)
        lines.append(
            f"• Delta {label}: buy `{fmt_num(b, 2)}`, sell `{fmt_num(s, 2)}`, net `{fmt_num(n, 2)}`"
        )

    # 6) Liquidation heatmap (placeholder)
    lines.append(
        "\n🔥 *Liquidation Heatmap:* (placeholder)\n"
        "• Liquidation cluster gần nhất: (cần API riêng như Coinalyze / Coinglass)\n"
        "• Liquidity lớn ở trên: ...\n"
        "• Liquidity lớn ở dưới: ..."
    )

    return "\n".join(lines)


# ========= BASIC COMMANDS =========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        get_main_menu_text(),
        parse_mode=constants.ParseMode.MARKDOWN,
        reply_markup=build_main_menu_kb(),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        get_help_text(),
        parse_mode=constants.ParseMode.MARKDOWN,
    )


async def report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symbol_raw = context.args[0] if context.args else "BTC"
    symbol = normalize_symbol(symbol_raw)
    text = build_full_report_text(symbol)
    await update.message.reply_text(
        text,
        parse_mode=constants.ParseMode.MARKDOWN,
    )


# ========= PLAN BUILDERS =========
def build_long_plan(symbol, tf, entry=None):
    sh, slv, close = get_swing_levels(symbol, tf)
    ind = get_indicators(symbol, tf)
    price = close
    use_entry = entry if entry is not None else price
    sl = min(slv, use_entry * 0.995)
    risk = use_entry - sl
    if risk <= 0:
        risk = use_entry * 0.01
        sl = use_entry - risk
    tp1 = use_entry + risk * 1.5
    tp2 = use_entry + risk * 2
    return {
        "entry": use_entry,
        "price": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "sh": sh,
        "slv": slv,
        "rsi": ind["rsi14"],
        "ma20": ind["ma20"],
        "ma50": ind["ma50"],
    }


def build_short_plan(symbol, tf, entry=None):
    sh, slv, close = get_swing_levels(symbol, tf)
    ind = get_indicators(symbol, tf)
    price = close
    use_entry = entry if entry is not None else price
    sl = max(sh, use_entry * 1.005)
    risk = sl - use_entry
    if risk <= 0:
        risk = use_entry * 0.01
        sl = use_entry + risk
    tp1 = use_entry - risk * 1.5
    tp2 = use_entry - risk * 2
    return {
        "entry": use_entry,
        "price": price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "sh": sh,
        "slv": slv,
        "rsi": ind["rsi14"],
        "ma20": ind["ma20"],
        "ma50": ind["ma50"],
    }


def parse_entry_tf(args):
    """
    /longbtc
    /longbtc 62000
    /longbtc 62000 4h
    /longbtc 4h 62000
    """
    if not args:
        return None, "1h"
    if len(args) == 1:
        a = args[0]
        try:
            return float(a), "1h"
        except ValueError:
            return None, a
    a0, a1 = args[0], args[1]
    e0 = e1 = None
    try:
        e0 = float(a0)
    except ValueError:
        pass
    try:
        e1 = float(a1)
    except ValueError:
        pass
    if e0 is not None and e1 is None:
        return e0, a1
    if e1 is not None and e0 is None:
        return e1, a0
    return None, a0


# ========= SUGGEST PLAN (LONG/SHORT) =========
async def suggest_plan(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    sym_key: str,
    side: str,
    tf: str,
    entry=None,
):
    symbol = normalize_symbol(sym_key)
    try:
        if side == "long":
            plan = build_long_plan(symbol, tf, entry)
        else:
            plan = build_short_plan(symbol, tf, entry)

        e = plan["entry"]
        p = plan["price"]
        sl = plan["sl"]
        tp1 = plan["tp1"]
        tp2 = plan["tp2"]
        sh = plan["sh"]
        slv = plan["slv"]
        rsi = plan["rsi"]
        ma20 = plan["ma20"]
        ma50 = plan["ma50"]

        if entry is not None:
            pnl = (p - e) / e * 100 if side == "long" else (e - p) / e * 100
            pnl_txt = f"Lệnh hiện tại ~ {pnl:+.2f}% so với entry\n"
        else:
            pnl_txt = "Chưa có entry thật, đây là kế hoạch tham khảo.\n"

        ai_text = ai_trade_view(
            symbol, side, tf, e, p, sl, tp1, tp2, rsi, ma20, ma50, sh, slv
        )

        text = (
            f"📌 {side.upper()} {symbol} ({tf})\n\n"
            f"Giá hiện tại: {fmt_num(p)}\n"
            f"Entry xét: {fmt_num(e)}\n"
            f"{pnl_txt}\n"
            f"Swing high: {fmt_num(sh)}\n"
            f"Swing low : {fmt_num(slv)}\n\n"
            f"SL : {fmt_num(sl)}\n"
            f"TP1: {fmt_num(tp1)}\n"
            f"TP2: {fmt_num(tp2)}\n\n"
            f"MA20: {fmt_num(ma20)}\n"
            f"MA50: {fmt_num(ma50)}\n"
            f"RSI14: {fmt_num(rsi, 2)}\n\n"
            f"🤖 Góc nhìn AI (Groq LLaMA 3.1 70B, chỉ THAM KHẢO):\n"
            f"{ai_text}"
        )

        await context.bot.send_message(chat_id=chat_id, text=text)

    except Exception as e:
        logger.error("suggest_plan error: %s", e)
        await context.bot.send_message(chat_id=chat_id, text=f"Lỗi phân tích: {e}")


# ========= LONG/SHORT COMMANDS =========
async def longbtc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry, tf = parse_entry_tf(context.args)
    await suggest_plan(context, update.effective_chat.id, "BTC", "long", tf, entry)


async def shortbtc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry, tf = parse_entry_tf(context.args)
    await suggest_plan(context, update.effective_chat.id, "BTC", "short", tf, entry)


async def longeth(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry, tf = parse_entry_tf(context.args)
    await suggest_plan(context, update.effective_chat.id, "ETH", "long", tf, entry)


async def shorteth(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry, tf = parse_entry_tf(context.args)
    await suggest_plan(context, update.effective_chat.id, "ETH", "short", tf, entry)


async def longsol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry, tf = parse_entry_tf(context.args)
    await suggest_plan(context, update.effective_chat.id, "SOL", "long", tf, entry)


async def shortsol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry, tf = parse_entry_tf(context.args)
    await suggest_plan(context, update.effective_chat.id, "SOL", "short", tf, entry)


async def longtrump(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry, tf = parse_entry_tf(context.args)
    await suggest_plan(context, update.effective_chat.id, "TRUMP", "long", tf, entry)


async def shorttrump(update: Update, context: ContextTypes.DEFAULT_TYPE):
    entry, tf = parse_entry_tf(context.args)
    await suggest_plan(context, update.effective_chat.id, "TRUMP", "short", tf, entry)


# ========= ALERTS (GIÁ) =========
async def alert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global alerts
    if len(context.args) < 4:
        await update.message.reply_text(
            "Cú pháp: /alert SYMBOL TF above|below PRICE\n"
            "Ví dụ: /alert BTC 1h below 60000\n"
            "       /alert BTC 1h above 65000"
        )
        return

    sym = context.args[0]
    tf = context.args[1]
    direction = context.args[2].lower()
    try:
        lv = float(context.args[3])
    except ValueError:
        await update.message.reply_text("PRICE phải là số, ví dụ 60000")
        return

    alerts.append(
        {
            "type": "price",
            "user_id": update.effective_user.id,
            "chat_id": update.effective_chat.id,
            "symbol": normalize_symbol(sym),
            "tf": tf,
            "dir": direction,
            "lv": lv,
            "active": True,
        }
    )
    save_alerts()
    await update.message.reply_text("✅ Đã đặt alert giá.")


async def check_alerts(context: ContextTypes.DEFAULT_TYPE):
    global alerts
    if not alerts:
        return

    price_cache = {}
    changed = False

    for a in alerts:
        if not a.get("active", True):
            continue
        if a["type"] != "price":
            continue

        sym = a["symbol"]
        tf = a["tf"]
        direction = a["dir"]
        level = a["lv"]

        if sym not in price_cache:
            try:
                price_cache[sym] = get_price(sym)
            except Exception:
                continue

        price = price_cache[sym]
        triggered = False

        if direction == "below" and price <= level:
            triggered = True
        elif direction == "above" and price >= level:
            triggered = True

        if triggered:
            msg = (
                f"⚠️ Alert giá cho {sym} khung {tf} đã kích hoạt!\n"
                f"Giá hiện tại: {fmt_num(price)}\n"
                f"Điều kiện: {direction} {level}"
            )
            try:
                await context.bot.send_message(chat_id=a["chat_id"], text=msg)
            except Exception:
                pass
            a["active"] = False
            changed = True

    if changed:
        save_alerts()


# ========= CALLBACK HANDLER =========
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    chat_id = query.message.chat_id

    await query.answer()

    if data == "MENU_MAIN":
        await query.edit_message_text(
            get_main_menu_text(),
            parse_mode=constants.ParseMode.MARKDOWN,
            reply_markup=build_main_menu_kb(),
        )

    elif data == "MENU_LONG":
        await query.edit_message_text(
            "Chọn coin để LONG (1h):",
            reply_markup=build_long_menu_kb(),
        )

    elif data == "MENU_SHORT":
        await query.edit_message_text(
            "Chọn coin để SHORT (1h):",
            reply_markup=build_short_menu_kb(),
        )

    elif data == "MENU_REPORT":
        await query.edit_message_text(
            "Chọn coin để xem report:",
            reply_markup=build_report_menu_kb(),
        )

    elif data == "SHOW_HELP":
        await context.bot.send_message(
            chat_id,
            get_help_text(),
            parse_mode=constants.ParseMode.MARKDOWN,
        )

    elif data.startswith("PLAN|"):
        try:
            _, side, sym, tf = data.split("|")
        except ValueError:
            await context.bot.send_message(chat_id, "Callback PLAN lỗi format.")
            return
        await suggest_plan(context, chat_id, sym, side, tf)

    elif data.startswith("REPORT|"):
        try:
            _, sym = data.split("|")
        except ValueError:
            await context.bot.send_message(chat_id, "Callback REPORT lỗi format.")
            return

        symbol = normalize_symbol(sym)
        text = build_full_report_text(symbol)
        await context.bot.send_message(
            chat_id,
            text,
            parse_mode=constants.ParseMode.MARKDOWN,
        )


# ========= MAIN =========
if __name__ == "__main__":
    load_alerts()

    app = ApplicationBuilder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("report", report))

    app.add_handler(CommandHandler("longbtc", longbtc))
    app.add_handler(CommandHandler("shortbtc", shortbtc))
    app.add_handler(CommandHandler("longeth", longeth))
    app.add_handler(CommandHandler("shorteth", shorteth))
    app.add_handler(CommandHandler("longsol", longsol))
    app.add_handler(CommandHandler("shortsol", shortsol))
    app.add_handler(CommandHandler("longtrump", longtrump))
    app.add_handler(CommandHandler("shorttrump", shorttrump))

    app.add_handler(CommandHandler("alert", alert_cmd))

    # Inline callbacks
    app.add_handler(CallbackQueryHandler(callback_handler))

    # JobQueue check alerts
    job = app.job_queue
    job.run_repeating(check_alerts, interval=60, first=10)

    print("Bot đang chạy… Ctrl+C để dừng.")
    app.run_polling()
