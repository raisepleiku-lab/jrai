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
TOKEN = "8340989991:AAFbc5IiM5onGkvJDdzTrVzBgvseMrD-8xA"

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


# ========= HELP / MENU TEXT =========
def get_help_text():
    return (
        "📌 *Các lệnh chính:*\n"
        "/start – mở menu chính\n"
        "/help – xem hướng dẫn nhanh\n"
        "/core – report combo BTC + ETH + BTCDOM (3 tin riêng)\n"
        "/report BTC – report đa khung cho 1 đồng\n"
        "/btc, /eth, /sol, /trump, /btcdom, /strk, /xrp, /tao, /icp, /virtual – report nhanh từng coin\n\n"
        "Alert giá:\n"
        "  /alert BTC 1h below 60000\n"
        "  /alert BTC 1h above 65000\n\n"
        "⚠️ Các tin nhắn REPORT sẽ tự xoá sau 5 phút."
    )


def get_main_menu_text():
    return "🏠 *Menu crypto bot – chế độ REPORT ONLY*\nChọn chức năng:"


def build_main_menu_kb():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("📊 Report", callback_data="MENU_REPORT"),
            ],
            [
                InlineKeyboardButton("📖 Help", callback_data="SHOW_HELP"),
            ],
        ]
    )


def build_report_menu_kb():
    """
    Menu report:
    - Hàng đầu: combo BTC + ETH + BTCDOM
    - Các hàng dưới: từng coin
    """
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🔥 BTC + ETH + BTCDOM", callback_data="REPORT3|CORE"
                ),
            ],
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


# ========= AUTO DELETE REPORT MESSAGE =========
async def delete_message_job(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    chat_id = job.chat_id
    message_id = job.data["message_id"]
    try:
        await context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception as e:
        logger.warning("Failed to delete message: %s", e)


def schedule_auto_delete(context: ContextTypes.DEFAULT_TYPE, message, delay: int = 300):
    """
    Đặt job tự xoá message sau <delay> giây (mặc định 300s = 5 phút).
    Chỉ dùng cho các tin nhắn REPORT.
    """
    try:
        context.job_queue.run_once(
            delete_message_job,
            when=delay,
            chat_id=message.chat_id,
            data={"message_id": message.message_id},
        )
    except Exception as e:
        logger.warning("Failed to schedule auto delete: %s", e)


# ========= BUILD REPORT =========
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
    """
    /report BTC – report 1 đồng bất kỳ (gõ symbol làm arg).
    Tin nhắn trả về sẽ auto xoá sau 5 phút.
    """
    symbol_raw = context.args[0] if context.args else "BTC"
    symbol = normalize_symbol(symbol_raw)
    text = build_full_report_text(symbol)
    msg = await update.message.reply_text(
        text,
        parse_mode=constants.ParseMode.MARKDOWN,
    )
    schedule_auto_delete(context, msg)


async def core(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Lệnh /core → gửi 3 tin riêng: BTC, ETH, BTCDOM
    để tránh lỗi 'message is too long'.
    Các tin này sẽ auto xoá sau 5 phút.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "BTCDOMUSDT"]
    name_map = {
        "BTCUSDT": "BTC",
        "ETHUSDT": "ETH",
        "BTCDOMUSDT": "BTCDOM",
    }

    for sym in symbols:
        label = name_map.get(sym, sym)
        try:
            text = build_full_report_text(sym)
            msg = await update.message.reply_text(
                f"===== {label} =====\n{text}",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            schedule_auto_delete(context, msg)
        except Exception as e:
            msg = await update.message.reply_text(
                f"===== {label} =====\nLỗi report: {e}",
                parse_mode=constants.ParseMode.MARKDOWN,
            )
            schedule_auto_delete(context, msg)


# ========= COIN SHORT COMMANDS (/btc /eth ...) =========
async def coin_report_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler chung cho các lệnh:
    /btc /eth /sol /trump /btcdom /strk /xrp /tao /icp /virtual
    """
    cmd = update.message.text.lstrip("/").split()[0].upper()

    cmd_to_symbol = {
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

    symbol = cmd_to_symbol.get(cmd, "BTCUSDT")
    text = build_full_report_text(symbol)
    msg = await update.message.reply_text(
        text,
        parse_mode=constants.ParseMode.MARKDOWN,
    )
    schedule_auto_delete(context, msg)


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

    elif data.startswith("REPORT|"):
        # Report 1 coin qua inline button
        try:
            _, sym = data.split("|")
        except ValueError:
            await context.bot.send_message(chat_id, "Callback REPORT lỗi format.")
            return

        symbol = normalize_symbol(sym)
        text = build_full_report_text(symbol)
        msg = await context.bot.send_message(
            chat_id,
            text,
            parse_mode=constants.ParseMode.MARKDOWN,
        )
        schedule_auto_delete(context, msg)

    elif data == "REPORT3|CORE":
        # Gửi 3 tin: BTC, ETH, BTCDOM – auto xoá sau 5 phút
        symbols = ["BTCUSDT", "ETHUSDT", "BTCDOMUSDT"]
        name_map = {
            "BTCUSDT": "BTC",
            "ETHUSDT": "ETH",
            "BTCDOMUSDT": "BTCDOM",
        }

        for sym in symbols:
            label = name_map.get(sym, sym)
            try:
                text = build_full_report_text(sym)
                msg = await context.bot.send_message(
                    chat_id,
                    f"===== {label} =====\n{text}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                schedule_auto_delete(context, msg)
            except Exception as e:
                msg = await context.bot.send_message(
                    chat_id,
                    f"===== {label} =====\nLỗi report: {e}",
                    parse_mode=constants.ParseMode.MARKDOWN,
                )
                schedule_auto_delete(context, msg)


# ========= SET SLASH COMMANDS CHO GỢI Ý "/" =========
async def post_init(app):
    commands = [
        # Ưu tiên core trước
        BotCommand("core", "Report BTC + ETH + BTCDOM (3 tin)"),
        # Report từng coin
        BotCommand("btc", "Report BTC"),
        BotCommand("eth", "Report ETH"),
        BotCommand("sol", "Report SOL"),
        BotCommand("trump", "Report TRUMP"),
        BotCommand("btcdom", "Report BTCDOM (BTC.D)"),
        BotCommand("strk", "Report STRK"),
        BotCommand("xrp", "Report XRP"),
        BotCommand("tao", "Report TAO"),
        BotCommand("icp", "Report ICP"),
        BotCommand("virtual", "Report VIRTUAL"),
        # Lệnh chung & tiện ích
        BotCommand("report", "Báo cáo 1 đồng bất kỳ (VD: /report BTC)"),
        BotCommand("alert", "Đặt alert giá (VD: /alert BTC 1h below 60000)"),
        BotCommand("start", "Mở menu chính"),
        BotCommand("help", "Xem hướng dẫn nhanh"),
    ]
    try:
        await app.bot.set_my_commands(commands)
        logger.info("Slash commands set successfully.")
    except Exception as e:
        logger.warning("Failed to set slash commands: %s", e)


# ========= MAIN =========
if __name__ == "__main__":
    load_alerts()

    app = (
        ApplicationBuilder()
        .token(TOKEN)
        .post_init(post_init)
        .build()
    )

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("report", report))
    app.add_handler(CommandHandler("core", core))

    # Coin short commands
    app.add_handler(CommandHandler("btc", coin_report_cmd))
    app.add_handler(CommandHandler("eth", coin_report_cmd))
    app.add_handler(CommandHandler("sol", coin_report_cmd))
    app.add_handler(CommandHandler("trump", coin_report_cmd))
    app.add_handler(CommandHandler("btcdom", coin_report_cmd))
    app.add_handler(CommandHandler("strk", coin_report_cmd))
    app.add_handler(CommandHandler("xrp", coin_report_cmd))
    app.add_handler(CommandHandler("tao", coin_report_cmd))
    app.add_handler(CommandHandler("icp", coin_report_cmd))
    app.add_handler(CommandHandler("virtual", coin_report_cmd))

    app.add_handler(CommandHandler("alert", alert_cmd))

    # Inline callbacks
    app.add_handler(CallbackQueryHandler(callback_handler))

    # JobQueue check alerts
    job = app.job_queue
    job.run_repeating(check_alerts, interval=60, first=10)

    print("Bot đang chạy… Ctrl+C để dừng.")
    app.run_polling()
