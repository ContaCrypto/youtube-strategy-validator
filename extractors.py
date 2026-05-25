from __future__ import annotations

import re
from typing import List, Optional


_SUBJECTIVE_TERMS: set = {
    "confirmation",
    "momentum",
    "market structure",
    "strong candle",
    "weak candle",
    "clean setup",
    "good entry",
    "wait for reaction",
    "looks bullish",
    "looks bearish",
    "feels like",
    "smart money",
    "liquidity grab",
    "order flow",
    "fair value gap",
    "imbalance",
    "respecting level",
    "clean break",
    "institutional",
    "killzone",
    "high probability",
    "high quality",
    "price action confirmation",
}

_PROMOTIONAL_TERMS: set = {
    "holy grail",
    "works on any market",
    "never loses",
    "never lose",
    "always works",
    "guaranteed profit",
    "risk free",
    "risk-free",
    "secret strategy",
    "life changing",
    "quit your job",
    "financial freedom",
    "passive income",
    "become rich",
    "easy money",
    "make money fast",
    "no risk",
    "can't lose",
    "perfect strategy",
    "100% win rate",
    "always profitable",
}

_EXACT_SIGNALS: list = [
    "cross",
    "above",
    "below",
    "greater than",
    "less than",
    "%",
    ">=",
    "<=",
    "pips",
    "points",
]

# Visual indicator signal phrases that describe mechanically actionable entry cues
# (vague — no formula given — but not subjective)
_VISUAL_SIGNAL_ENTRY_PHRASES: list = [
    "turns green",
    "goes green",
    "green bar",
    "green candle",
    "green arrow",
    "green dot",
    "buy signal",
    "buy arrow",
    "buy label",
    "signal appears",
    "signal fires",
    "arrow appears",
    "changes color",
    "changes colour",
]

# Visual indicator signal phrases that describe mechanically actionable exit cues
_VISUAL_SIGNAL_EXIT_PHRASES: list = [
    "turns red",
    "goes red",
    "red bar",
    "red candle",
    "red arrow",
    "red dot",
    "sell signal",
    "sell arrow",
    "sell label",
    "opposite signal",
    "exit signal",
    "changes color",
    "changes colour",
]


def find_indicators(text: str) -> List[str]:
    indicator_patterns = {
        "RSI": r"\brsi\b|relative strength index",
        "MACD": r"\bmacd\b",
        "EMA": r"\bema\b|exponential moving average",
        "SMA": r"\bsma\b|simple moving average",
        "VWAP": r"\bvwap\b",
        "Bollinger Bands": r"bollinger",
        "Stochastic": r"stochastic",
        "ATR": r"\batr\b|average true range",
        "Volume": r"\bvolume\b",
        "Support/Resistance": r"support|resistance",
        "Fibonacci": r"fibonacci|fib retracement",
    }

    lowered = text.lower()
    return [
        name
        for name, pattern in indicator_patterns.items()
        if re.search(pattern, lowered)
    ]


def detect_timeframe(text: str) -> Optional[str]:
    match = re.search(
        r"\b(1m|3m|5m|15m|30m|1h|2h|4h|1d|daily|weekly|monthly|one minute|five minute|fifteen minute|hourly)\b",
        text.lower(),
    )
    return match.group(1) if match else None


def detect_market(text: str) -> Optional[str]:
    lowered = text.lower()

    if any(
        word in lowered
        for word in ["bitcoin", "btc", "ethereum", "eth", "crypto", "usdt"]
    ):
        return "crypto"

    if any(
        word in lowered
        for word in ["forex", "eurusd", "gbpusd", "xauusd", "nas100", "us30"]
    ):
        return "forex/cfd"

    if any(
        word in lowered for word in ["stock", "stocks", "spy", "qqq", "nasdaq", "s&p"]
    ):
        return "stocks/indices"

    return None


def detect_strategy_type(text: str) -> Optional[str]:
    lowered = text.lower()

    if "trend line" in lowered or "trendline" in lowered:
        return "trendline_breakout"

    if "break and retest" in lowered:
        return "break_and_retest"

    if "order block" in lowered:
        return "smart_money"

    if "rsi" in lowered:
        return "indicator_strategy"

    if "ema" in lowered or "moving average" in lowered:
        return "moving_average"

    if "scalp" in lowered:
        return "scalping"

    return None


def detect_promotional_claims(text: str) -> List[str]:
    """Return sorted list of promotional phrases found in the transcript."""
    lowered = text.lower()
    return sorted({term for term in _PROMOTIONAL_TERMS if term in lowered})


def detect_session_filter(text: str) -> Optional[str]:
    """Detect if a specific trading session or time filter is mentioned."""
    lowered = text.lower()
    if any(w in lowered for w in ["london session", "london open", "london killzone"]):
        return "London session"
    if any(
        w in lowered
        for w in ["new york session", "ny session", "new york open", "ny open"]
    ):
        return "New York session"
    if any(w in lowered for w in ["asian session", "tokyo session", "asia session"]):
        return "Asian session"
    m = re.search(r"\b(\d{1,2}:\d{2})\s*(am|pm|utc|gmt|est|pst)\b", lowered)
    if m:
        return f"Specific time: {m.group(0)}"
    return None


def detect_backtest_evidence(text: str) -> Optional[str]:
    """Return the first mention of backtest data, win rate, or sample size."""
    lowered = text.lower()
    for pattern in [
        r"(\d+\.?\d*)\s*%\s*(win rate|accuracy|profitable|success rate)",
        r"backtest(?:ed|ing)?\s+[^.!?]{0,60}",
        r"(\d{2,})\s+(trades?|samples?)\s+[^.!?]{0,40}",
        r"(forward test|paper trade|live test)[^.!?]{0,60}",
    ]:
        m = re.search(pattern, lowered)
        if m:
            return m.group(0).strip()
    return None
