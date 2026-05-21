"""
AI YouTube Strategy Validator MVP

Fixed version:
- Does NOT require FastAPI just to run or test core logic.
- FastAPI is optional and only used if installed.
- Core functions can run in sandboxed environments.
- Includes built-in tests for URL parsing and rule scoring.

What it does:
1. Takes a YouTube URL
2. Extracts a video ID
3. Optionally fetches transcript if youtube-transcript-api is installed
4. Optionally uses OpenAI if openai is installed and OPENAI_API_KEY exists
5. Falls back to a local heuristic validator when AI dependencies are unavailable
6. Can run as CLI, tests, or FastAPI app

Run tests:
    python app.py --test

Run CLI with transcript text:
    python app.py --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --transcript "Buy when RSI crosses above 30. Sell when RSI crosses below 70. Stop loss 2%. Take profit 4%."

Run CLI and try fetching YouTube transcript:
    pip install youtube-transcript-api
    python app.py --url "https://www.youtube.com/watch?v=VIDEO_ID"

Run API only if FastAPI is installed:
    pip install fastapi uvicorn youtube-transcript-api openai pydantic
    set OPENAI_API_KEY=your_key_here
    uvicorn app:app --reload

API endpoint:
    POST /validate
    {
      "youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import unittest
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


try:
    import db as _db

    _db.init_db()
except Exception as _db_init_err:
    logging.warning("DB init failed: %s", _db_init_err)
    _db = None


from reports import print_score_bar

# Optional dependencies. The app must not crash if these are missing.
try:
    from fastapi import FastAPI, HTTPException, Request, Form
    from fastapi.responses import HTMLResponse
    from fastapi.templating import Jinja2Templates
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None
    HTTPException = None
    Request = None
    Form = None
    HTMLResponse = None
    Jinja2Templates = None

try:
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
    BaseModel = None
    Field = None

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
except ModuleNotFoundError:  # pragma: no cover
    YouTubeTranscriptApi = None
    TranscriptsDisabled = Exception
    NoTranscriptFound = Exception

try:
    from openai import OpenAI
except ModuleNotFoundError:  # pragma: no cover
    OpenAI = None


class StrategyValidatorError(Exception):
    """Base app error."""


class TranscriptUnavailableError(StrategyValidatorError):
    """Raised when transcript extraction is unavailable or fails."""


@dataclass
class StrategyRule:
    category: str
    rule: str
    clarity: str


@dataclass
class StrategyExtraction:
    strategy_name: Optional[str] = None
    market: Optional[StrategyRule] = None
    timeframe: Optional[StrategyRule] = None

    indicators: List[StrategyRule] = field(default_factory=list)

    entry_rules_long: List[StrategyRule] = field(default_factory=list)
    entry_rules_short: List[StrategyRule] = field(default_factory=list)
    exit_rules: List[StrategyRule] = field(default_factory=list)

    stop_loss: StrategyRule = field(
        default_factory=lambda: StrategyRule("stop_loss", "Missing", "missing")
    )

    take_profit: StrategyRule = field(
        default_factory=lambda: StrategyRule("take_profit", "Missing", "missing")
    )

    risk_management: StrategyRule = field(
        default_factory=lambda: StrategyRule("risk_management", "Missing", "missing")
    )

    position_sizing: StrategyRule = field(
        default_factory=lambda: StrategyRule("position_sizing", "Missing", "missing")
    )

    repainting_risk: StrategyRule = field(
        default_factory=lambda: StrategyRule("repainting_risk", "Unknown", "unknown")
    )

    missing_information: List[StrategyRule] = field(default_factory=list)
    subjective_terms: List[StrategyRule] = field(default_factory=list)

    coding_readiness_score: int = 0

    scam_or_cherry_pick_warning: StrategyRule = field(
        default_factory=lambda: StrategyRule("warning", "Not assessed", "unknown")
    )

    summary: str = ""

    failure_reasons: List[StrategyRule] = field(default_factory=list)

    strategy_type: Optional[str] = None

    pine_script_ready: bool = False
    confidence_score: int = 0

    session_filter: StrategyRule = field(
        default_factory=lambda: StrategyRule(
            "session_filter", "Not specified", "missing"
        )
    )
    backtest_evidence: StrategyRule = field(
        default_factory=lambda: StrategyRule(
            "backtest_evidence", "None mentioned", "missing"
        )
    )
    promotional_claims: List[str] = field(default_factory=list)

    entry_quality_score: int = 0
    exit_quality_score: int = 0
    risk_quality_score: int = 0
    automation_feasibility_score: int = 0
    hype_risk_score: int = 0
    backtest_evidence_score: int = 0
    formalization_score: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_video_id(url: str) -> str:
    """Extract a YouTube video ID from common YouTube URL formats."""
    if not url or not isinstance(url, str):
        raise ValueError("YouTube URL must be a non-empty string")

    patterns = [
        r"(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/live/)([a-zA-Z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    # Accept raw video ID for easier testing.
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url

    raise ValueError("Could not extract YouTube video ID")


def get_transcript(video_id: str, languages: Optional[List[str]] = None) -> str:
    """Fetch transcript if youtube-transcript-api is installed and captions exist.

    Supports youtube-transcript-api 1.x, where the API is instance-based:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=[...])

    Also keeps compatibility with older versions that had:
        YouTubeTranscriptApi.get_transcript(...)
    """
    if YouTubeTranscriptApi is None:
        raise TranscriptUnavailableError(
            "youtube-transcript-api is not installed. Install it or pass transcript text directly."
        )

    languages = languages or ["en", "sv"]

    try:
        # New youtube-transcript-api 1.x syntax
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=languages)
        snippets = list(fetched)
        text = " ".join(getattr(item, "text", "") for item in snippets)
        return text[:45000]
    except Exception as new_api_exc:
        # Older youtube-transcript-api syntax fallback
        try:
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=languages
                )
                text = " ".join(item.get("text", "") for item in transcript)
                return text[:45000]
        except NoTranscriptFound as exc:
            raise TranscriptUnavailableError(f"No transcript found: {exc}") from exc
        except TranscriptsDisabled as exc:
            raise TranscriptUnavailableError(
                "Transcripts are disabled for this video"
            ) from exc
        except Exception as old_api_exc:
            raise TranscriptUnavailableError(
                f"Transcript extraction failed. New API error: {new_api_exc}. Old API error: {old_api_exc}"
            ) from old_api_exc

        raise TranscriptUnavailableError(f"Transcript extraction failed: {new_api_exc}")


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


def extract_rules_with_keywords(text: str) -> StrategyExtraction:
    """
    Improved heuristic fallback validator.
    No external dependencies required.
    """
    lowered = text.lower()
    indicators = find_indicators(text)
    timeframe = detect_timeframe(text)
    market = detect_market(text)
    strategy_type = detect_strategy_type(text)
    promotional_claims = detect_promotional_claims(text)

    subjective_terms = sorted({term for term in _SUBJECTIVE_TERMS if term in lowered})

    entry_rules_long: List[StrategyRule] = []
    entry_rules_short: List[StrategyRule] = []
    exit_rules: List[StrategyRule] = []

    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        s = sentence.strip()
        if not s:
            continue
        sl = s.lower()

        has_subj = any(term in sl for term in subjective_terms)
        has_exact = any(sig in sl for sig in _EXACT_SIGNALS)

        if has_subj:
            clarity = "subjective"
        elif has_exact:
            clarity = "exact"
        else:
            clarity = "vague"

        if any(w in sl for w in ["buy", "long", "enter long", "go long"]):
            entry_rules_long.append(StrategyRule("long_entry", s, clarity))

        if re.search(r"\b(short|sell short|enter short|go short)\b", sl):
            entry_rules_short.append(StrategyRule("short_entry", s, clarity))

        if any(
            w in sl
            for w in [
                "exit",
                "close",
                "sell when",
                "take profit",
                "stop loss",
                "stop out",
            ]
        ):
            e_clarity = clarity
            if clarity != "subjective" and any(
                w in sl
                for w in [
                    "cross",
                    "above",
                    "below",
                    "%",
                    "atr",
                    "risk reward",
                    "r:r",
                    "1:",
                    "2:",
                ]
            ):
                e_clarity = "exact"
            exit_rules.append(StrategyRule("exit", s, e_clarity))

    # ── Stop loss ─────────────────────────────────────────────────────────────
    stop_loss = StrategyRule("stop_loss", "Missing", "missing")
    stop_match = re.search(
        r"(stop[\s-]?loss|stop out|\bsl\b)[^.!?]{0,100}", text, flags=re.IGNORECASE
    )
    if stop_match:
        t = stop_match.group(0).strip()
        stop_loss = StrategyRule(
            "stop_loss",
            t,
            "exact" if re.search(r"\d|atr|%|pips?|points?", t.lower()) else "vague",
        )

    # ── Take profit ───────────────────────────────────────────────────────────
    take_profit = StrategyRule("take_profit", "Missing", "missing")
    tp_match = re.search(
        r"(take[\s-]?profits?|profit target|\btp\b|\btarget\b)[^.!?]{0,100}",
        text,
        flags=re.IGNORECASE,
    )
    if tp_match:
        t = tp_match.group(0).strip()
        take_profit = StrategyRule(
            "take_profit",
            t,
            "exact"
            if re.search(r"\d|%|risk[\s-]reward|r:r|1r|2r|3r|pips?", t.lower())
            else "vague",
        )

    # ── Risk management ────────────────────────────────────────────────────────
    risk_management = StrategyRule("risk_management", "Missing", "missing")
    risk_match = re.search(
        r"(risk[\s-]?management|risk per trade|max[\s-]?risk|risk no more|never risk|\brisk\s+\d+%)[^.!?]{0,120}",
        text,
        flags=re.IGNORECASE,
    )
    if risk_match:
        t = risk_match.group(0).strip()
        risk_management = StrategyRule(
            "risk_management",
            t,
            "exact" if re.search(r"\d|%", t) else "vague",
        )

    # ── Position sizing ────────────────────────────────────────────────────────
    position_sizing = StrategyRule("position_sizing", "Missing", "missing")
    size_match = re.search(
        r"(position[\s-]?siz|lot[\s-]?siz|contract[\s-]?siz|trade[\s-]?siz|risk per)[^.!?]{0,120}",
        text,
        flags=re.IGNORECASE,
    )
    if size_match:
        t = size_match.group(0).strip()
        position_sizing = StrategyRule(
            "position_sizing",
            t,
            "exact" if re.search(r"\d|%", t) else "vague",
        )

    # ── Session filter ─────────────────────────────────────────────────────────
    session_text = detect_session_filter(text)
    session_filter = StrategyRule(
        "session_filter",
        session_text if session_text else "Not specified",
        "exact" if session_text else "missing",
    )

    # ── Backtest evidence ──────────────────────────────────────────────────────
    bt_text = detect_backtest_evidence(text)
    backtest_evidence = StrategyRule(
        "backtest_evidence",
        bt_text if bt_text else "None mentioned",
        "exact" if bt_text else "missing",
    )

    # ── Repainting risk ────────────────────────────────────────────────────────
    repainting_risk = "Unknown"
    if re.search(r"\brepaint", lowered):
        repainting_risk = "Mentioned in transcript. Needs manual review."
    elif any(
        w in lowered
        for w in ["pivot", "zigzag", "fractal", "future candle", "lookahead"]
    ):
        repainting_risk = "Possible repainting risk due to indicator type."

    # ── Missing information ────────────────────────────────────────────────────
    missing = []
    if not timeframe:
        missing.append("Timeframe")
    if not indicators:
        missing.append("Indicators")
    if not entry_rules_long and not entry_rules_short:
        missing.append("Entry rules")
    if not exit_rules:
        missing.append("Exit rules")
    if stop_loss.clarity == "missing":
        missing.append("Stop loss")
    if take_profit.clarity == "missing":
        missing.append("Take profit")
    if risk_management.clarity == "missing":
        missing.append("Risk management")
    if position_sizing.clarity == "missing":
        missing.append("Position sizing")

    # ── Coding readiness score ─────────────────────────────────────────────────
    score = 0
    if timeframe:
        score += 10
    if indicators:
        score += 10
    if entry_rules_long or entry_rules_short:
        score += 15
        all_entries = entry_rules_long + entry_rules_short
        if any(r.clarity == "exact" for r in all_entries):
            score += 10
    if exit_rules:
        score += 10
        if any(r.clarity == "exact" for r in exit_rules):
            score += 5
    if stop_loss.clarity == "exact":
        score += 15
    if take_profit.clarity == "exact":
        score += 10
    if risk_management.clarity == "exact":
        score += 10
    if position_sizing.clarity == "exact":
        score += 5

    score -= len(subjective_terms) * 4
    score -= len(promotional_claims) * 8
    score = max(0, min(100, score))

    # ── Failure reasons ────────────────────────────────────────────────────────
    failure_reasons = []
    if not timeframe:
        failure_reasons.append("Missing timeframe")
    if not indicators:
        failure_reasons.append("No indicators detected")
    if not entry_rules_long and not entry_rules_short:
        failure_reasons.append("No clear entry rules")
    if not exit_rules:
        failure_reasons.append("No clear exit rules")
    if stop_loss.clarity != "exact":
        failure_reasons.append("No exact stop loss")
    if take_profit.clarity != "exact":
        failure_reasons.append("No exact take profit")
    if risk_management.clarity != "exact":
        failure_reasons.append("No exact risk management")
    if position_sizing.clarity != "exact":
        failure_reasons.append("No exact position sizing")
    if subjective_terms:
        failure_reasons.append(
            f"Contains subjective language: {', '.join(subjective_terms[:3])}"
        )
    if promotional_claims:
        failure_reasons.append(
            f"Contains promotional claims: {', '.join(promotional_claims[:2])}"
        )

    # ── Warning ────────────────────────────────────────────────────────────────
    if promotional_claims or score < 30:
        warning = "High warning. Strategy contains promotional language or lacks testable rules."
    elif score < 50:
        warning = "High warning. Strategy is not code-ready and may be cherry-picked or too vague."
    elif score < 75:
        warning = "Medium warning. Strategy has testable parts but important assumptions are missing."
    else:
        warning = "Low warning. Rules look partly testable."

    # ── Pine Script ready ──────────────────────────────────────────────────────
    pine_script_ready = (
        len(missing) <= 2
        and len(subjective_terms) == 0
        and len(promotional_claims) == 0
        and stop_loss.clarity == "exact"
        and take_profit.clarity == "exact"
        and bool(entry_rules_long or entry_rules_short)
    )

    # ── Confidence score ───────────────────────────────────────────────────────
    confidence_score = score
    if subjective_terms:
        confidence_score -= 15
    if promotional_claims:
        confidence_score -= 20
    if len(failure_reasons) >= 5:
        confidence_score -= 15
    if pine_script_ready:
        confidence_score += 15
    if backtest_evidence.clarity == "exact":
        confidence_score += 10
    confidence_score = max(0, min(100, confidence_score))

    # ── Granular quality sub-scores ────────────────────────────────────────────
    all_entries = entry_rules_long + entry_rules_short
    exact_entries = [r for r in all_entries if r.clarity == "exact"]
    entry_q = 0
    if all_entries:
        entry_q += 40
        entry_q += int(30 * len(exact_entries) / len(all_entries))
    if timeframe:
        entry_q += 15
    if indicators:
        entry_q += 15
    entry_q = min(100, entry_q)

    exact_exits = [r for r in exit_rules if r.clarity == "exact"]
    exit_q = 0
    if exit_rules:
        exit_q += 20
        exit_q += int(20 * len(exact_exits) / len(exit_rules))
    if stop_loss.clarity == "exact":
        exit_q += 30
    elif stop_loss.clarity == "vague":
        exit_q += 10
    if take_profit.clarity == "exact":
        exit_q += 30
    elif take_profit.clarity == "vague":
        exit_q += 10
    exit_q = min(100, exit_q)

    risk_q = 0
    if risk_management.clarity == "exact":
        risk_q += 50
    elif risk_management.clarity == "vague":
        risk_q += 20
    if position_sizing.clarity == "exact":
        risk_q += 50
    elif position_sizing.clarity == "vague":
        risk_q += 20
    risk_q = min(100, risk_q)

    hype_risk = min(100, len(promotional_claims) * 20 + len(subjective_terms) * 5)

    bt_score = 0
    if backtest_evidence.clarity == "exact":
        bt_score = 70
        if re.search(r"\d+\.?\d*\s*%\s*(win rate|accuracy|profitable)", lowered):
            bt_score = 90
    elif re.search(r"\bbacktest", lowered):
        bt_score = 30

    auto_feas = max(0, min(100, int((entry_q + exit_q + risk_q) / 3) - hype_risk // 3))

    _clarity_val = {
        "exact": 100,
        "vague": 40,
        "subjective": 10,
        "missing": 0,
        "unknown": 20,
    }
    formal_base = (
        sum(
            _clarity_val.get(f.clarity, 0)
            for f in [stop_loss, take_profit, risk_management, position_sizing]
        )
        // 4
    )
    formal_base = (formal_base + entry_q) // 2 if all_entries else formal_base
    formalization = max(0, min(100, formal_base - len(subjective_terms) * 3))

    return StrategyExtraction(
        strategy_name=None,
        market=market,
        timeframe=timeframe,
        strategy_type=strategy_type,
        indicators=indicators,
        entry_rules_long=entry_rules_long,
        entry_rules_short=entry_rules_short,
        exit_rules=exit_rules,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_management=risk_management,
        position_sizing=position_sizing,
        session_filter=session_filter,
        backtest_evidence=backtest_evidence,
        repainting_risk=repainting_risk,
        missing_information=missing,
        subjective_terms=subjective_terms,
        promotional_claims=promotional_claims,
        coding_readiness_score=score,
        scam_or_cherry_pick_warning=warning,
        summary="Heuristic analysis. Install OpenAI package and set OPENAI_API_KEY for AI-powered extraction.",
        failure_reasons=failure_reasons,
        confidence_score=confidence_score,
        pine_script_ready=pine_script_ready,
        entry_quality_score=entry_q,
        exit_quality_score=exit_q,
        risk_quality_score=risk_q,
        automation_feasibility_score=auto_feas,
        hype_risk_score=hype_risk,
        backtest_evidence_score=bt_score,
        formalization_score=formalization,
    )


def analyze_strategy_with_openai(transcript_text: str) -> Optional[StrategyExtraction]:
    """Use OpenAI if the dependency and API key are available. Otherwise return None."""
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return None

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    schema_prompt = """You are a strict, expert trading strategy auditor.

Your job: analyze a YouTube trading video transcript and extract ONLY explicitly stated, measurable rules.

EXTRACTION RULES:
1. Extract ONLY what is explicitly stated. Do NOT invent, infer, or fill in missing information.
2. If a rule exists but uses no exact numbers or conditions, mark clarity as "vague".
3. If a rule uses discretionary or unmeasurable language, mark clarity as "subjective".
4. If information is completely absent, mark clarity as "missing" and use rule text "Missing".
5. Every rule object must be: {"category": "...", "rule": "...", "clarity": "exact|vague|subjective|missing"}

WHAT IS SUBJECTIVE (clarity: "subjective") — flag these exact terms when found:
confirmation, momentum, market structure, strong candle, weak candle, smart money,
liquidity grab, clean setup, good entry, wait for reaction, high probability,
respecting the level, clean break, institutional move, order flow, fair value gap,
looks bullish, looks bearish, feels like a good entry, price action confirmation.

WHAT IS PROMOTIONAL (add to promotional_claims list as exact phrase):
holy grail, never loses, 100% win rate, guaranteed profit, risk free, secret strategy,
works on any market, life changing, quit your job, financial freedom, passive income,
easy money, make money fast, no risk, can't lose, perfect strategy, always works.

WHAT IS EXACT (clarity: "exact") — requires measurable conditions:
- Specific numbers: "RSI below 30", "EMA(20) crosses above EMA(50)"
- Specific percentages: "stop loss 1.5%", "risk 1% of account per trade"
- Specific ratios: "1:2 risk-reward", "take profit at 2R"
- Specific price conditions: "price closes above the 200 SMA on the daily chart"

RETURN a JSON object with exactly these keys:
- strategy_name: string or null
- strategy_type: string or null
- market: string or null
- timeframe: string or null
- session_filter: rule object (trading session / time of day restriction, or missing)
- indicators: list of rule objects
- entry_rules_long: list of rule objects
- entry_rules_short: list of rule objects
- exit_rules: list of rule objects
- stop_loss: rule object
- take_profit: rule object
- risk_management: rule object
- position_sizing: rule object
- repainting_risk: string ("None detected" | "Possible" | "Confirmed" | "Unknown")
- backtest_evidence: rule object (win rate, sample size, backtest mention, or missing)
- missing_information: list of strings (components absent from the transcript)
- subjective_terms: list of strings (discretionary phrases found verbatim)
- promotional_claims: list of strings (hype/promotional phrases found verbatim)
- failure_reasons: list of strings (specific reasons this strategy is NOT automatable)
- coding_readiness_score: integer 0-100
- confidence_score: integer 0-100
- pine_script_ready: boolean
- entry_quality_score: integer 0-100
- exit_quality_score: integer 0-100
- risk_quality_score: integer 0-100
- automation_feasibility_score: integer 0-100
- hype_risk_score: integer 0-100 (100 = extremely promotional/scammy)
- backtest_evidence_score: integer 0-100
- formalization_score: integer 0-100
- scam_or_cherry_pick_warning: string
- summary: string (2-3 sentences: strengths, weaknesses, whether it can be backtested)

SCORING GUIDANCE:
- coding_readiness_score: 0-30 if vague/promotional, 30-60 if partially testable, 60-100 if mostly explicit
- hype_risk_score: add 15-20 per promotional claim found
- entry/exit/risk quality: 0 if missing, 40 if vague, 70+ if some exact rules, 90+ if fully defined
- Be strict: it is better to mark something missing than to invent a rule."""

    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
        input=[
            {"role": "system", "content": schema_prompt},
            {"role": "user", "content": transcript_text[:45000]},
        ],
        text={"format": {"type": "json_object"}},
    )

    raw_text = response.output_text
    data = json.loads(raw_text)

    def rule_from_any(value: Any, fallback_category: str) -> StrategyRule:
        """Accept AI output as dict, list, string, None, or other primitive.

        LLMs sometimes return a list for fields like stop_loss even when we asked for one object.
        This normalizes that into one StrategyRule instead of crashing.
        """
        if value is None:
            return StrategyRule(fallback_category, "Missing", "missing")

        if isinstance(value, list):
            if not value:
                return StrategyRule(fallback_category, "Missing", "missing")
            normalized_items = [
                rule_from_any(item, fallback_category) for item in value
            ]
            combined_rule = " | ".join(
                item.rule
                for item in normalized_items
                if item.rule and item.rule != "Missing"
            )
            if not combined_rule:
                combined_rule = "Missing"
            clarity_order = {"exact": 3, "vague": 2, "subjective": 1, "missing": 0}
            best_clarity = max(
                normalized_items, key=lambda item: clarity_order.get(item.clarity, 0)
            ).clarity
            return StrategyRule(fallback_category, combined_rule, best_clarity)

        if isinstance(value, dict):
            return StrategyRule(
                category=str(value.get("category", fallback_category)),
                rule=str(value.get("rule", "Missing")),
                clarity=str(value.get("clarity", "missing")),
            )

        if isinstance(value, str):
            return StrategyRule(
                fallback_category, value, "vague" if value.strip() else "missing"
            )

        return StrategyRule(fallback_category, str(value), "vague")

    def rules_list_from_any(value: Any, fallback_category: str) -> List[StrategyRule]:
        """Normalize AI output into a list of StrategyRule objects."""
        if value is None:
            return []
        if isinstance(value, list):
            return [rule_from_any(item, fallback_category) for item in value]
        return [rule_from_any(value, fallback_category)]

    return StrategyExtraction(
        strategy_name=data.get("strategy_name"),
        market=data.get("market"),
        timeframe=data.get("timeframe"),
        indicators=data.get("indicators", []),
        entry_rules_long=rules_list_from_any(
            data.get("entry_rules_long", []), "long_entry"
        ),
        entry_rules_short=rules_list_from_any(
            data.get("entry_rules_short", []), "short_entry"
        ),
        exit_rules=rules_list_from_any(data.get("exit_rules", []), "exit"),
        stop_loss=rule_from_any(data.get("stop_loss"), "stop_loss"),
        take_profit=rule_from_any(data.get("take_profit"), "take_profit"),
        risk_management=rule_from_any(data.get("risk_management"), "risk_management"),
        position_sizing=rule_from_any(data.get("position_sizing"), "position_sizing"),
        session_filter=rule_from_any(data.get("session_filter"), "session_filter"),
        backtest_evidence=rule_from_any(
            data.get("backtest_evidence"), "backtest_evidence"
        ),
        repainting_risk=data.get("repainting_risk", "Unknown"),
        missing_information=data.get("missing_information", []),
        subjective_terms=data.get("subjective_terms", []),
        promotional_claims=data.get("promotional_claims", []),
        coding_readiness_score=int(data.get("coding_readiness_score", 0)),
        scam_or_cherry_pick_warning=data.get(
            "scam_or_cherry_pick_warning", "Not assessed"
        ),
        summary=data.get("summary", ""),
        failure_reasons=data.get("failure_reasons", []),
        strategy_type=data.get("strategy_type"),
        pine_script_ready=bool(data.get("pine_script_ready", False)),
        confidence_score=int(
            data.get("confidence_score", data.get("coding_readiness_score", 0))
        ),
        entry_quality_score=int(data.get("entry_quality_score", 0)),
        exit_quality_score=int(data.get("exit_quality_score", 0)),
        risk_quality_score=int(data.get("risk_quality_score", 0)),
        automation_feasibility_score=int(data.get("automation_feasibility_score", 0)),
        hype_risk_score=int(data.get("hype_risk_score", 0)),
        backtest_evidence_score=int(data.get("backtest_evidence_score", 0)),
        formalization_score=int(data.get("formalization_score", 0)),
    )


def analyze_strategy(transcript_text: str) -> StrategyExtraction:
    ai_result = analyze_strategy_with_openai(transcript_text)
    if ai_result is not None:
        return ai_result
    return extract_rules_with_keywords(transcript_text)


def save_report(video_id: str, result: StrategyExtraction) -> str:
    """Save analysis result as JSON in the reports folder."""
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reports/{video_id}_{timestamp}_strategy_report.json"

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(result.to_dict(), file, indent=2, ensure_ascii=False)

    return filename


def generate_markdown_report(video_id: str, result: StrategyExtraction) -> str:
    os.makedirs("reports", exist_ok=True)

    filename = f"reports/{video_id}_report.md"

    def bullet_list(items):
        if not items:
            return "- None"

        formatted = []

        for item in items:
            if isinstance(item, StrategyRule):
                formatted.append(f"- {item.rule}")
            else:
                formatted.append(f"- {item}")

        return "\n".join(formatted)

    markdown = f"""# Strategy Analysis Report

## Strategy Information

- Name: {result.strategy_name or "Unknown"}
- Type: {result.strategy_type or "Unknown"}
- Market: {result.market or "Unknown"}
- Timeframe: {result.timeframe or "Unknown"}

---

## Scores

### Coding Readiness
{result.coding_readiness_score}/100

### Confidence Score
{result.confidence_score}/100

### Pine Script Ready
{result.pine_script_ready}

---

## Indicators

{bullet_list(result.indicators)}

---

## Missing Information

{bullet_list(result.missing_information)}

---

## Failure Reasons

{bullet_list(result.failure_reasons)}

---

## Subjective Terms

{bullet_list(result.subjective_terms)}

---

## Warning

{result.scam_or_cherry_pick_warning}

---

## Summary

{result.summary}
"""

    with open(filename, "w", encoding="utf-8") as file:
        file.write(markdown)

    return filename


def validate_strategy_core(
    youtube_url: str, transcript_text: Optional[str] = None, save: bool = False
) -> StrategyExtraction:
    video_id = extract_video_id(youtube_url)
    if transcript_text is None:
        transcript_text = get_transcript(video_id)
    result = analyze_strategy(transcript_text)

    if save:
        report_path = save_report(video_id, result)
        print(f"Report saved to: {report_path}", file=sys.stderr)

    return result


def _rule_text(item: Any) -> str:
    """Extract plain text from a rule that may be a dict, StrategyRule, or string."""
    if isinstance(item, dict):
        return item.get("rule", str(item))
    if hasattr(item, "rule"):
        return item.rule
    return str(item)


def compute_verdict(result: Dict[str, Any]) -> Dict[str, Any]:
    """Derive a verdict, automation difficulty, explanation and next steps from a result dict."""
    cr = int(result.get("coding_readiness_score", 0))
    cf = int(result.get("confidence_score", 0))

    raw_subj = result.get("subjective_terms", [])
    subj_count = len(raw_subj) if isinstance(raw_subj, list) else 0

    raw_missing = result.get("missing_information", [])
    missing_count = len(raw_missing) if isinstance(raw_missing, list) else 0

    raw_failure = result.get("failure_reasons", [])
    failure_count = len(raw_failure) if isinstance(raw_failure, list) else 0

    pine_ready = bool(result.get("pine_script_ready", False))

    # ── Verdict ──────────────────────────────────────────────────────────────
    if cf < 20 or (cr < 25 and failure_count >= 6):
        verdict = "Likely Scam"
        verdict_color = "red"
        difficulty = "Impossible"
        difficulty_color = "red"
    elif cf < 45 or cr < 35:
        verdict = "Too Vague To Automate"
        verdict_color = "orange"
        difficulty = "Hard"
        difficulty_color = "orange"
    elif cf >= 70 and cr >= 70 and subj_count == 0 and missing_count <= 1:
        verdict = "Fully Codable"
        verdict_color = "green"
        difficulty = "Easy"
        difficulty_color = "green"
    else:
        verdict = "Semi-Codable"
        verdict_color = "yellow"
        difficulty = "Medium"
        difficulty_color = "yellow"

    # ── Why this verdict ─────────────────────────────────────────────────────
    why_parts: List[str] = []

    if verdict == "Likely Scam":
        why_parts.append(
            "The strategy scores extremely low on both coding readiness and confidence. "
            "It provides almost no testable rules and is likely cherry-picked or fabricated."
        )
    elif verdict == "Too Vague To Automate":
        why_parts.append(
            f"With a coding readiness of {cr}/100 and confidence of {cf}/100, "
            "this strategy lacks the specificity needed for reliable automation."
        )
        if subj_count:
            why_parts.append(
                f"It contains {subj_count} subjective term(s) that cannot be translated into code."
            )
        if missing_count:
            why_parts.append(f"{missing_count} key component(s) are missing entirely.")
    elif verdict == "Semi-Codable":
        why_parts.append(
            f"The strategy scores {cr}/100 on coding readiness and {cf}/100 on confidence. "
            "Some rules are explicit enough to automate, but important gaps remain."
        )
        if subj_count:
            why_parts.append(
                f"{subj_count} subjective term(s) need to be replaced with exact conditions."
            )
        if missing_count:
            why_parts.append(f"{missing_count} component(s) are still undefined.")
        if pine_ready:
            why_parts.append("A partial Pine Script implementation is feasible.")
    else:
        why_parts.append(
            f"Coding readiness is {cr}/100 and confidence is {cf}/100. "
            "The rules are specific, measurable, and largely free of subjective language. "
            "This strategy can realistically be backtested and automated."
        )
        if pine_ready:
            why_parts.append("It is flagged as Pine Script ready.")

    why = " ".join(why_parts)

    # ── Next steps ───────────────────────────────────────────────────────────
    steps: List[str] = []

    missing_texts = (
        [_rule_text(m).lower() for m in raw_missing]
        if isinstance(raw_missing, list)
        else []
    )

    if any("stop loss" in t for t in missing_texts):
        steps.append("Define an exact stop loss (e.g. 1.5% below entry or 1× ATR)")
    if any("take profit" in t for t in missing_texts):
        steps.append("Define an exact take profit target or risk-reward ratio")
    if any("entry" in t for t in missing_texts):
        steps.append("Specify a precise entry trigger with a measurable condition")
    if any("position" in t for t in missing_texts):
        steps.append("Define position sizing (e.g. 1% account risk per trade)")
    if any("risk" in t for t in missing_texts):
        steps.append("Clarify risk management rules")
    if any("timeframe" in t for t in missing_texts):
        steps.append("State which timeframe the strategy runs on")
    if any("indicator" in t for t in missing_texts):
        steps.append("Name the specific indicator(s) used and their settings")
    if any("exit" in t for t in missing_texts):
        steps.append("Define clear exit conditions")
    if subj_count:
        steps.append(
            f"Replace {subj_count} subjective term(s) with measurable conditions"
        )
    if verdict == "Likely Scam":
        steps.append("Seek a strategy with independently verifiable backtest results")
    if not steps:
        if verdict in ("Fully Codable", "Semi-Codable"):
            steps.append(
                "Implement and forward-test the strategy in a paper trading account"
            )
            steps.append("Write unit tests for each entry and exit condition")
        else:
            steps.append(
                "Find a more rule-based strategy before attempting to automate"
            )

    return {
        "verdict": verdict,
        "verdict_color": verdict_color,
        "difficulty": difficulty,
        "difficulty_color": difficulty_color,
        "why": why,
        "next_steps": steps,
    }


# Optional FastAPI layer. This is only created when FastAPI and Pydantic exist.
if FastAPI is not None and BaseModel is not None:
    app = FastAPI(title="AI YouTube Strategy Validator")
    templates = Jinja2Templates(directory="templates")

    class ValidateRequest(BaseModel):
        youtube_url: str
        transcript_text: Optional[str] = None

    @app.post("/validate")
    def validate_strategy_api(request: ValidateRequest):
        try:
            result = validate_strategy_core(
                request.youtube_url, request.transcript_text
            )
            return result.to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except TranscriptUnavailableError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request):
        return templates.TemplateResponse(request=request, name="index.html")

    @app.post("/", response_class=HTMLResponse)
    def home_analyze(request: Request, youtube_url: str = Form(...)):
        try:
            result = validate_strategy_core(youtube_url)
            result_dict = result.to_dict()
            verdict = compute_verdict(result_dict)

            saved_id: Optional[int] = None
            if _db is not None:
                try:
                    video_id = extract_video_id(youtube_url)
                    saved_id = _db.save_analysis(
                        youtube_url, video_id, result_dict, verdict
                    )
                except Exception:
                    logging.exception("Failed to save analysis to DB")

            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "result": result_dict,
                    "verdict": verdict,
                    "url": youtube_url,
                    "saved_id": saved_id,
                },
            )
        except (ValueError, StrategyValidatorError) as exc:
            logging.exception("Strategy validation error")
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"error": str(exc), "url": youtube_url},
            )
        except Exception as exc:
            logging.exception("Unexpected error in home_analyze")
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={"error": f"Unexpected error: {exc}", "url": youtube_url},
            )

    @app.get("/history", response_class=HTMLResponse)
    def history(request: Request):
        rows = _db.get_all_analyses() if _db is not None else []
        return templates.TemplateResponse(
            request=request,
            name="history.html",
            context={"rows": rows},
        )

    @app.get("/analysis/{analysis_id}", response_class=HTMLResponse)
    def analysis_detail(request: Request, analysis_id: int):
        if _db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")
        rec = _db.get_analysis(analysis_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        verdict = {
            "verdict": rec["verdict"],
            "verdict_color": _verdict_color(rec["verdict"]),
            "difficulty": rec["automation_difficulty"],
            "difficulty_color": _verdict_color(rec["verdict"]),
        }
        return templates.TemplateResponse(
            request=request,
            name="analysis.html",
            context={"rec": rec, "verdict": verdict},
        )

else:
    app = None


def _verdict_color(verdict: Optional[str]) -> str:
    mapping = {
        "Likely Scam": "red",
        "Too Vague To Automate": "orange",
        "Semi-Codable": "yellow",
        "Fully Codable": "green",
    }
    return mapping.get(verdict or "", "orange")


class TestYouTubeStrategyValidator(unittest.TestCase):
    def test_extract_video_id_watch_url(self):
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )

    def test_extract_video_id_short_url(self):
        self.assertEqual(
            extract_video_id("https://youtu.be/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )

    def test_extract_video_id_shorts_url(self):
        self.assertEqual(
            extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            "dQw4w9WgXcQ",
        )

    def test_extract_video_id_raw_id(self):
        self.assertEqual(extract_video_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_extract_video_id_invalid_url(self):
        with self.assertRaises(ValueError):
            extract_video_id("https://example.com/not-youtube")

    def test_fallback_analysis_detects_basic_strategy(self):
        transcript = (
            "Use RSI on the 15m timeframe. Buy when RSI crosses above 30. "
            "Exit when RSI crosses below 70. Stop loss 2%. Take profit 4%. "
            "Risk 1% per trade. Position size should be based on account risk."
        )
        result = extract_rules_with_keywords(transcript)
        self.assertIn("RSI", result.indicators)
        self.assertEqual(result.timeframe, "15m")
        self.assertGreaterEqual(result.coding_readiness_score, 60)
        self.assertEqual(result.stop_loss.clarity, "exact")
        self.assertEqual(result.take_profit.clarity, "exact")

    def test_fallback_analysis_penalizes_vague_strategy(self):
        transcript = (
            "Wait for confirmation and enter with momentum. Exit when it feels weak."
        )
        result = extract_rules_with_keywords(transcript)
        self.assertIn("confirmation", result.subjective_terms)
        self.assertLess(result.coding_readiness_score, 60)
        self.assertIn("Stop loss", result.missing_information)
        self.assertIn("Take profit", result.missing_information)

    def test_promotional_claims_detection(self):
        transcript = (
            "This is a holy grail strategy that works on any market. "
            "You will achieve financial freedom and quit your job with this passive income strategy. "
            "It never loses and has a guaranteed profit system."
        )
        result = extract_rules_with_keywords(transcript)
        self.assertGreaterEqual(len(result.promotional_claims), 3)
        self.assertGreater(result.hype_risk_score, 40)
        self.assertLess(result.coding_readiness_score, 30)

    def test_vague_subjective_strategy(self):
        transcript = (
            "Wait for a clean setup with confirmation. "
            "Enter when you see a strong candle and market structure aligns. "
            "Exit when momentum shifts. Use good entries only."
        )
        result = extract_rules_with_keywords(transcript)
        self.assertGreater(len(result.subjective_terms), 2)
        self.assertLess(result.coding_readiness_score, 30)
        self.assertLess(result.entry_quality_score, 40)

    def test_exact_rsi_strategy(self):
        transcript = (
            "On the 1h chart, buy when RSI crosses above 30 and price is above the 200 EMA. "
            "Stop loss 1.5% below entry. Take profit at 3% above entry. "
            "Risk 1% of account per trade."
        )
        result = extract_rules_with_keywords(transcript)
        self.assertIn("RSI", result.indicators)
        self.assertEqual(result.timeframe, "1h")
        self.assertEqual(result.stop_loss.clarity, "exact")
        self.assertEqual(result.take_profit.clarity, "exact")
        self.assertGreaterEqual(result.coding_readiness_score, 60)
        self.assertGreaterEqual(result.entry_quality_score, 50)

    def test_missing_stop_loss(self):
        transcript = (
            "Buy when RSI crosses above 30 on the 15m chart. "
            "Exit when the trade looks weak."
        )
        result = extract_rules_with_keywords(transcript)
        self.assertEqual(result.stop_loss.clarity, "missing")
        self.assertIn("Stop loss", result.missing_information)
        self.assertIn("No exact stop loss", result.failure_reasons)

    def test_exact_risk_management(self):
        transcript = (
            "Risk 1% of your account per trade. "
            "Position size is based on risk per trade divided by stop loss distance. "
            "Maximum 3 trades at a time."
        )
        result = extract_rules_with_keywords(transcript)
        self.assertEqual(result.risk_management.clarity, "exact")
        self.assertGreaterEqual(result.risk_quality_score, 50)


def load_leaderboard() -> List[Dict[str, Any]]:
    """Load and rank saved strategy reports."""
    reports_dir = "reports"

    if not os.path.exists(reports_dir):
        return []

    leaderboard = []

    for filename in os.listdir(reports_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(reports_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)

            leaderboard.append(
                {
                    "file": filename,
                    "strategy": data.get("strategy_name", "Unknown"),
                    "score": data.get("coding_readiness_score", 0),
                    "missing": len(data.get("missing_information", [])),
                    "warning": str(data.get("scam_or_cherry_pick_warning", "Unknown"))[
                        :160
                    ],
                }
            )

        except Exception:
            continue

    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    return leaderboard


def print_leaderboard() -> None:
    leaderboard = load_leaderboard()

    if not leaderboard:
        print("No saved reports found.")
        return

    print("\n=== Strategy Leaderboard ===\n")

    for index, item in enumerate(leaderboard, start=1):
        print(f"{index}. {item['strategy']}")
        print(f"   Score: {item['score']}/100")
        print(f"   Missing Rules: {item['missing']}")
        print(f"   File: {item['file']}")
        print()


def export_leaderboard_csv(output_file: str = "leaderboard.csv") -> str:
    leaderboard = load_leaderboard()

    if not leaderboard:
        raise StrategyValidatorError("No saved reports found to export.")

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "strategy", "score", "missing", "file", "warning"],
        )

        writer.writeheader()

        for index, item in enumerate(leaderboard, start=1):
            writer.writerow(
                {
                    "rank": index,
                    "strategy": item["strategy"],
                    "score": item["score"],
                    "missing": item["missing"],
                    "file": item["file"],
                    "warning": item["warning"],
                }
            )

    return output_file


def run_batch(batch_file: str, save: bool = False, summary: bool = False) -> int:
    if not os.path.exists(batch_file):
        print(f"Batch file not found: {batch_file}", file=sys.stderr)
        return 1

    with open(batch_file, "r", encoding="utf-8") as file:
        urls = [line.strip() for line in file if line.strip()]

    print(f"Scanning {len(urls)} videos...\n")
    for url in urls:
        try:
            result = validate_strategy_core(url, save=save)

            if summary:
                print(f"URL: {url}")
                print(f"Strategy: {result.strategy_name or 'Unknown'}")
                print(f"Coding Score: {result.coding_readiness_score}/100")
                print(f"Missing Rules: {len(result.missing_information)}")
                print(f"Pine Script Ready: {result.pine_script_ready}")
                print(f"Confidence Score: {result.confidence_score}/100")
                print()
        except Exception as exc:
            print(f"Failed: {url}")
            print(f"Reason: {exc}")
            print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AI YouTube Strategy Validator")

    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--url", help="YouTube URL or video ID")
    parser.add_argument("--transcript", help="Transcript text")
    parser.add_argument("--save", action="store_true", help="Save result")
    parser.add_argument("--summary", action="store_true", help="Show summary")
    parser.add_argument("--leaderboard", action="store_true", help="Show leaderboard")
    parser.add_argument("--export-csv", action="store_true", help="Export leaderboard")
    parser.add_argument("--batch", help="Batch file with URLs")

    parser.add_argument(
        "--markdown", action="store_true", help="Export analysis as Markdown report"
    )
    args = parser.parse_args()

    if args.test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            TestYouTubeStrategyValidator
        )

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return 0 if result.wasSuccessful() else 1

    if args.export_csv:
        try:
            output_file = export_leaderboard_csv()
            print(f"Exported leaderboard to: {output_file}")
            return 0

        except StrategyValidatorError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.leaderboard:
        print_leaderboard()
        return 0

    if args.batch:
        return run_batch(
            args.batch,
            save=args.save,
            summary=args.summary,
        )

    if not args.url:
        parser.print_help()
        return 0

    try:
        result = validate_strategy_core(
            args.url,
            args.transcript,
            save=args.save,
        )

        video_id = extract_video_id(args.url)

        if args.markdown:
            markdown_file = generate_markdown_report(video_id, result)
            print(f"\nMarkdown report saved to: {markdown_file}")

        if args.summary:
            print(f"Strategy: {result.strategy_name or 'Unknown'}")
            print(f"Strategy Type: {result.strategy_type or 'Unknown'}")
            print(f"Market: {result.market or 'Unknown'}")
            print(f"Timeframe: {result.timeframe or 'Unknown'}")

            print("\nIndicators:")

            if result.indicators:
                for indicator in result.indicators:
                    if isinstance(indicator, dict):
                        print(f"  - {indicator.get('rule', indicator)}")
                    else:
                        print(f"  - {indicator}")
            else:
                print("  None detected")

            print(f"\nCoding Score: {result.coding_readiness_score}/100")
            print_score_bar(result.coding_readiness_score)

            print(f"Confidence Score: {result.confidence_score}/100")
            print_score_bar(result.confidence_score)

            print(f"Pine Script Ready: {result.pine_script_ready}")

            print(f"\nMissing Rules ({len(result.missing_information)}):")

            if result.missing_information:
                for item in result.missing_information:
                    print(f"  - {item}")
            else:
                print("  None")

            print("\nFailure Reasons:")

            if result.failure_reasons:
                for reason in result.failure_reasons:
                    print(f"  - {reason}")
            else:
                print("  None")

            print("\nSubjective Terms:")

            if result.subjective_terms:
                for term in result.subjective_terms:
                    print(f"  - {term}")
            else:
                print("  None")

            print("\nWarning:")
            print(result.scam_or_cherry_pick_warning)

        else:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

        return 0

    except StrategyValidatorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
