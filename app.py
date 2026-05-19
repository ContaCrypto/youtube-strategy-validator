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
import os
import re
import sys
import unittest
from datetime import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Optional dependencies. The app must not crash if these are missing.
try:
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None
    HTTPException = None

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


def extract_rules_with_keywords(text: str) -> StrategyExtraction:
    """
    Dependency-free fallback validator.
    This is not as strong as an LLM, but it prevents the app from crashing in restricted environments.
    """
    lowered = text.lower()
    indicators = find_indicators(text)
    timeframe = detect_timeframe(text)
    market = detect_market(text)
    strategy_type = detect_strategy_type(text)

    subjective_terms = sorted(
        {
            term
            for term in [
                "confirmation",
                "momentum",
                "market structure",
                "strong candle",
                "weak candle",
                "smart money",
                "liquidity grab",
                "clean setup",
                "good entry",
                "wait for reaction",
            ]
            if term in lowered
        }
    )

    entry_rules_long: List[StrategyRule] = []
    entry_rules_short: List[StrategyRule] = []
    exit_rules: List[StrategyRule] = []

    sentence_split = re.split(r"(?<=[.!?])\s+", text.strip())

    for sentence in sentence_split:
        s = sentence.strip()
        sl = s.lower()

        if not s:
            continue

        clarity = (
            "subjective" if any(term in sl for term in subjective_terms) else "vague"
        )

        if any(word in sl for word in ["buy", "long", "enter long"]):
            if any(
                word in sl
                for word in [
                    "cross",
                    "above",
                    "below",
                    "greater than",
                    "less than",
                    "%",
                ]
            ):
                clarity = "exact"
            entry_rules_long.append(StrategyRule("long_entry", s, clarity))

        if any(word in sl for word in ["sell short", "short", "enter short"]):
            if any(
                word in sl
                for word in [
                    "cross",
                    "above",
                    "below",
                    "greater than",
                    "less than",
                    "%",
                ]
            ):
                clarity = "exact"
            entry_rules_short.append(StrategyRule("short_entry", s, clarity))

        if any(
            word in sl
            for word in ["exit", "close", "sell when", "take profit", "stop loss"]
        ):
            if any(
                word in sl
                for word in [
                    "cross",
                    "above",
                    "below",
                    "%",
                    "atr",
                    "risk reward",
                    "r:r",
                ]
            ):
                clarity = "exact"
            exit_rules.append(StrategyRule("exit", s, clarity))

    stop_loss = StrategyRule("stop_loss", "Missing", "missing")
    stop_match = re.search(
        r"(stop loss|stop-loss|sl)[^.!?]{0,80}", text, flags=re.IGNORECASE
    )
    if stop_match:
        stop_text = stop_match.group(0).strip()
        stop_loss = StrategyRule(
            "stop_loss",
            stop_text,
            "exact" if re.search(r"\d|atr|%", stop_text.lower()) else "vague",
        )

    take_profit = StrategyRule("take_profit", "Missing", "missing")
    tp_match = re.search(
        r"(take profit|take-profit|tp|target)[^.!?]{0,80}", text, flags=re.IGNORECASE
    )
    if tp_match:
        tp_text = tp_match.group(0).strip()
        take_profit = StrategyRule(
            "take_profit",
            tp_text,
            "exact" if re.search(r"\d|%|risk reward|r:r", tp_text.lower()) else "vague",
        )

    risk_management = StrategyRule("risk_management", "Missing", "missing")
    risk_match = re.search(
        r"(risk|risk management)[^.!?]{0,100}", text, flags=re.IGNORECASE
    )
    if risk_match:
        risk_text = risk_match.group(0).strip()
        risk_management = StrategyRule(
            "risk_management",
            risk_text,
            "exact" if re.search(r"\d|%", risk_text) else "vague",
        )

    position_sizing = StrategyRule("position_sizing", "Missing", "missing")
    size_match = re.search(
        r"(position size|position sizing|lot size|size)[^.!?]{0,100}",
        text,
        flags=re.IGNORECASE,
    )
    if size_match:
        size_text = size_match.group(0).strip()
        position_sizing = StrategyRule(
            "position_sizing",
            size_text,
            "exact" if re.search(r"\d|%", size_text) else "vague",
        )

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

    score = 0

    if timeframe:
        score += 10

    if indicators:
        score += 15

    if entry_rules_long or entry_rules_short:
        score += 20

    if exit_rules:
        score += 20

    if stop_loss.clarity == "exact":
        score += 10

    if take_profit.clarity == "exact":
        score += 10

    if risk_management.clarity == "exact":
        score += 10

    if position_sizing.clarity == "exact":
        score += 5

    score -= len(subjective_terms) * 5
    score = max(0, min(100, score))

    failure_reasons = []

    if not timeframe:
        failure_reasons.append("Missing timeframe")

    if not indicators:
        failure_reasons.append("No indicators detected")

    if not entry_rules_long and not entry_rules_short:
        failure_reasons.append("No clear entry rules")

    if not exit_rules:
        failure_reasons.append("No clear exit rules")

    if subjective_terms:
        failure_reasons.append("Contains subjective language")

    if stop_loss.clarity != "exact":
        failure_reasons.append("No exact stop loss")

    if take_profit.clarity != "exact":
        failure_reasons.append("No exact take profit")

    if risk_management.clarity != "exact":
        failure_reasons.append("No exact risk management")

    if position_sizing.clarity != "exact":
        failure_reasons.append("No exact position sizing")

    repainting_risk = "Unknown"

    if any(word in lowered for word in ["repaint", "repainting"]):
        repainting_risk = "Mentioned in transcript. Needs manual review."
    elif any(
        word in lowered for word in ["pivot", "zigzag", "fractal", "future candle"]
    ):
        repainting_risk = "Possible repainting risk due to indicator type."

    warning = "Low warning. Rules look partly testable."

    if score < 50:
        warning = "High warning. Strategy is not code-ready and may be cherry-picked or too vague."
    elif score < 75:
        warning = "Medium warning. Strategy has testable parts but important assumptions are missing."

    pine_script_ready = (
        len(missing) <= 2
        and len(subjective_terms) == 0
        and stop_loss.clarity == "exact"
        and take_profit.clarity == "exact"
        and bool(entry_rules_long or entry_rules_short)
    )

    confidence_score = score

    if subjective_terms:
        confidence_score -= 20

    if len(failure_reasons) >= 5:
        confidence_score -= 20

    if pine_script_ready:
        confidence_score += 15

    confidence_score = max(0, min(100, confidence_score))

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
        repainting_risk=repainting_risk,
        missing_information=missing,
        subjective_terms=subjective_terms,
        coding_readiness_score=score,
        scam_or_cherry_pick_warning=warning,
        summary="Fallback heuristic analysis used. Install OpenAI package and set OPENAI_API_KEY for stricter AI extraction.",
        failure_reasons=failure_reasons,
        confidence_score=confidence_score,
        pine_script_ready=pine_script_ready,
    )


def analyze_strategy_with_openai(transcript_text: str) -> Optional[StrategyExtraction]:
    """Use OpenAI if the dependency and API key are available. Otherwise return None."""
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return None

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    schema_prompt = """
You are a strict trading strategy auditor.
Extract only explicit, testable trading rules from the transcript.
Do not invent missing rules.
Mark vague language as vague or subjective.
Return valid JSON with these keys:
strategy_name, market, timeframe, indicators, entry_rules_long, entry_rules_short,
exit_rules, stop_loss, take_profit, risk_management, position_sizing,
repainting_risk, missing_information, subjective_terms, coding_readiness_score,
scam_or_cherry_pick_warning, summary, failure_reasons, strategy_type, pine_script_ready, confidence_score.
Each rule object must contain category, rule, clarity.
"""

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
        repainting_risk=data.get("repainting_risk", "Unknown"),
        missing_information=data.get("missing_information", []),
        subjective_terms=data.get("subjective_terms", []),
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


# Optional FastAPI layer. This is only created when FastAPI and Pydantic exist.
if FastAPI is not None and BaseModel is not None:
    app = FastAPI(title="AI YouTube Strategy Validator")

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

    @app.get("/")
    def root():
        return {
            "app": "AI YouTube Strategy Validator",
            "status": "running",
            "endpoint": "POST /validate",
            "fastapi_enabled": True,
        }

else:
    app = None


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


def print_score_bar(score: int) -> None:
    filled = int(score / 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    print(f"[{bar}] {score}/100")


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
