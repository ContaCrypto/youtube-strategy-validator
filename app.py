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
from typing import Any, Dict, List, Optional

from models import StrategyRule, StrategyExtraction
from extractors import (
    find_indicators,
    detect_timeframe,
    detect_market,
    detect_strategy_type,
    detect_promotional_claims,
    detect_session_filter,
    detect_backtest_evidence,
    _SUBJECTIVE_TERMS,
    _EXACT_SIGNALS,
)


try:
    import db as _db

    _db.init_db()
except Exception as _db_init_err:
    logging.warning("DB init failed: %s", _db_init_err)
    _db = None



# Optional dependencies. The app must not crash if these are missing.
try:
    from fastapi import FastAPI, HTTPException, Request, Form
    from fastapi.responses import HTMLResponse, Response
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

from analyzer import analyze_strategy, extract_rules_with_keywords
from reports import save_report, generate_markdown_report, build_markdown_from_dict


class StrategyValidatorError(Exception):
    """Base app error."""


class TranscriptUnavailableError(StrategyValidatorError):
    """Raised when transcript extraction is unavailable or fails."""


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

    @app.get("/analysis/{analysis_id}/markdown")
    def analysis_markdown_export(analysis_id: int):
        if _db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")
        rec = _db.get_analysis(analysis_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        verdict = compute_verdict(rec)
        md = build_markdown_from_dict(rec, verdict)
        filename = f"strategy_{analysis_id}.md"
        return Response(
            content=md,
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/analysis/{analysis_id}/json")
    def analysis_json_export(analysis_id: int):
        if _db is None:
            raise HTTPException(status_code=503, detail="Database unavailable")
        rec = _db.get_analysis(analysis_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        filename = f"strategy_{analysis_id}.json"
        return Response(
            content=json.dumps(rec, indent=2, ensure_ascii=False, default=str),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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

    def test_markdown_export_contains_required_sections(self):
        rec = {
            "id": 1,
            "youtube_url": "https://youtube.com/watch?v=test123",
            "strategy_name": "Test RSI Strategy",
            "strategy_type": "indicator_strategy",
            "market": "crypto",
            "timeframe": "1h",
            "verdict": "Semi-Codable",
            "automation_difficulty": "Medium",
            "warning": "Medium warning.",
            "summary": "A test strategy summary.",
            "coding_readiness_score": 60,
            "confidence_score": 55,
            "pine_script_ready": False,
            "missing_information": ["Stop loss", "Take profit"],
            "failure_reasons": ["No exact stop loss"],
            "subjective_terms": ["confirmation"],
            "promotional_claims": [],
            "entry_quality_score": 70,
            "exit_quality_score": 30,
            "risk_quality_score": 50,
            "automation_feasibility_score": 50,
            "hype_risk_score": 5,
            "backtest_evidence_score": 0,
            "formalization_score": 40,
            "created_at": "2026-01-01T00:00:00",
        }
        md = build_markdown_from_dict(rec)
        self.assertIn("# Strategy Analysis Report", md)
        self.assertIn("Test RSI Strategy", md)
        self.assertIn("Coding Readiness", md)
        self.assertIn("Missing Information", md)
        self.assertIn("Stop loss", md)
        self.assertIn("Disclaimer", md)
        self.assertIn("Semi-Codable", md)

    def test_markdown_export_includes_promotional_claims(self):
        rec = {
            "id": 2,
            "youtube_url": "https://youtube.com/watch?v=scam123",
            "strategy_name": None,
            "strategy_type": None,
            "market": None,
            "timeframe": None,
            "verdict": "Likely Scam",
            "automation_difficulty": "Impossible",
            "warning": "High warning.",
            "summary": "",
            "coding_readiness_score": 5,
            "confidence_score": 5,
            "pine_script_ready": False,
            "missing_information": [],
            "failure_reasons": ["Contains promotional claims"],
            "subjective_terms": [],
            "promotional_claims": ["holy grail", "financial freedom"],
            "entry_quality_score": 0,
            "exit_quality_score": 0,
            "risk_quality_score": 0,
            "automation_feasibility_score": 0,
            "hype_risk_score": 80,
            "backtest_evidence_score": 0,
            "formalization_score": 0,
            "created_at": "2026-01-01T00:00:00",
        }
        md = build_markdown_from_dict(rec)
        self.assertIn("Promotional", md)
        self.assertIn("holy grail", md)
        self.assertIn("financial freedom", md)
        self.assertIn("Disclaimer", md)


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
