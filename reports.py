from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from models import StrategyExtraction, StrategyRule


def print_score_bar(score: int) -> None:
    filled = int(score / 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    print(f"[{bar}] {score}/100")


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


def build_markdown_from_dict(
    rec: Dict[str, Any],
    verdict: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a rich Markdown export string from a DB analysis record dict."""

    def bullet(items: Any, empty: str = "None") -> str:
        if not items:
            return f"- {empty}"
        lines = []
        for item in items:
            if isinstance(item, dict):
                text = item.get("rule", str(item))
                clarity = item.get("clarity", "")
                lines.append(
                    f"- {text}" + (f" *(clarity: {clarity})*" if clarity else "")
                )
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)

    def bar(score: int) -> str:
        score = max(0, min(100, int(score or 0)))
        filled = round(score / 10)
        return f"`{'█' * filled}{'░' * (10 - filled)}` {score}/100"

    v = verdict or {}
    name = rec.get("strategy_name") or "Unknown Strategy"
    created = str(rec.get("created_at", ""))[:10]
    url = rec.get("youtube_url", "")
    verdict_text = rec.get("verdict") or v.get("verdict", "Unknown")
    difficulty = rec.get("automation_difficulty") or v.get("difficulty", "Unknown")
    why = v.get("why", "")
    next_steps: list = v.get("next_steps") or []

    cr = rec.get("coding_readiness_score") or 0
    cf = rec.get("confidence_score") or 0
    pine = rec.get("pine_script_ready", False)

    promo = rec.get("promotional_claims") or []

    md = f"""# Strategy Analysis Report

## {name}

> **Source:** {url}
> **Analyzed:** {created}

---

## Verdict

| Field | Value |
|---|---|
| **Overall Verdict** | {verdict_text} |
| **Automation Difficulty** | {difficulty} |
| **Pine Script Ready** | {"Yes ✓" if pine else "No ✗"} |

{why}

---

## Strategy Details

| Field | Value |
|---|---|
| **Type** | {rec.get("strategy_type") or "Unknown"} |
| **Market** | {rec.get("market") or "Unknown"} |
| **Timeframe** | {rec.get("timeframe") or "Unknown"} |

---

## Scores

| Score | Value |
|---|---|
| Coding Readiness | {bar(cr)} |
| Confidence | {bar(cf)} |
| Entry Quality | {bar(rec.get("entry_quality_score"))} |
| Exit Quality | {bar(rec.get("exit_quality_score"))} |
| Risk Quality | {bar(rec.get("risk_quality_score"))} |
| Automation Feasibility | {bar(rec.get("automation_feasibility_score"))} |
| Hype / Scam Risk | {bar(rec.get("hype_risk_score"))} |
| Backtest Evidence | {bar(rec.get("backtest_evidence_score"))} |
| Formalization | {bar(rec.get("formalization_score"))} |

---

## Warning

{rec.get("warning") or "None"}

---

## Summary

{rec.get("summary") or "No summary available."}

---

## Missing Information

{bullet(rec.get("missing_information"))}

---

## Failure Reasons

{bullet(rec.get("failure_reasons"))}

---

## Subjective / Unmeasurable Terms

{bullet(rec.get("subjective_terms"))}

"""

    if promo:
        md += f"## Promotional / Hype Claims\n\n{bullet(promo)}\n\n---\n\n"

    if next_steps:
        md += "## Recommended Next Steps\n\n"
        for i, step in enumerate(next_steps, 1):
            md += f"{i}. {step}\n"
        md += "\n---\n\n"

    md += f"""## Disclaimer

This report was automatically generated by AI YouTube Strategy Validator.
It is for educational and research purposes only and does not constitute financial advice.
Always validate any trading strategy independently before risking real capital.

---
*Generated: {created}*
"""
    return md
