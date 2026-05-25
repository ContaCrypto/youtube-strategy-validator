from __future__ import annotations

import json
import os
import re
from typing import Any, List, Optional

from models import StrategyExtraction, StrategyRule
from extractors import (
    _EXACT_SIGNALS,
    _SUBJECTIVE_TERMS,
    _VISUAL_SIGNAL_ENTRY_PHRASES,
    _VISUAL_SIGNAL_EXIT_PHRASES,
    detect_backtest_evidence,
    detect_market,
    detect_promotional_claims,
    detect_session_filter,
    detect_strategy_type,
    detect_timeframe,
    find_indicators,
)

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None


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
        has_visual_entry = any(phrase in sl for phrase in _VISUAL_SIGNAL_ENTRY_PHRASES)
        has_visual_exit = any(phrase in sl for phrase in _VISUAL_SIGNAL_EXIT_PHRASES)

        if has_subj:
            clarity = "subjective"
        elif has_exact:
            clarity = "exact"
        else:
            clarity = "vague"

        # Long entry: classic keywords + standalone "enter" verb + visual entry signals
        if (
            any(w in sl for w in ["buy", "long", "enter long", "go long"])
            or re.search(r"\benter\b", sl)
            or has_visual_entry
        ):
            entry_rules_long.append(StrategyRule("long_entry", s, clarity))

        # Short entry: classic keywords only (visual exit phrases imply exit, not always short)
        if re.search(r"\b(short|sell short|enter short|go short)\b", sl):
            entry_rules_short.append(StrategyRule("short_entry", s, clarity))

        # Exit: classic keywords + visual exit signal phrases
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
        ) or has_visual_exit:
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

    # ── Granular quality sub-scores (computed first to allow cross-capping) ─────
    all_entries = entry_rules_long + entry_rules_short
    exact_entries = [r for r in all_entries if r.clarity == "exact"]

    # Entry quality: timeframe/indicator bonus only awarded when entry rules exist.
    # Purely-subjective entries score lower than vague/exact ones.
    entry_q = 0
    if all_entries:
        non_subj_entries = [r for r in all_entries if r.clarity != "subjective"]
        if non_subj_entries:
            entry_q += 40
            entry_q += int(30 * len(exact_entries) / len(all_entries))
        else:
            # Some entry language exists but it's all subjective — not codable
            entry_q += 15
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

    # Formalization: based on clarity of formal rules + entry quality blend
    # No bonus for bare indicators — rules must be formalizable to count
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
    if all_entries:
        formal_base = (formal_base + entry_q) // 2
    formalization = max(0, min(100, formal_base - len(subjective_terms) * 3))

    # Automation feasibility incorporates all four quality dimensions
    auto_feas = max(
        0,
        min(
            100,
            int((entry_q + exit_q + risk_q + formalization) / 4) - hype_risk // 3,
        ),
    )

    # ── Coding readiness score ─────────────────────────────────────────────────
    score = 0
    if timeframe:
        score += 10
    if indicators:
        score += 10
    if entry_rules_long or entry_rules_short:
        score += 15
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

    # Cap coding readiness when fundamental sub-scores are missing
    if entry_q == 0:
        score = min(score, 30)
    if exit_q == 0:
        score = min(score, 35)
    if risk_q == 0:
        score = min(score, 40)
    if formalization == 0:
        score = min(score, 25)

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
    # Ensure a non-zero floor for strategies with any positive signal and no promo hype
    if score > 0 and not promotional_claims:
        confidence_score = max(5, confidence_score)
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
