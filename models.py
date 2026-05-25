from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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
        default_factory=lambda: StrategyRule("session_filter", "Not specified", "missing")
    )
    backtest_evidence: StrategyRule = field(
        default_factory=lambda: StrategyRule("backtest_evidence", "None mentioned", "missing")
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
