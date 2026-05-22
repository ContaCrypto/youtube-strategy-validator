# YouTube Strategy Validator

AI-powered trading strategy analyzer for YouTube videos.

This tool evaluates whether a trading strategy shown in a video is:

- actually testable
- automatable
- measurable
- overly subjective
- potentially promotional or misleading

The system extracts strategy logic from transcripts and scores the strategy based on clarity, structure, and automation feasibility.

---

# Features

## Core Analysis

- Strategy extraction
- Entry/exit detection
- Stop loss detection
- Take profit detection
- Risk management detection
- Position sizing analysis
- Session filter detection
- Backtest evidence detection

## AI Scoring System

- Coding readiness score
- Confidence score
- Entry quality score
- Exit quality score
- Risk quality score
- Automation feasibility score
- Hype risk score
- Backtest evidence score
- Formalization score

## Verdict Engine

Possible verdicts:

- Likely Scam
- Too Vague To Automate
- Semi-Codable
- Fully Codable

## Web Features

- FastAPI web interface
- SQLite analysis history
- Saved analysis pages
- Markdown report generation
- JSON API

---

# Installation

## Clone repository

```bash
git clone https://github.com/ContaCrypto/youtube-strategy-validator.git
cd youtube-strategy-validator